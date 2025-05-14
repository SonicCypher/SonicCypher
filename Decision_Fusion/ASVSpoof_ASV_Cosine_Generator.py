import os
import sys
import torch
import torchaudio
from tqdm.contrib import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import speechbrain as sb
from speechbrain.utils.logger import get_logger
from Res2Net.models.resnet_models import se_res2net50_v1b
from Preprocessing.ASVSpoof_Eval.prepare_ASVSpoof_csv_file import prepare_ASV_verification


# Compute embeddings from waveforms
def compute_embedding(wavs,model):
    with torch.no_grad():
        lengths = [len(wav) for wav in wavs]
        max_length = max(lengths)
        wav_lens = torch.tensor([length / max_length for length in lengths], dtype=torch.float32).to(run_opts["device"])

        feats = sb.lobes.features.MFCC(n_mfcc=24, n_mels=44, deltas=False, context=False)
        normalization = sb.processing.features.InputNormalization(norm_type="sentence", std_norm=False)

        features = feats(wavs)
        feats = normalization(features, wav_lens)

        if feats.dim() == 3:
            feats = feats.unsqueeze(1)

        embeddings = model.extract(feats)
    embeddings = embeddings.squeeze(1)
    return embeddings

# Enrollment: compute mean embedding per speaker
def compute_mean_enrol_embeddings(data_loader,model):
    speaker_embeddings = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing enrollment embeddings"):
            wavs = batch["sig"].to(run_opts["device"])
            spk_ids = batch["spk_id"]

            emb = compute_embedding(wavs,model).unsqueeze(1)

            for i, spk_id in enumerate(spk_ids):
                if spk_id not in speaker_embeddings:
                    speaker_embeddings[spk_id] = []
                speaker_embeddings[spk_id].append(emb[i].detach().clone())

    # Average per speaker
    mean_embeddings = {
        spk_id: torch.stack(emb_list).mean(dim=0)
        for spk_id, emb_list in speaker_embeddings.items()
    }
    return mean_embeddings

# Test: compute per-utterance embeddings
def compute_test_embeddings(data_loader,model):
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing test embeddings"):
            seg_ids = batch["id"]
            wavs = batch["sig"]

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs = wavs.to(run_opts["device"])

            emb = compute_embedding(wavs,model).unsqueeze(1)
        
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()

    return embedding_dict

def read_eval_file(eval_file_path):
    eval_entries = []
    with open(eval_file_path, "r") as f:
        for line in f:
            label, spk_section, utt_section = line.strip().split(", ")
            claimed_spk_id = spk_section.split(":")[0]
            test_utt_id = utt_section.strip()
            eval_entries.append((label, claimed_spk_id, test_utt_id))
    return eval_entries

def compute_cosine_scores(eval_entries, enrol_dict, test_dict, output_path):
    with open(output_path, "w") as f:
        for label, spk_id, test_utt_id in eval_entries:
            if spk_id not in enrol_dict:
                print(f"[WARN] Missing enrollment for speaker {spk_id}")
                continue
            if test_utt_id not in test_dict:
                print(f"[WARN] Missing test embedding for {test_utt_id}")
                continue

            emb_enrol = enrol_dict[spk_id]
            emb_test = test_dict[test_utt_id]

            # Normalize
            emb_enrol = F.normalize(emb_enrol, dim=1)
            emb_test = F.normalize(emb_test, dim=1)

            cosine_score = F.cosine_similarity(emb_enrol, emb_test,dim=1)
            threshold = 0.67
            for i, score in enumerate(cosine_score):
                text_utt_id = test_utt_id.split("/")[-1]
                binary_result = 1 if score.item() > threshold else 0
                f.write(f"{spk_id} {text_utt_id} _ {label} {score.item():.4f} {binary_result}\n")


def run_verification_pipeline(
    data_folder,
    save_folder_csv,
    verification_pairs_file,
    model_ckpt_path,
    score_output_path,
):
    # Prepare CSV files from pair list
    prepare_ASV_verification(data_folder, save_folder_csv, verification_pairs_file)

    # Prepare dataloaders
    enrol_dataloader, test_dataloader = dataio_prep(data_folder, save_folder_csv)

    # Load model
    model = se_res2net50_v1b(num_classes=1211)
    last_best_model = torch.load(model_ckpt_path, map_location=run_opts["device"])
    model.load_state_dict(last_best_model["model_state_dict"])
    model.eval()
    model.to(run_opts["device"])

    # Compute embeddings
    enrol_dict = compute_mean_enrol_embeddings(enrol_dataloader,model)
    test_dict = compute_test_embeddings(test_dataloader,model)
    eval_entries = read_eval_file(verification_pairs_file)

    # Score and save
    compute_cosine_scores(eval_entries, enrol_dict, test_dict, score_output_path)
    print(f"Cosine scores saved to: {score_output_path}")


# Data loading and preparation
def dataio_prep(data_folder, save_folder_csv):
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(save_folder_csv, "enrol.csv"),
        replacements={"data_root": data_folder},
    ).filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(save_folder_csv, "test.csv"),
        replacements={"data_root": data_folder},
    ).filtered_sorted(sort_key="duration")

    datasets = [enrol_data, test_data]

    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(wav, num_frames=num_frames, frame_offset=start)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id"])

    def collate_fn(batch):
        ids = [item["id"] for item in batch]
        spk_ids = [item["spk_id"] for item in batch]
        sigs = [item["sig"] for item in batch]
        sigs_padded = pad_sequence(sigs, batch_first=True)
        return {"id": ids, "spk_id": spk_ids, "sig": sigs_padded}

    enrol_dataloader = DataLoader(enrol_data, batch_size=8, shuffle=False, num_workers=6, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=6, collate_fn=collate_fn)

    return enrol_dataloader, test_dataloader

# Main execution
if __name__ == "__main__":
    logger = get_logger(__name__)
    run_opts = {
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    saved_model_path = "/home/hansini/Campus/FYP/SonicCypher/Trained_Models/model_epoch_10.pth"

    run_verification_pipeline(
        data_folder="/home/hansini/Campus/FYP/LA/ASVspoof2019_LA_eval",
        save_folder_csv="/home/hansini/Campus/FYP/SonicCypher/Decision_Fusion/output/eval",
        verification_pairs_file="/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Eval/verify_eval_gi.txt",
        model_ckpt_path=saved_model_path,
        score_output_path="/home/hansini/Campus/FYP/SonicCypher/Decision_Fusion/output/cosine_scores_eval.txt"
    )
     
    print("\n")
    print("=======================================")
    print("\n")

    run_verification_pipeline(
        data_folder="/home/hansini/Campus/FYP/LA/ASVspoof2019_LA_dev",
        save_folder_csv="/home/hansini/Campus/FYP/SonicCypher/Decision_Fusion/output/dev",
        verification_pairs_file="/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Eval/verify_dev_gi.txt",
        model_ckpt_path=saved_model_path,
        score_output_path="/home/hansini/Campus/FYP/SonicCypher/Decision_Fusion/output/cosine_scores_dev.txt"
    )
