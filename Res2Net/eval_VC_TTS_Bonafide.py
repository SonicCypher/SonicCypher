import os
import sys
import torch
import torchaudio
import numpy as np
from tqdm.contrib import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_curve
import speechbrain as sb
from speechbrain.utils.logger import get_logger
from Res2Net.models.resnet_models import se_res2net50_v1b
from Preprocessing.ASVSpoof_Final.perpare_Categories_csv import prepare_ASV_verification

# Global device
run_opts = {"device": "cuda" if torch.cuda.is_available() else "cpu"}

# Compute embeddings from waveforms
def compute_embedding(wavs, model):
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
    return embeddings.squeeze(1)

# Enrollment: compute mean embedding per speaker
def compute_mean_enrol_embeddings(data_loader, model):
    speaker_embeddings = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing enrollment embeddings"):
            wavs = batch["sig"].to(run_opts["device"])
            spk_ids = batch["spk_id"]

            emb = compute_embedding(wavs, model).unsqueeze(1)

            for i, spk_id in enumerate(spk_ids):
                if spk_id not in speaker_embeddings:
                    speaker_embeddings[spk_id] = []
                speaker_embeddings[spk_id].append(emb[i].detach().clone())

    mean_embeddings = {
        spk_id: torch.stack(emb_list).mean(dim=0)
        for spk_id, emb_list in speaker_embeddings.items()
    }
    return mean_embeddings

# Test: compute per-utterance embeddings
def compute_test_embeddings(data_loader, model):
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

# Read evaluation file
def read_eval_file(eval_file_path):
    eval_entries = []
    with open(eval_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) != 3:
                print(f"[ERROR] Invalid line in eval file: {line.strip()}")
                continue
            label = parts[0]
            claimed_spk_id = parts[1].split(":")[0]
            test_utt_id = parts[2].strip()
            eval_entries.append((label, claimed_spk_id, test_utt_id))
    return eval_entries

# Calculate EER from labels and scores
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.abs(fnr - fpr).argmin()
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer * 100

# Compute scores and EER
def compute_cosine_scores(eval_entries, enrol_dict, test_dict, score_output_path=None):
    scores = []
    labels = []

    with open(score_output_path, "w") if score_output_path else open(os.devnull, "w") as f:
        for label, spk_id, test_utt_id in eval_entries:
            if spk_id not in enrol_dict:
                print(f"[WARN] Missing enrollment for speaker {spk_id}")
                continue
            if test_utt_id not in test_dict:
                print(f"[WARN] Missing test embedding for {test_utt_id}")
                continue

            emb_enrol = enrol_dict[spk_id]
            emb_test = test_dict[test_utt_id]

            emb_enrol = F.normalize(emb_enrol, dim=1)
            emb_test = F.normalize(emb_test, dim=1)

            # print(f"[DEBUG] enrol: {emb_enrol.shape}, test: {emb_test.shape}")
            score = F.cosine_similarity(emb_enrol, emb_test,dim=1)
            binary_label = 1 if label == "bonafide" else 0

            labels.append(binary_label)
            scores.append(score.cpu().item())

            text_utt_id = test_utt_id.split("/")[-1]
            f.write(f"{spk_id} {text_utt_id} _ {label} {score.item():.4f} {binary_label}\n")

    eer = compute_eer(labels, scores)
    print(f"EER = {eer:.2f}%")

# Run verification pipeline
def run_verification_pipeline(data_folder, save_folder_csv, verification_pairs_file, model_ckpt_path, score_output_path):
    prepare_ASV_verification(data_folder, save_folder_csv, verification_pairs_file)
    enrol_loader, test_loader = dataio_prep(data_folder, save_folder_csv)

    model = se_res2net50_v1b(num_classes=1211)
    ckpt = torch.load(model_ckpt_path, map_location=run_opts["device"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(run_opts["device"])

    enrol_dict = compute_mean_enrol_embeddings(enrol_loader, model)
    test_dict = compute_test_embeddings(test_loader, model)
    eval_entries = read_eval_file(verification_pairs_file)

    compute_cosine_scores(eval_entries, enrol_dict, test_dict, score_output_path)

# Data preparation
def dataio_prep(data_folder, save_folder_csv):
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(save_folder_csv, "enrol.csv"),
        replacements={"data_root": data_folder}
    ).filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(save_folder_csv, "test.csv"),
        replacements={"data_root": data_folder}
    ).filtered_sorted(sort_key="duration")

    datasets = [enrol_data, test_data]

    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(wav, num_frames=num_frames, frame_offset=start)
        return sig.transpose(0, 1).squeeze(1)

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id"])

    def collate_fn(batch):
        ids = [item["id"] for item in batch]
        spk_ids = [item["spk_id"] for item in batch]
        sigs = [item["sig"] for item in batch]
        sigs_padded = pad_sequence(sigs, batch_first=True)
        return {"id": ids, "spk_id": spk_ids, "sig": sigs_padded}

    enrol_loader = DataLoader(enrol_data, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
    return enrol_loader, test_loader

# Main
if __name__ == "__main__":
    logger = get_logger(__name__)

    model_ckpt = "/home/hansini/Campus/FYP/SonicCypher/Trained_Models/model_epoch_148.pth"
    root_folder = "/home/hansini/Campus/FYP/LA/ASVspoof2019_LA_eval"
    output_dir = "/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/output"
    print("\n=======================================\n")
    print("VC")
    run_verification_pipeline(
        data_folder=root_folder,
        save_folder_csv="/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/output/vc",
        verification_pairs_file="/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/verify_vc_eval_gi.txt",
        model_ckpt_path=model_ckpt,
        score_output_path=f"{output_dir}/cosine_scores_vc.txt"
    )

    print("\n=======================================\n")
    print("TTS")

    run_verification_pipeline(
        data_folder=root_folder,
        save_folder_csv="/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/output/tts",
        verification_pairs_file="/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/verify_tts_eval_gi.txt",
        model_ckpt_path=model_ckpt,
        score_output_path=f"{output_dir}/cosine_scores_tts.txt"
    )

    print("\n=======================================\n")
    print("All")

    run_verification_pipeline(
        data_folder=root_folder,
        save_folder_csv="/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/output/All",
        verification_pairs_file="/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/verify_All_eval_gi.txt",
        model_ckpt_path=model_ckpt,
        score_output_path=f"{output_dir}/cosine_scores_All.txt"
    )
