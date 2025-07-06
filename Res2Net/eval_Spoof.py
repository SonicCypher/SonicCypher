import os
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm.contrib import tqdm

import speechbrain as sb
from speechbrain.utils.logger import get_logger
from speechbrain.utils.metric_stats import EER, minDCF
from torch.utils.data import DataLoader
from models.resnet_models import se_res2net50_v1b
from Preprocessing.ASVSpoof.prepare_ASV import prepare_ASV_verification
from torch.nn.utils.rnn import pad_sequence



# Compute embeddings from the waveforms
def compute_embedding(wavs):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : torch.Tensor
        torch.Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens : torch.Tensor
        torch.Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])

    Returns
    -------
    embeddings : torch.Tensor
    """
    with torch.no_grad():
        lengths = [len(wav) for wav in wavs]  # Get lengths of each waveform
        max_length = max(lengths)  # Find the maximum length in the batch
        wav_lens = torch.tensor([length / max_length for length in lengths], dtype=torch.float32).to(run_opts["device"])
        feats = sb.lobes.features.MFCC(n_mfcc=24, n_mels=44, deltas=False, context=False)
        normalization = sb.processing.features.InputNormalization(norm_type="sentence",std_norm=False)

        features = feats(wavs)
        feats = normalization(features, wav_lens)

        if feats.dim() == 3:
            feats = feats.unsqueeze(1)  # Add a channel dimension if missing

        embeddings = model.extract(feats)
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            seg_ids = batch["id"]
            wavs = batch["sig"]

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs = wavs.to(run_opts["device"])

            emb = compute_embedding(wavs).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


def get_verification_scores(veri_test):
    """Computes positive and negative scores given the verification split."""
    scores = []
    positive_scores = []
    negative_scores = []

    save_file = os.path.join('output_folder', "scores.txt")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    s_file = open(save_file, "w", encoding="utf-8")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    for i, line in enumerate(veri_test):
        lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip()
        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]

        # Compute the score for the given sentence
        score = similarity(enrol, test)[0]

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        scores.append(score)

        if lab_pair == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    s_file.close()
    return positive_scores, negative_scores


def dataio_prep():
    "Creates the dataloaders and their data processing pipelines."

    data_folder ="/home/cse/SonicCypher/ASVSpoof2019"

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path='./Preprocessing/ASVSpoof/output/enrol.csv',
        replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path='./Preprocessing/ASVSpoof/output/test.csv',
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [enrol_data, test_data]

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    def collate_fn(batch):
        ids = [item["id"] for item in batch]
        sigs = [item["sig"] for item in batch]
        sigs_padded = pad_sequence(sigs, batch_first=True)
        return {"id": ids, "sig": sigs_padded}


    # Create dataloaders
    enrol_dataloader = DataLoader(enrol_data, batch_size=8, shuffle=False, num_workers=10, collate_fn=collate_fn)

    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=10, collate_fn=collate_fn)

    return enrol_dataloader, test_dataloader


if __name__ == "__main__":
    # Logger setup
    logger = get_logger(__name__)
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # sys.path.append(os.path.dirname(current_dir))

    run_opts = {
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

    data_folder ="/home/cse/SonicCypher/ASVSpoof2019"
    save_folder_csv = "./Preprocessing/ASVSpoof/output"
    verification_pairs_file = r"/home/cse/SonicCypher/ASVSpoof2019/test_file_alternated.txt"
    print("Preparing VoxCeleb data...")

    prepare_ASV_verification(data_folder,save_folder_csv, verification_pairs_file)

    enrol_dataloader, test_dataloader = dataio_prep()

    model = se_res2net50_v1b(num_classes=1211)
    last_best_model = torch.load("checkpoints/model_epoch_31.pth", map_location=run_opts["device"])
    model.load_state_dict(last_best_model["model_state_dict"])
    model.eval()
    model.to(run_opts["device"]) 

    enrol_dict = compute_embedding_loop(enrol_dataloader)
    test_dict = compute_embedding_loop(test_dataloader)

    # Compute the EER
    print("Computing EER..")
    # Reading standard verification split
    with open(verification_pairs_file, encoding="utf-8") as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores, negative_scores = get_verification_scores(veri_test)
    del enrol_dict, test_dict

    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    print("EER(%%)=%f", eer * 100)

    min_dcf, th = minDCF(
        torch.tensor(positive_scores), torch.tensor(negative_scores)
    )
    print("minDCF=%f", min_dcf * 100)
