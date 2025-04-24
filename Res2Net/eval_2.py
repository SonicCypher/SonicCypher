import os
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm.contrib import tqdm

import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from speechbrain.utils.metric_stats import EER, minDCF
from torch.utils.data import DataLoader
from models.resnet_models import se_res2net50_v1b
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
        feats = sb.lobes.features.MFCC(n_mfcc=80, n_mels=100, deltas=False, context=False)
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

    # creating cohort for score normalization
    # if "score_norm" in params:
    train_cohort = torch.stack(list(train_dict.values()))

    for i, line in enumerate(veri_test):
        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip()
        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]

        # if "score_norm" in params:
        # Getting norm stats for enrol impostors
        enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
        score_e_c = similarity(enrol_rep, train_cohort)

        num_cohort_samples = train_cohort.shape[0]
        k = min(20000, num_cohort_samples)
        score_e_c = torch.topk(
             score_e_c, k=k, dim=0
         )[0]

        mean_e_c = torch.mean(score_e_c, dim=0)
        std_e_c = torch.std(score_e_c, dim=0)

        # Getting norm stats for test impostors
        test_rep = test.repeat(train_cohort.shape[0], 1, 1)
        score_t_c = similarity(test_rep, train_cohort)

        # if "cohort_size" in params:
        score_t_c = torch.topk(
            # score_t_c, k=params["cohort_size"], dim=0
            score_t_c, k=k, dim=0
         )[0]

        mean_t_c = torch.mean(score_t_c, dim=0)
        std_t_c = torch.std(score_t_c, dim=0)

        # Compute the score for the given sentence
        score = similarity(enrol, test)[0]

        # Perform score normalization
        # if "score_norm" in params:
        #     if params["score_norm"] == "z-norm":
        #         score = (score - mean_e_c) / std_e_c
        #     elif params["score_norm"] == "t-norm":
        #         score = (score - mean_t_c) / std_t_c
        #     elif params["score_norm"] == "s-norm":
        score_e = (score - mean_e_c) / std_e_c
        score_t = (score - mean_t_c) / std_t_c
        score = 0.5 * (score_e + score_t)

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

    # data_folder = r"/mnt/additional-volume/voxdata/vox1_dev_wav"
    data_folder ="/home/cse/SonicCypher/Speaker_Veri_Dataset/voxdata/vox1_dev_wav"
    # Train data (used for normalization)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path='./Preprocessing/Voxceleb/output/train.csv',
        replacements={"data_root": data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration", select_n=400000
    )

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path='./Preprocessing/Voxceleb/output/enrol.csv',
        replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path='./Preprocessing/Voxceleb/output/test.csv',
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data]

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
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=False, num_workers=10, collate_fn=collate_fn)

    enrol_dataloader = DataLoader(enrol_data, batch_size=8, shuffle=False, num_workers=10, collate_fn=collate_fn)

    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=10, collate_fn=collate_fn)

    return train_dataloader, enrol_dataloader, test_dataloader


if __name__ == "__main__":
    # Logger setup
    logger = get_logger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    run_opts = {
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


    # # Load hyperparameters file with command-line overrides
    # params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    # with open(params_file, encoding="utf-8") as fin:
    #     params = load_hyperpyyaml(fin, overrides)

    # # Download verification list (to exclude verification sentences from train)
    # veri_file_path = os.path.join(
    #     params["save_folder"], os.path.basename(params["verification_file"])
    # )
    # download_file(params["verification_file"], veri_file_path)

    # from Preprocessing.Voxceleb.prepare_voxceleb import prepare_voxceleb

    # # Create experiment directory
    # sb.core.create_experiment_directory(
    #     experiment_directory=params["output_folder"],
    #     hyperparams_to_save=params_file,
    #     overrides=overrides,
    # )

    # # Prepare data from dev of Voxceleb1
    # prepare_voxceleb(
    #     data_folder=params["data_folder"],
    #     save_folder=params["save_folder"],
    #     verification_pairs_file=veri_file_path,
    #     splits=["train", "dev", "test"],
    #     split_ratio=params["split_ratio"],
    #     seg_dur=3.0,
    #     skip_prep=params["skip_prep"],
    #     source=(
    #         params["voxceleb_source"] if "voxceleb_source" in params else None
    #     ),
    # )

    # # here we create the datasets objects as well as tokenization and encoding
    # train_dataloader, enrol_dataloader, test_dataloader = dataio_prep(params)

    # # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # # the path given in the YAML file). The tokenizer is loaded at the same time.
    # run_on_main(params["pretrainer"].collect_files)
    # params["pretrainer"].load_collected()
    # params["embedding_model"].eval()
    # params["embedding_model"].to(run_opts["device"])

    # # Computing  enrollment and test embeddings
    # logger.info("Computing enroll/test embeddings...")

    # # First run
    # enrol_dict = compute_embedding_loop(enrol_dataloader)
    # test_dict = compute_embedding_loop(test_dataloader)

    # if "score_norm" in params:
    #     train_dict = compute_embedding_loop(train_dataloader)

    # # Compute the EER
    # logger.info("Computing EER..")
    # # Reading standard verification split
    # with open(veri_file_path, encoding="utf-8") as f:
    #     veri_test = [line.rstrip() for line in f]

    # positive_scores, negative_scores = get_verification_scores(veri_test)
    # del enrol_dict, test_dict

    # eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    # logger.info("EER(%%)=%f", eer * 100)

    # min_dcf, th = minDCF(
    #     torch.tensor(positive_scores), torch.tensor(negative_scores)
    # )
    # logger.info("minDCF=%f", min_dcf * 100)

    from Preprocessing.Voxceleb.prepare_voxceleb import prepare_voxceleb

    # data_folder = "/mnt/additional-volume/voxdata/vox1_dev_wav"
    data_folder ="/home/cse/SonicCypher/Speaker_Veri_Dataset/voxceleb/wav/vox1_dev_wav"
    save_folder_csv = "./Preprocessing/Voxceleb/output"
    splits = ['train', 'dev','test']
    split_ratio = [90, 10]
    # verification_pairs_file = r"/mnt/additional-volume/voxdata/vox1_dev_wav/veri_test.txt"
    verification_pairs_file = r"/home/cse/SonicCypher/Speaker_Veri_Dataset/voxceleb/meta/veri_test.txt"
    print("Preparing VoxCeleb data...")

    prepare_voxceleb(data_folder,save_folder_csv, verification_pairs_file,splits,split_ratio)

    train_dataloader, enrol_dataloader, test_dataloader = dataio_prep()

    model = se_res2net50_v1b(num_classes=1211)
    # model.load_state_dict(torch.load("best_model.pth", map_location=run_opts["device"]))
    last_best_model = torch.load("checkpoints/model_epoch_13.pth", map_location=run_opts["device"])
    model.load_state_dict(last_best_model["model_state_dict"])
    model.eval()

    enrol_dict = compute_embedding_loop(enrol_dataloader)
    test_dict = compute_embedding_loop(test_dataloader)
    train_dict = compute_embedding_loop(train_dataloader)

    # Compute the EER
    logger.info("Computing EER..")
    # Reading standard verification split
    with open(verification_pairs_file, encoding="utf-8") as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores, negative_scores = get_verification_scores(veri_test)
    del enrol_dict, test_dict

    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    logger.info("EER(%%)=%f", eer * 100)

    min_dcf, th = minDCF(
        torch.tensor(positive_scores), torch.tensor(negative_scores)
    )
    logger.info("minDCF=%f", min_dcf * 100)