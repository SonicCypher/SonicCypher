import os
import random
import sys
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import get_all_files
from speechbrain.augment.preparation import prepare_csv
from speechbrain.augment.time_domain import AddNoise
from speechbrain.augment.augmenter import Augmenter
import speechbrain as sb
from torch.utils.data import DataLoader


def dataio_prep(data_folder, save_folder, train_annotation, valid_annotation):
    "Creates the datasets and their data processing pipelines."

    data_folder = data_folder

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_annotation,
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_annotation,
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(save_folder, "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder

def MFCC_extracter_train(data, output_dir, device):

    noise_folder = r"Model\noise\free-sound"
    speech_folder = r"Model\noise\librivox"

    noise_filelist = get_all_files(noise_folder, match_and=['.wav'])
    speech_filelist = get_all_files(speech_folder, match_and=['.wav'])

    noise_csv = r"Model\noise_csv\noise.csv"
    speech_csv = r"Model\noise_csv\speech.csv"

    prepare_csv(noise_filelist, noise_csv)
    prepare_csv(speech_filelist, speech_csv)

    add_noise = AddNoise(
        csv_file = noise_csv, 
        snr_low=0, 
        snr_high=16, 
        noise_sample_rate=16000, 
        clean_sample_rate=16000, 
        num_workers=0
        )
    
    add_babble = AddNoise(
        csv_file = speech_csv, 
        snr_low=0, 
        snr_high=16, 
        noise_sample_rate=16000, 
        clean_sample_rate=16000, 
        num_workers=0, 
        )
    
    augmenter = Augmenter(
        parallel_augment= True,
        concat_original= True,
        min_augmentations= 2,
        max_augmentations= 2,
        augment_prob=1.0,
        augmentations=[add_noise, add_babble],
    )

    feats = sb.lobes.features.MFCC(n_mfcc=80, n_mels=100, deltas=False, context=False)

    # Assuming you have defined your dataset
    train_dataloader = DataLoader(data, batch_size=25, shuffle=False, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)

    mfcc_dir = os.path.join(output_dir, 'mfcc')
    os.makedirs(mfcc_dir, exist_ok=True)

    spkid_dir = os.path.join(output_dir, 'spkid')
    os.makedirs(spkid_dir, exist_ok=True)

    mfcc_features = []
    spkid_labels = []

    for batch_num, batch in enumerate(tqdm(train_dataloader, desc="Processing Batches", dynamic_ncols=True)):
        wavs = batch["sig"]    # Get waveforms from the batch
        lengths = [len(wav) for wav in wavs]  # Get lengths of each waveform
        max_length = max(lengths)  # Find the maximum length in the batch
        lens = torch.tensor([length / max_length for length in lengths], dtype=torch.float32).to(device)
        ids = batch["id"]    # Get unique IDs from the batch
        spkids = batch["spk_id_encoded"]

        wavs, lens = augmenter(wavs, lens)
        features = feats(wavs)
        spkids = augmenter.replicate_labels(spkids)

        # Append to in-memory lists
        mfcc_features.extend(features.cpu().numpy())
        spkid_labels.extend(spkids.cpu().numpy())

    return np.array(mfcc_features), np.array(spkid_labels)

    #     for idx, (mfcc, spkid) in enumerate(zip(features, spkids)):
    #         # Save MFCC features
    #         mfcc_save_path = os.path.join(mfcc_dir, f"mfcc_batch{batch_num}_idx{idx}.npy")
    #         np.save(mfcc_save_path, mfcc.numpy())  # Save MFCC features as .npy file

    #         # Save encoded speaker ID
    #         spkid_save_path = os.path.join(spkid_dir, f"spkid_batch{batch_num}_idx{idx}.npy")
    #         np.save(spkid_save_path, spkid.numpy())  # Save speaker ID as .npy file
    # print(f"Saved train MFCCs")

def MFCC_extracter_valid(data, output_dir, device):
        
        feats = sb.lobes.features.MFCC(n_mfcc=80, n_mels=100, deltas=False, context=False)

        # Assuming you have defined your dataset
        train_dataloader = DataLoader(data, batch_size=25, shuffle=False, num_workers=0)

        os.makedirs(output_dir, exist_ok=True)

        mfcc_dir = os.path.join(output_dir, 'mfcc')
        os.makedirs(mfcc_dir, exist_ok=True)

        spkid_dir = os.path.join(output_dir, 'spkid')
        os.makedirs(spkid_dir, exist_ok=True)

        
        mfcc_features = []
        spkid_labels = []

        for batch_num, batch in enumerate(tqdm(train_dataloader, desc="Processing Batches", dynamic_ncols=True)):
            wavs = batch["sig"]    # Get waveforms from the batch
            spkids = batch["spk_id_encoded"]
            # Extract MFCC features
            features = feats(wavs)

            # Append to in-memory lists
            mfcc_features.extend(features.cpu().numpy())
            spkid_labels.extend(spkids.cpu().numpy())

        return np.array(mfcc_features), np.array(spkid_labels)

        #     for idx, (mfcc, spkid) in enumerate(zip(features, spkids)):
        #         # Save MFCC features
        #         mfcc_save_path = os.path.join(mfcc_dir, f"mfcc_batch{batch_num}_idx{idx}.npy")
        #         np.save(mfcc_save_path, mfcc.cpu().numpy()) # Save MFCC features as .npy file

        #         # Save encoded speaker ID
        #         spkid_save_path = os.path.join(spkid_dir, f"spkid_batch{batch_num}_idx{idx}.npy")
        #         np.save(spkid_save_path, spkid.cpu().numpy())  # Save speaker ID as .npy file
        # print(f"Saved Valid MFCCs")