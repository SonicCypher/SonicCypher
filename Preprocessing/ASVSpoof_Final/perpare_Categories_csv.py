"""
For Preparing data.

"""
import csv
import glob
import os
import random
import shutil
import sys  # noqa F401
from collections import defaultdict

import numpy as np
import ast
import torchaudio
from tqdm.contrib import tqdm

from speechbrain.dataio.dataio import load_pkl, save_pkl
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
OPT_FILE = "opt_ASVSpoof_eval_prepare.pkl"
TEST_CSV = "test.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000
META = "meta"


def prepare_ASV_verification(
    data_folder,
    save_folder,
    verification_pairs_file,
    skip_prep=False,
):

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting output files
    save_opt = os.path.join(save_folder, OPT_FILE)
    
    # Check if this phase is already done (if so, skip it)
    if skip(["test","enrol"], save_folder, conf):
        print("Skipping preparation, completed in previous run.")
        return
    
    if "," in data_folder:
        data_folder = data_folder.replace(" ", "").split(",")
    else:
        data_folder = [data_folder]


    msg = "Creating csv file for the ASVSpoof Dataset.."
    print(msg)
    
    prepare_csv_enrol_test(
        data_folder, save_folder, verification_pairs_file
    )
    
    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    # Checking csv files
    skip = True

    split_files = {
        "test": TEST_CSV,
        "enrol": ENROL_CSV,
    }
    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False
    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip

def prepare_csv_enrol_test(data_folders, save_folder, verification_pairs_file):
    """
    Creates the csv files for test and enrollment data based on the verification pairs file.

    Arguments
    ---------
    data_folders : list of str
        Paths of the data folders (e.g., base path to LA data).
    save_folder : str
        Directory where to store the csv files.
    verification_pairs_file : str
        Path to the file with verification pairs. Format:
        [target/nontarget/spoof, claimed speaker id:[enrollment utterance ids], claimed speaker id/testing utterance id]
    """
    csv_output_head = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    for data_folder in data_folders:
        enrol_map = defaultdict(list)  # speaker_id -> list of enrol utterance ids
        test_ids = []

        # Parse verification pairs
        with open(verification_pairs_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(", ")
                speaker_id = parts[1].split(":")[0]
                enrol_list_str = parts[1].split(":")[1]
                enrol_list = ast.literal_eval(enrol_list_str)
                test_id = parts[2].strip()

                enrol_map[speaker_id].extend(enrol_list)
                test_ids.append(test_id)

        # Remove duplicates
        for k in enrol_map:
            enrol_map[k] = list(set(enrol_map[k]))
        test_ids = list(set(test_ids))

        # Prepare enrollment CSV
        print("Preparing enrollment CSV...")
        enrol_csv = []
        for spk_id, utt_list in enrol_map.items():
            for utt_id in utt_list:
                flac = os.path.join(data_folder, "flac", utt_id + ".flac")
                if not os.path.isfile(flac):
                    print(f"[WARNING] File not found: {flac}")
                    continue

                signal, fs = torchaudio.load(flac)
                signal = signal.squeeze(0)
                audio_duration = signal.shape[0] / SAMPLERATE
                start_sample = 0
                stop_sample = signal.shape[0]

                full_id = f"{spk_id}/{utt_id}"  # e.g., LA_0015/LA_E_Axxxxx
                csv_line = [full_id, audio_duration, flac, start_sample, stop_sample, spk_id]
                enrol_csv.append(csv_line)

        enrol_csv_full = csv_output_head + enrol_csv
        enrol_csv_file = os.path.join(save_folder, ENROL_CSV)
        with open(enrol_csv_file, mode="w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(enrol_csv_full)

        # Prepare test CSV
        print("Preparing test CSV...")
        test_csv = []
        for full_id in test_ids:
            utt_id = full_id.split("/")[-1]
            spk_id = full_id.split("/")[0]
            flac = os.path.join(data_folder, "flac", utt_id + ".flac")

            if not os.path.isfile(flac):
                print(f"[WARNING] File not found: {flac}")
                continue

            signal, fs = torchaudio.load(flac)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]

            csv_line = [full_id, audio_duration, flac, start_sample, stop_sample, spk_id]
            test_csv.append(csv_line)

        test_csv_full = csv_output_head + test_csv
        test_csv_file = os.path.join(save_folder, TEST_CSV)
        with open(test_csv_file, mode="w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(test_csv_full)

        print("CSV generation complete.")