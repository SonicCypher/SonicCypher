"""
For Preparing data.

"""
import csv
import glob
import os
import random
import shutil
import sys  # noqa F401

import numpy as np
import torch
import torchaudio
from tqdm.contrib import tqdm

from speechbrain.dataio.dataio import load_pkl, save_pkl
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
OPT_FILE = "opt_ASV_prepare.pkl"
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


    msg = "\tCreating csv file for the VoxCeleb Dataset.."
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
    Creates the csv file for test data (useful for verification)

    Arguments
    ---------
    data_folders : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        Path to the file with verification pairs.
    """
    csv_output_head = [
        ["ID", "duration", "wav", "start", "stop", "spk_id"]
    ]  # noqa E231

    for data_folder in data_folders:
        test_lst_file = verification_pairs_file

        enrol_ids, test_ids = [], []

        # Get unique ids (enrol and test utterances)
        for line in open(test_lst_file, encoding="utf-8"):
            e_id = line.split(" ")[1].strip()
            t_id = line.split(" ")[2].strip()
            enrol_ids.append(e_id)
            test_ids.append(t_id)

        enrol_ids = list(np.unique(np.array(enrol_ids)))
        test_ids = list(np.unique(np.array(test_ids)))

        # Prepare enrol csv
        print("preparing enrol csv")
        enrol_csv = []
        for id in enrol_ids:
            utt_id = id.split("/")[-1]
            flac = data_folder + "/flac/" + utt_id + ".flac"

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(flac)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            spk_id = id.split("/")[0]

            csv_line = [
                id,
                audio_duration,
                flac,
                start_sample,
                stop_sample,
                spk_id,
            ]

            enrol_csv.append(csv_line)

        csv_output = csv_output_head + enrol_csv
        csv_file = os.path.join(save_folder, ENROL_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)

        # Prepare test csv
        print("preparing test csv")
        test_csv = []
        for id in test_ids:
            utt_id = id.split("/")[-1]
            flac = data_folder + "/flac/" + utt_id + ".flac"

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(flac)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            spk_id = id.split("/")[0]

            csv_line = [
                id,
                audio_duration,
                flac,
                start_sample,
                stop_sample,
                spk_id,
            ]

            test_csv.append(csv_line)

        csv_output = csv_output_head + test_csv
        csv_file = os.path.join(save_folder, TEST_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)