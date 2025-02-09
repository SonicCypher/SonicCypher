import os
import torch
from Preprocessing.Voxceleb.prepare_voxceleb import prepare_voxceleb

# Import the required functions
from Model.model import dataio_prep, MFCC_extracter_train,MFCC_extracter_valid  # Replace 'your_module' with the actual module name

def MFCC_Extraction():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folder = "/mnt/additional-volume/voxdata/vox1_dev_wav_10" # TODO: Change this to the actual path
    save_folder_csv = os.path.join("Preprocessing", "Voxceleb", "output")
    splits = ['train', 'dev']
    split_ratio = [90, 10]
    print("Preparing VoxCeleb data...")

    prepare_voxceleb(data_folder,save_folder_csv,splits,split_ratio)
    # # Define paths and parameters
    save_folder = "Model/output"  # Path to save processed data
    # save_folder_mfcc_train = r"Model\output\train"  # Path to save processed data
    # save_folder_mfcc_valid = r"Model\output\valid"  # Path to save processed data

    train_annotation = "Preprocessing/Voxceleb/output/train.csv"  # Training annotations CSV
    valid_annotation = "Preprocessing/Voxceleb/output/dev.csv"  # Validation annotations CSV

    # # Step 1: Prepare the data using `dataio_prep`
    train_data, valid_data, label_encoder = dataio_prep(data_folder, save_folder, train_annotation, valid_annotation)
    MFCC_extracter_train(train_data, device)
    MFCC_extracter_valid(valid_data, device)
    # # Step 2: Use the output of `dataio_prep` as input to `MFCC_extracter`

    # print("Extracting MFCCs for training data...")
    # train_mfccs, train_spkids = MFCC_extracter_train(train_data, save_folder_mfcc_train, device)
    # print("Completed Extracting MFCCs for training data...")
    # print("Extracting MFCCs for valid data...")
    # valid_mfccs, valid_spkids = MFCC_extracter_valid(valid_data, save_folder_mfcc_valid, device)
    # print("Completed Extracting MFCCs for valid data...")
    # print(train_mfccs.shape, train_spkids.shape)
    # print(valid_mfccs.shape, valid_spkids.shape)

    # return train_data, valid_data, label_encoder


# if __name__ == "__main__":
#     main()
