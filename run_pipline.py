import os
import torch
from Preprocessing.Voxceleb.prepare_voxceleb import prepare_voxceleb

# Import the required functions
from Model.model import dataio_prep, MFCC_extracter_train,MFCC_extracter_valid  # Replace 'your_module' with the actual module name

def MFCC_Extraction():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data_folder = "/mnt/additional-volume/voxdata/vox1_dev_wav"
    data_folder ="/home/cse/SonicCypher/Speaker_Veri_Dataset/voxceleb/wav/vox1_dev_wav"
    save_folder_csv = "./Preprocessing/Voxceleb/output"
    splits = ['train', 'dev','test']
    split_ratio = [90, 10]
    # verification_pairs_file = r"/mnt/additional-volume/voxdata/vox1_dev_wav/veri_test.txt"
    verification_pairs_file = r"/home/cse/SonicCypher/Speaker_Veri_Dataset/voxceleb/meta/veri_test.txt"
    print("Preparing VoxCeleb data...")

    prepare_voxceleb(data_folder,save_folder_csv, verification_pairs_file,splits,split_ratio)
    # # Define paths and parameters
    save_folder = r"Model/output"  # Path to save processed data
    save_folder_mfcc_train = r"Model/output/train"  # Path to save processed data
    save_folder_mfcc_valid = r"Model/output/valid"  # Path to save processed data

    # Ensure directories exist
    for folder in [save_folder, save_folder_mfcc_train, save_folder_mfcc_valid]:
        os.makedirs(folder, exist_ok=True)  # Create folder if it doesn’t exist

    train_annotation = "Preprocessing/Voxceleb/output/train.csv"  # Training annotations CSV
    valid_annotation = "Preprocessing/Voxceleb/output/dev.csv"  # Validation annotations CSV

    if not os.listdir(save_folder_mfcc_train) or not os.listdir(save_folder_mfcc_valid):
        print("Extracting MFCCs...")
        train_data, valid_data, label_encoder = dataio_prep(data_folder, save_folder, train_annotation, valid_annotation)
        MFCC_extracter_train(train_data, device)
        MFCC_extracter_valid(valid_data, device)
        print("MFCC extraction completed.")
    else:
        print("MFCC extraction skipped (already exists).")
    
    # # Step 1: Prepare the data using `dataio_prep`
    # train_data, valid_data, label_encoder = dataio_prep(data_folder, save_folder, train_annotation, valid_annotation)
    # MFCC_extracter_train(train_data, device)
    # MFCC_extracter_valid(valid_data, device)
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
