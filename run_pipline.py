import os
import torch
from Preprocessing.Voxceleb.prepare_voxceleb import prepare_voxceleb

# Import the required functions
from Model.model import dataio_prep, MFCC_extracter_train,MFCC_extracter_valid  # Replace 'your_module' with the actual module name

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folder = os.path.join( "Voxceleb")
    save_folder_csv = os.path.join("Preprocessing", "Voxceleb", "output")
    splits = ['train', 'dev']
    split_ratio = [90, 10]

    prepare_voxceleb(data_folder,save_folder_csv,splits,split_ratio)
    # # Define paths and parameters
    save_folder = r"Model\output"  # Path to save processed data
    save_folder_mfcc_train = r"Model\output\train"  # Path to save processed data
    save_folder_mfcc_valid = r"Model\output\valid"  # Path to save processed data

    train_annotation = r"Preprocessing\Voxceleb\output\train.csv"  # Training annotations CSV
    valid_annotation = r"Preprocessing\Voxceleb\output\dev.csv"  # Validation annotations CSV

    # # Step 1: Prepare the data using `dataio_prep`
    train_data, valid_data, label_encoder = dataio_prep(data_folder, save_folder, train_annotation, valid_annotation)

    # Move data to the GPU
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)


    # Step 2: Use the output of `dataio_prep` as input to `MFCC_extracter`
    print("Extracting MFCCs for training data...")
    MFCC_extracter_train(train_data, save_folder_mfcc_train, device)
    print("Completed Extracting MFCCs for training data...")
    print("Extracting MFCCs for valid data...")
    MFCC_extracter_valid(valid_data, save_folder_mfcc_valid, device)
    print("Completed Extracting MFCCs for valid data...")

if __name__ == "__main__":
    main()
