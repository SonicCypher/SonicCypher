from models.resnet_models import se_res2net50_v1b
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from math import pow
import os
import sys
from tqdm import tqdm
import glob
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_pipline import MFCC_Extraction

checkpoint_dir = "checkpoints"

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

class MFCCDataset(Dataset):
    def __init__(self, mfcc_files, spkid_files):
        """
        Initialize the dataset with paths to MFCC and speaker ID files.
        """
        self.mfcc_files = mfcc_files
        self.spkid_files = spkid_files

    def __len__(self):
        return len(self.mfcc_files)

    def __getitem__(self, idx):
      """
      Load and return an MFCC sample and its corresponding speaker ID.
      """
      mfcc_data = np.load(self.mfcc_files[idx])
      spkid_data = np.load(self.spkid_files[idx])
      # Add channel dimension to mfcc_data
      mfcc_data = np.expand_dims(mfcc_data, axis=0)
      # Ensure spkid_data is 1D (flatten if necessary)
      spkid_data = np.squeeze(spkid_data)
      return torch.tensor(mfcc_data, dtype=torch.float32), torch.tensor(spkid_data, dtype=torch.long)


def train_model(model,train_loader, val_loader, epochs, device, patience=12, pretrained=False):

    writer = SummaryWriter(log_dir='runs/speaker_verification') 

    no_improve_epochs = 0
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    loadedFromCheckpoint = False

    if pretrained:
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            print(f"Loading pretrained model from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            val_accuracy = checkpoint['val_accuracy']
            best_val_loss = checkpoint['best_val_loss']
            start_epoch = checkpoint.get('epoch', 1)
            loadedFromCheckpoint = True
            print(f"Resuming training from epoch {start_epoch} with val_loss: {best_val_loss:.4f}",
              f"best_val_accuracy: {val_accuracy:.2f}%" )
        else:
            print("No pretrained model found, training from scratch.")
            start_epoch = 1 
            best_val_loss = float('inf')
    else:
        print("Training from scratch.")
        start_epoch = 1
        best_val_loss = float('inf')

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if loadedFromCheckpoint:
        start_epoch += 1

    for epoch in range(start_epoch,epochs+1):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']


        train_accuracy = 100.0 * correct / total
        print(f"Epoch {epoch}/{epochs}, Learning Rate: {current_lr:.6f}")
        print(f"Epoch {epoch}/{epochs}, Training Loss: {train_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", unit="batch"):
                inputs, labels = inputs.to(device), labels.to(device)
                # print labels
                # print(labels)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # print(f"Validation Loss large: {loss.item():.4f}")
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                # print predicted
                # print(predicted)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        print(f"validation loss not divided by len: {val_loss:.4f} and len of loader: {len(val_loader):.4f}")
        val_accuracy = 100.0 * correct / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Log metrics to TensorBoard
        writer.add_scalar('Learning Rate', current_lr, epoch)
        writer.add_scalar('Training Loss', train_loss/len(train_loader), epoch)
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)
        writer.add_scalar('Validation Loss', val_loss/len(val_loader), epoch)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

        # Early stopping
        current_val_loss = val_loss/len(val_loader)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            no_improve_epochs = 0

        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        #     no_improve_epochs = 0
            
            # Save the model checkpoint
            checkpoint ={
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_accuracy': val_accuracy,
            }
            save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(checkpoint, save_path)
            print(f"Model saved at {save_path} with validation accuracy: {val_accuracy:.2f}% and best validation loss: {best_val_loss:.4f}")

        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break
        print(f"{epoch} epochs is done.")

        writer.close()


MFCC_Extraction()

# Paths to the directories containing the MFCC and speaker ID files
base_dir = r"./Model/output"
train_mfcc_folder = os.path.join(base_dir, "train/augmented/mfcc")
train_spkid_folder = os.path.join(base_dir, "train/augmented/spkid")
valid_mfcc_folder = os.path.join(base_dir, "valid/mfcc")
valid_spkid_folder = os.path.join(base_dir, "valid/spkid")

# Load file paths
train_mfcc_files = sorted([os.path.join(train_mfcc_folder, f) for f in os.listdir(train_mfcc_folder) if f.endswith('.npy')])
train_spkid_files = sorted([os.path.join(train_spkid_folder, f) for f in os.listdir(train_spkid_folder) if f.endswith('.npy')])

val_mfcc_files = sorted([os.path.join(valid_mfcc_folder, f) for f in os.listdir(valid_mfcc_folder) if f.endswith('.npy')])
val_spkid_files = sorted([os.path.join(valid_spkid_folder, f) for f in os.listdir(valid_spkid_folder) if f.endswith('.npy')])

# Create the dataset
full_train_dataset = MFCCDataset(train_mfcc_files, train_spkid_files)
full_val_dataset = MFCCDataset(val_mfcc_files, val_spkid_files)

# Create DataLoaders
train_loader = DataLoader(full_train_dataset, batch_size=15, shuffle=False)
val_loader = DataLoader(full_val_dataset, batch_size=15, shuffle=False)

device = torch.device("cuda")
model = se_res2net50_v1b(dropblock_prob=0.1, num_classes=1211)

epochs = 100
patience = 10
pretrained = True

train_model(model,train_loader,val_loader, epochs, device, patience, pretrained)
