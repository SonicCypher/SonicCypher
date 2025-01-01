from models.resnet_models import res2net50_v1b
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from math import pow
import os

# ScheduledOptim class definition
class ScheduledOptim(object):
    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        self.n_current_steps += self.delta
        new_lr = pow(self.d_model, -0.5) * min(
            pow(self.n_current_steps, -0.5),
            pow(self.n_warmup_steps, -1.5) * self.n_current_steps
        )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

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
        

def train_model(model, train_loader, val_loader, epochs, warmup_steps, device, patience=5, pretrained=False):
    if pretrained:
        try:
            model.load_state_dict(torch.load("best_model.pth"))
            print("Loaded pretrained model from best_model.pth")
        except FileNotFoundError:
            print("No pretrained model found. Starting training from scratch.")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0)
    scheduler = ScheduledOptim(optimizer, warmup_steps)

    best_val_accuracy = 0
    no_improve_epochs = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            scheduler.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            scheduler.step()
            scheduler.update_learning_rate()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_accuracy = 100.0 * correct / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save the best model
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

# Paths to the directories containing the MFCC and speaker ID files
mfcc_folder = "./Model/output/mfcc"
spkid_folder = "./Model/output/spkid"

# Load file paths
mfcc_files = sorted([os.path.join(mfcc_folder, f) for f in os.listdir(mfcc_folder) if f.endswith('.npy')])
spkid_files = sorted([os.path.join(spkid_folder, f) for f in os.listdir(spkid_folder) if f.endswith('.npy')])

# Create the dataset
full_dataset = MFCCDataset(mfcc_files, spkid_files)

# Split into training and validation datasets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cpu")
model = res2net50_v1b(num_classes=len(np.unique([np.load(f) for f in spkid_files])))

epochs = 15
warmup_steps = 4000
patience = 5  # Early stopping patience
pretrained = False  # Load pretrained model if available

train_model(model, train_loader, val_loader, epochs, warmup_steps, device, patience, pretrained)
