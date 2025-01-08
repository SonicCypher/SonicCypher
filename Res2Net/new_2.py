from models.resnet_models import se_res2net50_v1b
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from math import pow
import os
import sys
from tqdm import tqdm
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_pipline import MFCC_Extraction
from Model.model import MFCC_extracter_train, MFCC_extracter_valid

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

class MFCCDatasetInMemory(Dataset):
    def __init__(self, mfcc_features, spkid_labels):
        self.mfcc_features = mfcc_features
        self.spkid_labels = spkid_labels

    def __len__(self):
        return len(self.mfcc_features)

    def __getitem__(self, idx):
        mfcc_data = self.mfcc_features[idx]
        spkid_data = self.spkid_labels[idx]
        mfcc_data = np.expand_dims(mfcc_data, axis=0)
        spkid_data = np.squeeze(spkid_data)

        return torch.tensor(mfcc_data, dtype=torch.float32), torch.tensor(spkid_data, dtype=torch.long)

def train_model_in_memory(model, epochs, warmup_steps, device, patience=5, pretrained=False):
    # Extract features and labels in memory
    train_data, valid_data, label_encoder = MFCC_Extraction()
    # num_classes = len(np.unique(spkid_labels))

    train_dataloader = DataLoader(train_data, batch_size=5, shuffle=False, num_workers=0)
    val_loader = DataLoader(valid_data, batch_size=5, shuffle=False, num_workers=0)
    # Split data into training and validation sets
    # train_size = int(0.8 * len(mfcc_features))
    # val_size = len(mfcc_features) - train_size
    # train_features, val_features = mfcc_features[:train_size], mfcc_features[train_size:]
    # train_labels, val_labels = spkid_labels[:train_size], spkid_labels[train_size:]

    # Create datasets and dataloaders
    # train_dataset = MFCCDatasetInMemory(train_features, train_labels)
    # val_dataset = MFCCDatasetInMemory(val_features, val_labels)

    # train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

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
        print(f"Epoch {epoch+1}/{epochs}")
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(tqdm(train_dataloader, desc="Processing Batches for training", dynamic_ncols=True)):
            print(f"Batch {i+1}/{len(train_dataloader)}")
            mfcc_features_train, spkid_labels_train = MFCC_extracter_train(data, device)
            # train_dataset = MFCCDatasetInMemory(mfcc_features_train, spkid_labels_train)
            # inputs_train, labels_train = train_dataset[0]
            # inputs_train = inputs_train.unsqueeze(0)
            # labels_train = labels_train.unsqueeze(0)
            # print(inputs_train.shape, labels_train.shape)
            # inputs_train, labels_train = inputs_train.to(device), labels_train.to(device)

            # Convert data into tensors and ensure proper batch sizes
            inputs_train = torch.tensor(mfcc_features_train, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
            labels_train = torch.tensor(spkid_labels_train, dtype=torch.long).to(device)
            labels_train = labels_train.squeeze()

            print(inputs_train.shape, labels_train.shape)  # Should print (batch_size, 1, 301, 80) and (batch_size,)


            scheduler.zero_grad()
            outputs = model(inputs_train)
            loss = criterion(outputs, labels_train)
            loss.backward()
            scheduler.step()
            scheduler.update_learning_rate()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels_train.size(0)
            correct += predicted.eq(labels_train).sum().item()

        train_accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_dataloader):.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0


        with torch.no_grad():
            for i,data in enumerate(tqdm(val_loader, desc="Processing Batches for validation", dynamic_ncols=True)):
                mfcc_features_valid, spkid_labels_valid = MFCC_extracter_valid(data, device)
                # val_dataset = MFCCDatasetInMemory(mfcc_features_valid, spkid_labels_valid)
                # inputs_valid, labels_valid = val_dataset[0]
                # inputs_valid, labels_valid = inputs_valid.to(device), labels_valid.to(device)
                inputs_valid = torch.tensor(mfcc_features_valid, dtype=torch.float32).unsqueeze(1).to(device)
                labels_valid = torch.tensor(spkid_labels_valid, dtype=torch.long).to(device)
                labels_valid = labels_valid.squeeze()
                outputs = model(inputs_valid)
                loss = criterion(outputs, labels_valid)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels_valid.size(0)
                correct += predicted.eq(labels_valid).sum().item()

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
        print("One epoch is done.")

# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = se_res2net50_v1b(num_classes=2)

train_model_in_memory(model, epochs=20, warmup_steps=1000, device=device, patience=5, pretrained=False)
