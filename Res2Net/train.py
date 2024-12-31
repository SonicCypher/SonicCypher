from models.resnet_models import res2net50_v1b
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from math import pow

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
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

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

num_samples = 1000
num_features = 20
sequence_length = 50
num_classes = 10

X_train = np.random.rand(num_samples, 1, num_features, sequence_length)
y_train = np.random.randint(0, num_classes, size=num_samples)

X_val = np.random.rand(int(num_samples * 0.2), 1, num_features, sequence_length)
y_val = np.random.randint(0, num_classes, size=int(num_samples * 0.2))

train_dataset = MFCCDataset(X_train, y_train)
val_dataset = MFCCDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = res2net50_v1b(num_classes=num_classes)

epochs = 10
warmup_steps = 4000
patience = 3  # Early stopping patience
pretrained = False  # Load pretrained model if available

train_model(model, train_loader, val_loader, epochs, warmup_steps, device, patience, pretrained)
