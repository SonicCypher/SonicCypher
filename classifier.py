import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Load precomputed Mel-spectrogram features and labels
def load_data(file_path):
    """
    Load precomputed Mel-spectrogram features and labels from a file.
    Assumes the file contains a dictionary with keys: 'features' and 'labels'.
    """
    data = np.load(file_path, allow_pickle=True)
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.float32)
    return features, labels

# Define Binary Classifier
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super(BinaryClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

# Train Model
def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation step
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs).squeeze()
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                
                # Compute accuracy
                predictions = (outputs >= 0.5).int()
                correct += (predictions == labels.int()).sum().item()
                total += labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}")

# Classification Function
def classify_audio(model, inputs, threshold=0.5):
    """
    Classifies audio as "spoof" or "bona fide" based on the model's prediction.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs).squeeze()
        predictions = (outputs >= threshold).int()  # Convert probabilities to binary labels
    
    for i, pred in enumerate(predictions):
        label = "bona fide" if pred.item() == 1 else "spoof"
        print(f"Audio Sample {i+1}: {label} (Score: {outputs[i].item():.4f})")

# Main script
if __name__ == "__main__":
    file_path = "E:/Semester 8/FYP/SonicCypher/mels"  # Path to your stored Mel features
    features, labels = load_data(file_path)
    
    # Split into train, validation, and test sets
    total_size = len(features)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        TensorDataset(features, labels), [train_size, val_size, test_size]
    )

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = features.shape[1]  # Assuming Mel features are (num_samples, feature_dim)
    model = BinaryClassifier(input_dim)

    # Train model
    train_model(model, train_loader, val_loader)

    # Evaluate on test set
    print("\nTesting on unseen data...")
    classify_audio(model, features[:5])  # Test with first 5 samples
