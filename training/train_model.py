 import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_prepare_data(output_dir="data"):
    # Load the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    data = pd.read_csv(url, header=None, names=columns)
    
    # Split into training and inference sets
    train_data, inference_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Save datasets locally
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, "training.csv"), index=False)
    inference_data.to_csv(os.path.join(output_dir, "inference.csv"), index=False)
    
    print(f"Data saved in {output_dir}")
    
if __name__ == "__main__":
    try:
        load_and_prepare_data()
    except Exception as e:
        print(f"Error during data preparation: {e}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

class IrisNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

def train_model(data_path="data/training.csv", model_path="training/model.pth"):
    # Load data
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(data.iloc[:, -1])
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Define model
    input_size = X.shape[1]
    hidden_size = 10
    output_size = len(set(y))
    model = IrisNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(50):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item()}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")

