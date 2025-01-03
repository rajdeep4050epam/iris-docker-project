 import torch
import pandas as pd
from training.train_model import IrisNet
import os

def run_inference(model_path="training/model.pth", data_path="data/inference.csv", output_path="inference/results.csv"):
    # Load data
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values
    
    # Load model
    input_size = X.shape[1]
    hidden_size = 10
    output_size = len(set(data.iloc[:, -1]))
    model = IrisNet(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Run inference
    X_tensor = torch.tensor(X, dtype=torch.float32)
    predictions = model(X_tensor).argmax(dim=1).numpy()
    
    # Save results
    data['prediction'] = predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Inference results saved at {output_path}")

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"Error during inference: {e}")

