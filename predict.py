import sys
import torch
import pandas as pd
import numpy as np
from model import Model

model_path = sys.argv[1] if len(sys.argv) > 1 else "solved_model.pth"

model = Model()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

df = pd.read_csv("historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)

with torch.no_grad():
    preds = model(X).squeeze().numpy()

y_true = df["true"].values
y_pred = df["pred"].values
print(f"MSE vs true: {np.mean((preds - y_true) ** 2):.6f}")
print(f"MSE vs pred: {np.mean((preds - y_pred) ** 2):.6f}")
print(f"\nSolution:\n{open('solution.txt').read().strip()}")
