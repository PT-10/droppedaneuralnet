import torch
import json
import numpy as np
from scipy.stats import skew
import pandas as pd
from model import Model

# --- Skewness check on highest-norm vs lowest-norm pairs ---
with open("paired_blocks.json") as f:
    data = json.load(f)

pairs = data["pairs"]
first_pair = pairs[0]   # highest norm (current block 0)
last_pair  = pairs[-1]  # lowest norm  (current block 47)

def weight_skewness(piece_idx):
    state = torch.load(f"pieces/piece_{piece_idx}.pth", map_location="cpu")
    w = next(v for k, v in state.items() if len(v.shape) == 2)
    return skew(w.numpy().flatten())

print("=== Skewness Check ===")
for label, pair in [("Highest-norm (block 0)", first_pair), ("Lowest-norm (block 47)", last_pair)]:
    sk_out = weight_skewness(pair["block_out_piece"])
    sk_in  = weight_skewness(pair["block_in_piece"])
    print(f"{label}:")
    print(f"  out piece {pair['block_out_piece']:2d}  skew={sk_out:.4f}  norm={pair['out_norm']:.3f}")
    print(f"  in  piece {pair['block_in_piece']:2d}  skew={sk_in:.4f}  norm={pair['in_norm']:.3f}")

# --- Rebuild with INVERTED block order ---
print("\n=== Rebuilding model with inverted block order ===")
inverted_pairs = list(reversed(pairs))

model = Model(num_blocks=len(inverted_pairs))

for i, pair in enumerate(inverted_pairs):
    inp_state = torch.load(f"pieces/piece_{pair['block_in_piece']}.pth", map_location="cpu")
    out_state = torch.load(f"pieces/piece_{pair['block_out_piece']}.pth", map_location="cpu")
    model.blocks[i].inp.load_state_dict(inp_state)
    model.blocks[i].out.load_state_dict(out_state)

final_piece = data["final_layer"][0]["piece"]
final_state = torch.load(f"pieces/piece_{final_piece}.pth", map_location="cpu")
model.final.layer.load_state_dict(final_state)
model.eval()

# --- Predict and compute MSE ---
df = pd.read_csv("historical_data.csv")
feat_cols = [f"measurement_{i}" for i in range(48)]
X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
y_true = df["true"].values

with torch.no_grad():
    preds = model(X).squeeze().numpy()

mse = np.mean((preds - y_true) ** 2)
print(f"\nSample preds: {preds[:5]}")
print(f"Sample true:  {y_true[:5]}")
print(f"\nFinal MSE (inverted order): {mse:.6f}")
