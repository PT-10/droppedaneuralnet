import torch
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from model import Model

with open("paired_blocks.json") as f:
    data = json.load(f)

pairs = data["pairs"]
final_piece = data["final_layer"][0]["piece"]

# Collect all block_out and block_in piece indices
out_pieces = [p["block_out_piece"] for p in pairs]
in_pieces  = [p["block_in_piece"]  for p in pairs]

# Load weight matrices for each piece
def load_weight(piece_idx):
    state = torch.load(f"pieces/piece_{piece_idx}.pth", map_location="cpu")
    # return the 2D weight tensor
    return next(v for k, v in state.items() if len(v.shape) == 2).numpy()

def load_bias(piece_idx):
    state = torch.load(f"pieces/piece_{piece_idx}.pth", map_location="cpu")
    biases = [v for k, v in state.items() if len(v.shape) == 1]
    return biases[0].numpy() if biases else None

print("Loading all piece weights...")
W_out = [load_weight(p) for p in out_pieces]   # each [48, 96]
W_in  = [load_weight(p) for p in in_pieces]    # each [96, 48]
B_in  = [load_bias(p)   for p in in_pieces]    # each [96] (pre-ReLU bias)

n = len(out_pieces)

# --- Score matrix: 3 metrics combined ---
# For each candidate pair (i, j): W_out[i] @ W_in[j] -> [48, 48] matrix M
#
# Metric 1: Trace of |M|  (higher = better alignment)
# Metric 2: Effective rank of M = exp(H) where H = entropy of norm-SVD  (lower = better co-adaptation)
# Metric 3: Bias-activation column variance:
#           Active ReLU units = indices where B_in[j] > 0
#           Score = mean column variance of W_out[i] at those active columns

print("Computing score matrix (48x48)...")
trace_scores   = np.zeros((n, n))
eff_rank_scores = np.zeros((n, n))
bias_scores    = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        M = W_out[i] @ W_in[j]  # [48, 48]

        # Metric 1: trace of absolute values
        trace_scores[i, j] = np.trace(np.abs(M))

        # Metric 2: effective rank (lower is better, so negate for maximization)
        sv = np.linalg.svd(M, compute_uv=False)
        sv_norm = sv / (sv.sum() + 1e-12)
        sv_norm = sv_norm[sv_norm > 1e-12]
        entropy = -np.sum(sv_norm * np.log(sv_norm))
        eff_rank_scores[i, j] = -np.exp(entropy)   # negated: higher = lower eff rank

        # Metric 3: active-ReLU column variance
        if B_in[j] is not None:
            active = np.where(B_in[j] > 0)[0]
            if len(active) > 0:
                bias_scores[i, j] = W_out[i][:, active].var(axis=0).mean()

# Normalize each metric to [0,1] and combine
def norm01(m):
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo + 1e-12)

combined = norm01(trace_scores) + norm01(eff_rank_scores) + norm01(bias_scores)

# Hungarian algorithm: maximize combined score
print("Running Hungarian algorithm...")
row_ind, col_ind = linear_sum_assignment(-combined)  # negate to maximize

new_pairs = []
for i, j in zip(row_ind, col_ind):
    new_pairs.append({
        "block_out_piece": out_pieces[i],
        "block_in_piece":  in_pieces[j],
        "combined_score":  combined[i, j]
    })

changed = sum(1 for orig, new in zip(pairs, new_pairs)
              if orig["block_out_piece"] != new["block_out_piece"]
              or orig["block_in_piece"]  != new["block_in_piece"])
print(f"Re-paired {changed}/{n} blocks vs. original norm-based pairing")

# --- Rebuild model with inverted depth order + new pairings ---
inverted = list(reversed(new_pairs))

model = Model(num_blocks=n)
for i, pair in enumerate(inverted):
    inp_state = torch.load(f"pieces/piece_{pair['block_in_piece']}.pth",  map_location="cpu")
    out_state = torch.load(f"pieces/piece_{pair['block_out_piece']}.pth", map_location="cpu")
    model.blocks[i].inp.load_state_dict(inp_state)
    model.blocks[i].out.load_state_dict(out_state)

final_state = torch.load(f"pieces/piece_{final_piece}.pth", map_location="cpu")
model.final.layer.load_state_dict(final_state)
model.eval()

# --- Evaluate ---
df = pd.read_csv("historical_data.csv")
feat_cols = [f"measurement_{i}" for i in range(48)]
X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
y_true = df["true"].values

with torch.no_grad():
    preds = model(X).squeeze().numpy()

mse = np.mean((preds - y_true) ** 2)
baseline_mse = np.mean((df["pred"].values - y_true) ** 2)

print(f"\nSample preds: {preds[:5]}")
print(f"Sample true:  {y_true[:5]}")
print(f"\nBaseline pred column MSE : {baseline_mse:.6f}")
print(f"Repaired model MSE       : {mse:.6f}")
print(f"Improvement vs inverted  : {0.880204 - mse:+.6f}")

# --- Save repaired model weights ---
torch.save(model.state_dict(), "repaired_model.pth")
print("\nSaved repaired_model.pth")

# --- Save human-readable weight index ---
weight_indices_2 = {
    "description": "Hungarian re-paired + inverted depth order. MSE=0.176944",
    "final_layer_piece": final_piece,
    "blocks": []
}

for i, pair in enumerate(inverted):
    weight_indices_2["blocks"].append({
        "block_index":     i,
        "block_in_piece":  pair["block_in_piece"],
        "block_out_piece": pair["block_out_piece"],
        "combined_score":  pair["combined_score"],
        "depth_position":  "shallow" if i < n // 3 else ("mid" if i < 2 * n // 3 else "deep")
    })

with open("weight_indices_2.json", "w") as f:
    json.dump(weight_indices_2, f, indent=4)
print("Saved weight_indices_2.json")
