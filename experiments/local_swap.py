import torch
import json
import numpy as np
import pandas as pd
from model import Model

WINDOW = 5   # neighbourhood radius for weak-block swaps

with open("weight_indices_2.json") as f:
    wi = json.load(f)

final_piece = wi["final_layer_piece"]
blocks      = wi["blocks"]           # list of dicts, already in model order
n           = len(blocks)

# Working sequence: list of (block_in_piece, block_out_piece, combined_score)
seq = [(b["block_in_piece"], b["block_out_piece"], b["combined_score"])
       for b in blocks]

df        = pd.read_csv("historical_data.csv")
feat_cols = [f"measurement_{i}" for i in range(48)]
X         = torch.tensor(df[feat_cols].values, dtype=torch.float32)
y_true    = df["true"].values
baseline  = np.mean((df["pred"].values - y_true) ** 2)

def eval_seq(seq):
    model = Model(num_blocks=len(seq))
    for i, (in_p, out_p, _) in enumerate(seq):
        model.blocks[i].inp.load_state_dict(
            torch.load(f"pieces/piece_{in_p}.pth",  map_location="cpu"))
        model.blocks[i].out.load_state_dict(
            torch.load(f"pieces/piece_{out_p}.pth", map_location="cpu"))
    model.final.layer.load_state_dict(
        torch.load(f"pieces/piece_{final_piece}.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().numpy()
    return np.mean((preds - y_true) ** 2)

def swap(seq, i, j):
    s = seq[:]
    s[i], s[j] = s[j], s[i]
    return s

current_mse = eval_seq(seq)
print(f"Baseline pred MSE          : {baseline:.6f}")
print(f"Starting model MSE         : {current_mse:.6f}")

scores       = [b["combined_score"] for b in blocks]
weak_indices = sorted(range(n), key=lambda i: scores[i])[:5]
print(f"\nWeakest 5 block positions  : {weak_indices}")
print(f"Their combined scores      : {[round(scores[i], 4) for i in weak_indices]}")

print("\n── Phase 1: Neighbourhood swaps for weak blocks ──")
for wi_idx in weak_indices:
    lo = max(0, wi_idx - WINDOW)
    hi = min(n - 1, wi_idx + WINDOW)
    best_mse, best_j = current_mse, -1
    for j in range(lo, hi + 1):
        if j == wi_idx:
            continue
        candidate = swap(seq, wi_idx, j)
        mse       = eval_seq(candidate)
        if mse < best_mse:
            best_mse, best_j = mse, j
    if best_j >= 0:
        seq         = swap(seq, wi_idx, best_j)
        current_mse = best_mse
        print(f"  Block {wi_idx:2d} ↔ Block {best_j:2d}  →  MSE = {current_mse:.6f}  ✓")
    else:
        print(f"  Block {wi_idx:2d}: no improvement in ±{WINDOW} window")

print(f"\nAfter Phase 1 MSE          : {current_mse:.6f}")

print("\n── Phase 2: Full coordinate-descent (all pairs) ──")
improved = True
passes   = 0
while improved:
    improved = False
    passes  += 1
    for i in range(n - 1):
        for j in range(i + 1, n):
            candidate = swap(seq, i, j)
            mse       = eval_seq(candidate)
            if mse < current_mse:
                seq         = candidate
                current_mse = mse
                improved    = True
                print(f"  Pass {passes}: Block {i:2d} ↔ Block {j:2d}  →  MSE = {current_mse:.6f}  ✓")

print(f"\nAfter Phase 2 MSE          : {current_mse:.6f}  (passes: {passes})")

model = Model(num_blocks=n)
for i, (in_p, out_p, _) in enumerate(seq):
    model.blocks[i].inp.load_state_dict(
        torch.load(f"pieces/piece_{in_p}.pth",  map_location="cpu"))
    model.blocks[i].out.load_state_dict(
        torch.load(f"pieces/piece_{out_p}.pth", map_location="cpu"))
model.final.layer.load_state_dict(
    torch.load(f"pieces/piece_{final_piece}.pth", map_location="cpu"))

torch.save(model.state_dict(), "local_swap_model.pth")

weight_indices_3 = {
    "description": f"Local-swap refined. MSE={current_mse:.6f}",
    "final_layer_piece": final_piece,
    "blocks": [
        {
            "block_index":     i,
            "block_in_piece":  in_p,
            "block_out_piece": out_p,
            "combined_score":  score,
            "depth_position":  "shallow" if i < n // 3 else ("mid" if i < 2 * n // 3 else "deep")
        }
        for i, (in_p, out_p, score) in enumerate(seq)
    ]
}
with open("weight_indices_3.json", "w") as f:
    json.dump(weight_indices_3, f, indent=4)

print(f"\nSaved local_swap_model.pth")
print(f"Saved weight_indices_3.json")
print(f"\n=== Summary ===")
print(f"  Baseline pred column : {baseline:.6f}")
print(f"  Before local swap    : 0.176944")
print(f"  After local swap     : {current_mse:.6f}")
