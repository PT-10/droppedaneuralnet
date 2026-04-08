import torch
import json
import numpy as np
import pandas as pd
from model import Model

with open("weight_indices_2.json") as f:
    wi = json.load(f)

final_piece = wi["final_layer_piece"]
blocks      = wi["blocks"]
n           = len(blocks)

seq = [(b["block_in_piece"], b["block_out_piece"], b["combined_score"])
       for b in blocks]

df        = pd.read_csv("historical_data.csv")
feat_cols = [f"measurement_{i}" for i in range(48)]
X         = torch.tensor(df[feat_cols].values, dtype=torch.float32)
y_pred    = df["pred"].values      # <-- target is pred, not true
y_true    = df["true"].values

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
    mse_pred = np.mean((preds - y_pred) ** 2)
    mse_true = np.mean((preds - y_true) ** 2)
    return mse_pred, mse_true

def swap(seq, i, j):
    s = seq[:]
    s[i], s[j] = s[j], s[i]
    return s

mse_pred, mse_true = eval_seq(seq)
print(f"Starting MSE vs pred : {mse_pred:.6f}")
print(f"Starting MSE vs true : {mse_true:.6f}")

print("\n── Coordinate descent (target = pred) ──")
improved = True
passes   = 0
while improved:
    improved = False
    passes  += 1
    for i in range(n - 1):
        for j in range(i + 1, n):
            candidate       = swap(seq, i, j)
            mse_p, mse_t    = eval_seq(candidate)
            if mse_p < mse_pred:
                seq      = candidate
                mse_pred = mse_p
                mse_true = mse_t
                improved = True
                print(f"  Pass {passes}: Block {i:2d} ↔ Block {j:2d}  →  MSE_pred={mse_pred:.6f}  MSE_true={mse_true:.6f}")

print(f"\nFinal MSE vs pred : {mse_pred:.6f}")
print(f"Final MSE vs true : {mse_true:.6f}")

# Save model
model = Model(num_blocks=n)
for i, (in_p, out_p, _) in enumerate(seq):
    model.blocks[i].inp.load_state_dict(
        torch.load(f"pieces/piece_{in_p}.pth",  map_location="cpu"))
    model.blocks[i].out.load_state_dict(
        torch.load(f"pieces/piece_{out_p}.pth", map_location="cpu"))
model.final.layer.load_state_dict(
    torch.load(f"pieces/piece_{final_piece}.pth", map_location="cpu"))
torch.save(model.state_dict(), "pred_optimized_model.pth")

# Save indices
wi_out = {
    "description": f"Optimized vs pred. MSE_pred={mse_pred:.6f} MSE_true={mse_true:.6f}",
    "final_layer_piece": final_piece,
    "blocks": [
        {
            "block_index":     i,
            "block_in_piece":  in_p,
            "block_out_piece": out_p,
            "combined_score":  score,
        }
        for i, (in_p, out_p, score) in enumerate(seq)
    ]
}
with open("weight_indices_pred.json", "w") as f:
    json.dump(wi_out, f, indent=4)

print("\nSaved pred_optimized_model.pth")
print("Saved weight_indices_pred.json")

# Print solution line
result = []
for b in wi_out["blocks"]:
    result.append(b["block_in_piece"])
    result.append(b["block_out_piece"])
result.append(final_piece)
print("\nSolution:")
print(", ".join(str(x) for x in result))
