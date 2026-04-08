"""
Full end-to-end solution for the Jane Street "Dropped a Neural Net" puzzle.

Steps:
  1. Classify all 97 pieces by shape
  2. Sort by Frobenius norm and INVERT for depth order (high norm = deep)
  3. Hungarian algorithm to find optimal within-block inp/out pairing
  4. Coordinate descent swapping block positions, optimizing against `pred`
  5. Print the 97-index permutation
"""

import time
import torch
import numpy as np
import pandas as pd
import json
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from model import Model

t0 = time.time()

def elapsed():
    return f"[{time.time() - t0:6.1f}s]"

N_PIECES = 97
PIECES_DIR = "pieces"

# ── 1. Classify pieces by shape ──────────────────────────────────────────────
print(f"{elapsed()} Step 1: Classifying pieces by tensor shape...")

block_in_pieces   = []   # shape [96, 48]
block_out_pieces  = []   # shape [48, 96]
final_layer_piece = None

for i in tqdm(range(N_PIECES), desc="  Classifying", unit="piece"):
    state = torch.load(f"{PIECES_DIR}/piece_{i}.pth", map_location="cpu")
    for k, v in state.items():
        if len(v.shape) == 2:
            if v.shape == torch.Size([96, 48]):
                block_in_pieces.append(i)
            elif v.shape == torch.Size([48, 96]):
                block_out_pieces.append(i)
            elif v.shape == torch.Size([1, 48]):
                final_layer_piece = i

print(f"{elapsed()}   block_in:  {len(block_in_pieces)} pieces")
print(f"{elapsed()}   block_out: {len(block_out_pieces)} pieces")
print(f"{elapsed()}   final:     piece_{final_layer_piece}")

# ── 2. Sort by Frobenius norm, INVERTED (low norm = shallow) ─────────────────
print(f"\n{elapsed()} Step 2: Sorting by Frobenius norm (inverted depth order)...")

def frobenius(piece_idx):
    state = torch.load(f"{PIECES_DIR}/piece_{piece_idx}.pth", map_location="cpu")
    w = next(v for k, v in state.items() if len(v.shape) == 2)
    return torch.norm(w, p="fro").item()

block_in_sorted  = sorted(tqdm(block_in_pieces,  desc="  Norm block_in",  unit="piece"), key=frobenius)
block_out_sorted = sorted(tqdm(block_out_pieces, desc="  Norm block_out", unit="piece"), key=frobenius)

print(f"{elapsed()}   Norm range block_in:  [{frobenius(block_in_sorted[0]):.3f}, {frobenius(block_in_sorted[-1]):.3f}]")
print(f"{elapsed()}   Norm range block_out: [{frobenius(block_out_sorted[0]):.3f}, {frobenius(block_out_sorted[-1]):.3f}]")

# ── 3. Load all weight matrices ───────────────────────────────────────────────
print(f"\n{elapsed()} Step 3: Loading weights for Hungarian pairing...")

def load_weight(piece_idx):
    state = torch.load(f"{PIECES_DIR}/piece_{piece_idx}.pth", map_location="cpu")
    return next(v for k, v in state.items() if len(v.shape) == 2).numpy()

def load_bias(piece_idx):
    state = torch.load(f"{PIECES_DIR}/piece_{piece_idx}.pth", map_location="cpu")
    biases = [v for k, v in state.items() if len(v.shape) == 1]
    return biases[0].numpy() if biases else None

n = len(block_in_sorted)
W_out_all = [load_weight(p) for p in tqdm(block_out_sorted, desc="  Loading block_out", unit="piece")]
W_in_all  = [load_weight(p) for p in tqdm(block_in_sorted,  desc="  Loading block_in",  unit="piece")]
B_in_all  = [load_bias(p)   for p in block_in_sorted]

# ── 4. Hungarian algorithm: optimal within-block pairing ─────────────────────
print(f"\n{elapsed()} Step 4: Building 48×48 score matrix for Hungarian pairing...")

def norm01(m):
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo + 1e-12)

trace_s  = np.zeros((n, n))
effrnk_s = np.zeros((n, n))
bias_s   = np.zeros((n, n))

for i in tqdm(range(n), desc="  Score matrix", unit="row"):
    for j in range(n):
        M = W_out_all[i] @ W_in_all[j]          # [48, 48]

        # Metric 1: trace of |W_out @ W_in| (diagonal alignment)
        trace_s[i, j] = np.trace(np.abs(M))

        # Metric 2: negative effective rank (lower = more co-adapted)
        sv   = np.linalg.svd(M, compute_uv=False)
        sv_n = sv / (sv.sum() + 1e-12)
        sv_n = sv_n[sv_n > 1e-12]
        effrnk_s[i, j] = -np.exp(-np.sum(sv_n * np.log(sv_n)))

        # Metric 3: variance of W_out columns at active ReLU units
        if B_in_all[j] is not None:
            active = np.where(B_in_all[j] > 0)[0]
            if len(active) > 0:
                bias_s[i, j] = W_out_all[i][:, active].var(axis=0).mean()

combined = norm01(trace_s) + norm01(effrnk_s) + norm01(bias_s)

print(f"{elapsed()} Running Hungarian algorithm...")
row_ind, col_ind = linear_sum_assignment(-combined)

# Build the ordered sequence: (block_in_piece, block_out_piece)
# row_ind[k] indexes into block_out_sorted, col_ind[k] into block_in_sorted
seq = [
    (block_in_sorted[col_ind[k]], block_out_sorted[row_ind[k]])
    for k in range(n)
]

print(f"{elapsed()}   Pairing done. Combined score: {combined[row_ind, col_ind].sum():.4f}")

# ── 5. Load data ──────────────────────────────────────────────────────────────
print(f"\n{elapsed()} Step 5: Loading CSV data...")
df        = pd.read_csv("historical_data.csv")
feat_cols = [f"measurement_{i}" for i in range(48)]
X         = torch.tensor(df[feat_cols].values, dtype=torch.float32)
y_pred    = df["pred"].values
y_true    = df["true"].values

def eval_seq(seq):
    model = Model(num_blocks=len(seq))
    for i, (in_p, out_p) in enumerate(seq):
        model.blocks[i].inp.load_state_dict(
            torch.load(f"{PIECES_DIR}/piece_{in_p}.pth",  map_location="cpu"))
        model.blocks[i].out.load_state_dict(
            torch.load(f"{PIECES_DIR}/piece_{out_p}.pth", map_location="cpu"))
    model.final.layer.load_state_dict(
        torch.load(f"{PIECES_DIR}/piece_{final_layer_piece}.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().numpy()
    return np.mean((preds - y_pred) ** 2), np.mean((preds - y_true) ** 2)

mse_p, mse_t = eval_seq(seq)
print(f"{elapsed()}   After Hungarian:  MSE_pred={mse_p:.6f}  MSE_true={mse_t:.6f}")

# ── 6. Coordinate descent: swap blocks to minimize MSE vs pred ───────────────
print(f"\n{elapsed()} Step 6: Coordinate descent (target = pred)...")

def swap(seq, i, j):
    s = seq[:]
    s[i], s[j] = s[j], s[i]
    return s

improved = True
passes   = 0
total_pairs = n * (n - 1) // 2
while improved:
    improved = False
    passes  += 1
    t_pass  = time.time()
    swaps_this_pass = 0
    pbar = tqdm(total=total_pairs, desc=f"  Pass {passes}", unit="swap", leave=True)
    for i in range(n - 1):
        for j in range(i + 1, n):
            candidate    = swap(seq, i, j)
            mse_p_new, _ = eval_seq(candidate)
            if mse_p_new < mse_p:
                seq      = candidate
                mse_p    = mse_p_new
                improved = True
                swaps_this_pass += 1
                pbar.set_postfix(MSE_pred=f"{mse_p:.6f}", swaps=swaps_this_pass)
                if mse_p == 0.0:
                    pbar.close()
                    improved = False   # don't start another pass
                    break
            pbar.update(1)
        if mse_p == 0.0:
            break
    pbar.close()
    _, mse_t = eval_seq(seq)
    print(f"{elapsed()}   Pass {passes} done: {swaps_this_pass} swaps  "
          f"MSE_pred={mse_p:.6f}  MSE_true={mse_t:.6f}  "
          f"({time.time()-t_pass:.1f}s)")

print(f"\n{elapsed()} Converged after {passes} passes.")
print(f"{elapsed()} Final MSE_pred={mse_p:.6f}  MSE_true={mse_t:.6f}")

# ── 7. Save model and solution ────────────────────────────────────────────────
print(f"\n{elapsed()} Step 7: Saving model and solution...")

model = Model(num_blocks=n)
for i, (in_p, out_p) in enumerate(seq):
    model.blocks[i].inp.load_state_dict(
        torch.load(f"{PIECES_DIR}/piece_{in_p}.pth",  map_location="cpu"))
    model.blocks[i].out.load_state_dict(
        torch.load(f"{PIECES_DIR}/piece_{out_p}.pth", map_location="cpu"))
model.final.layer.load_state_dict(
    torch.load(f"{PIECES_DIR}/piece_{final_layer_piece}.pth", map_location="cpu"))
torch.save(model.state_dict(), "solved_model.pth")

solution = []
for in_p, out_p in seq:
    solution.append(in_p)
    solution.append(out_p)
solution.append(final_layer_piece)

with open("solution.txt", "w") as f:
    f.write(", ".join(str(x) for x in solution) + "\n")

total = time.time() - t0
print(f"\n{'='*60}")
print(f"  Solved in {total:.1f}s  ({total/60:.1f} min)")
print(f"  MSE_pred = {mse_p:.6f}")
print(f"  MSE_true = {mse_t:.6f}")
print(f"  Saved:   solved_model.pth,  solution.txt")
print(f"{'='*60}")
print("\nSolution (paste into puzzle):")
print(", ".join(str(x) for x in solution))
