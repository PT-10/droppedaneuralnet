import torch
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from model import Model

with open("paired_blocks.json") as f:
    data = json.load(f)

pairs       = data["pairs"]
final_piece = data["final_layer"][0]["piece"]

out_pieces = [p["block_out_piece"] for p in pairs]
in_pieces  = [p["block_in_piece"]  for p in pairs]

def load_weight(piece_idx):
    state = torch.load(f"pieces/piece_{piece_idx}.pth", map_location="cpu")
    return next(v for k, v in state.items() if len(v.shape) == 2).numpy()

def load_bias(piece_idx):
    state = torch.load(f"pieces/piece_{piece_idx}.pth", map_location="cpu")
    biases = [v for k, v in state.items() if len(v.shape) == 1]
    return biases[0].numpy() if biases else None

print("Loading weights for Hungarian re-pairing...")
W_out_all = [load_weight(p) for p in out_pieces]
W_in_all  = [load_weight(p) for p in in_pieces]
B_in_all  = [load_bias(p)   for p in in_pieces]

n = len(pairs)

#Re-run Hungarian to recover the optimised pairing
def norm01(m):
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo + 1e-12)

print("Re-running Hungarian to recover optimised pairing...")
trace_s   = np.zeros((n, n))
effrnk_s  = np.zeros((n, n))
bias_s    = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        M   = W_out_all[i] @ W_in_all[j]
        trace_s[i, j] = np.trace(np.abs(M))
        sv  = np.linalg.svd(M, compute_uv=False)
        sv_n = sv / (sv.sum() + 1e-12)
        sv_n = sv_n[sv_n > 1e-12]
        effrnk_s[i, j] = -np.exp(-np.sum(sv_n * np.log(sv_n)))
        if B_in_all[j] is not None:
            active = np.where(B_in_all[j] > 0)[0]
            if len(active) > 0:
                bias_s[i, j] = W_out_all[i][:, active].var(axis=0).mean()

combined = norm01(trace_s) + norm01(effrnk_s) + norm01(bias_s)
row_ind, col_ind = linear_sum_assignment(-combined)

# Matched pairs: (W_out[row_ind[k]], W_in[col_ind[k]])
matched_out = [out_pieces[i] for i in row_ind]
matched_in  = [in_pieces[j]  for j in col_ind]
W_out_matched = [W_out_all[i] for i in row_ind]
W_in_matched  = [W_in_all[j]  for j in col_ind]

#Build inter-block compatibility matrix
# Compatibility(A→B): how well block B's input space aligns with block A's output.
# Proxy: spectral alignment = sum of top-k singular values of  W_in[B] @ W_out[A]
# (the [96,48]@[48,96] = [96,96] product captures how much variance A passes to B)
print("Building block-sequence compatibility matrix...")
compat = np.zeros((n, n))
for a in range(n):
    for b in range(n):
        if a == b:
            continue
        M  = W_in_matched[b] @ W_out_matched[a]   # [96, 96]
        sv = np.linalg.svd(M, compute_uv=False)
        compat[a, b] = sv[:8].sum()               # top-8 singular values

def greedy_path(start, compat):
    n     = compat.shape[0]
    path  = [start]
    used  = {start}
    score = 0.0
    for _ in range(n - 1):
        last = path[-1]
        best_score, best_next = -np.inf, -1
        for nxt in range(n):
            if nxt not in used and compat[last, nxt] > best_score:
                best_score, best_next = compat[last, nxt], nxt
        path.append(best_next)
        used.add(best_next)
        score += best_score
    return path, score

print("Greedy nearest-neighbour search over all starting nodes...")
best_path, best_score = None, -np.inf
for start in range(n):
    path, score = greedy_path(start, compat)
    if score > best_score:
        best_score, best_path = score, path

#2-opt local search to refine
def path_score(path, compat):
    return sum(compat[path[i], path[i+1]] for i in range(len(path)-1))

def two_opt(path, compat, max_iter=2000):
    n      = len(path)
    improved = True
    iters  = 0
    while improved and iters < max_iter:
        improved = False
        iters   += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                if path_score(new_path, compat) > path_score(path, compat):
                    path     = new_path
                    improved = True
    return path

print("2-opt local search...")
best_path = two_opt(best_path, compat)
print(f"Final path score: {path_score(best_path, compat):.4f}")

#Evaluate both forward and reverse of the found sequence
df        = pd.read_csv("historical_data.csv")
feat_cols = [f"measurement_{i}" for i in range(48)]
X         = torch.tensor(df[feat_cols].values, dtype=torch.float32)
y_true    = df["true"].values
baseline  = np.mean((df["pred"].values - y_true) ** 2)

def build_and_eval(ordered_path, label):
    model = Model(num_blocks=n)
    for i, blk in enumerate(ordered_path):
        inp_s = torch.load(f"pieces/piece_{matched_in[blk]}.pth",  map_location="cpu")
        out_s = torch.load(f"pieces/piece_{matched_out[blk]}.pth", map_location="cpu")
        model.blocks[i].inp.load_state_dict(inp_s)
        model.blocks[i].out.load_state_dict(out_s)
    final_s = torch.load(f"pieces/piece_{final_piece}.pth", map_location="cpu")
    model.final.layer.load_state_dict(final_s)
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().numpy()
    mse = np.mean((preds - y_true) ** 2)
    print(f"  {label:30s}  MSE = {mse:.6f}")
    return mse

print("\n=== Results ===")
print(f"  {'Baseline pred column':30s}  MSE = {baseline:.6f}")
print(f"  {'Norm-sort inverted':30s}  MSE = 0.880204")
print(f"  {'Hungarian re-paired':30s}  MSE = 0.176944")
mse_fwd = build_and_eval(best_path,             "2-opt forward")
mse_rev = build_and_eval(list(reversed(best_path)), "2-opt reversed")
