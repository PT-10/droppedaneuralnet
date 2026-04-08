# Dropped a Neural Net

Solution to the [Jane Street puzzle](https://huggingface.co/spaces/jane-street/droppedaneuralnet).

97 anonymous weight tensors from a disassembled 48-block residual network. Goal: recover the original layer permutation.

---

## Usage

```bash
python solve.py      # recover permutation, saves solved_model.pth + solution.txt
python predict.py    # evaluate any model and print solution string
```

`predict.py` accepts an optional model path: `python predict.py path/to/model.pth`

---

## Approach

Three stages, each narrowing the search:

**1. Shape classification**
Pieces are sorted by tensor shape into `block_in` [96,48], `block_out` [48,96], and `final` [1,48].

**2. Hungarian pairing**
Within each block, `W_in` and `W_out` co-adapt during training. Three metrics score candidate pairs:
- Trace of `|W_out @ W_in|` (diagonal alignment)
- Negative effective rank of `W_out @ W_in` (co-adaptation)
- Column variance at active ReLU units (bias-activation match)

The Hungarian algorithm finds the globally optimal 48-to-48 assignment.

**3. Coordinate descent**
Block positions are refined by trying all pairwise swaps and accepting any that reduce MSE against the `pred` column (the original model's exact output). Terminates when MSE = 0.

Note: blocks are ordered by ascending Frobenius norm (low norm = shallow). The intuitive descending sort produced MSE=449; ascending gave MSE=0.880 as the starting point for descent.

---

## Results

| Stage | MSE vs pred | MSE vs true |
|---|---|---|
| Norm sort + Hungarian | 0.076 | 0.177 |
| + Coordinate descent | **0.000** | **0.106** |

Runtime: ~10 min on CPU.

---
