import torch
import json

weight_indices = {"block_out": [], "block_in": [], "final_layer": []}

for i in range(97):
    pth_data = torch.load(f"pieces/piece_{i}.pth", map_location="cpu")

    if isinstance(pth_data, dict):
        for k, v in pth_data.items():

            if len(v.shape) == 2:

                frob = torch.norm(v, p="fro").item()

                s = torch.linalg.svdvals(v)
                spectral = s[0].item()

                sharpness = (s[0] / s.sum()).item()

                stable_rank = (frob ** 2) / (spectral ** 2)

                entry = {
                    "piece": i,
                    "frobenius_norm": frob,
                    "sharpness": sharpness,
                    "stable_rank": stable_rank
                }

                if v.shape == torch.Size([96, 48]):
                    weight_indices["block_in"].append(entry)

                elif v.shape == torch.Size([48, 96]):
                    weight_indices["block_out"].append(entry)

                elif v.shape == torch.Size([1, 48]):
                    weight_indices["final_layer"].append(entry)


# STEP A — sort by Frobenius norm
weight_indices["block_out"].sort(
    key=lambda x: x["frobenius_norm"], reverse=True
)

weight_indices["block_in"].sort(
    key=lambda x: x["frobenius_norm"], reverse=True
)

# STEP B — optional refinement with stable rank
# (lower stable rank → deeper layer)
weight_indices["block_out"].sort(
    key=lambda x: (x["frobenius_norm"], -x["stable_rank"]),
    reverse=True
)

weight_indices["block_in"].sort(
    key=lambda x: (x["frobenius_norm"], -x["stable_rank"]),
    reverse=True
)

# STEP C — pair blocks
pairs = []
for out_layer, in_layer in zip(weight_indices["block_out"], weight_indices["block_in"]):
    pairs.append({
        "block_out_piece": out_layer["piece"],
        "block_in_piece": in_layer["piece"],
        "out_norm": out_layer["frobenius_norm"],
        "in_norm": in_layer["frobenius_norm"],
        "out_stable_rank": out_layer["stable_rank"],
        "in_stable_rank": in_layer["stable_rank"]
    })

result = {
    "pairs": pairs,
    "final_layer": weight_indices["final_layer"]
}

with open("paired_blocks.json", "w") as f:
    json.dump(result, f, indent=4)