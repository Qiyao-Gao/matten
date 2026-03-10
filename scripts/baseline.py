import json
import torch

path_train = "../datasets/di_pizeoelectric_tensor/dielectric_tensor_train.json"
path_val = "../datasets/di_pizeoelectric_tensor/dielectric_tensor_val.json"

def load_tensors(path):
    with open(path, "r") as f:
        data = json.load(f)
    # 这里按你 json 里真正的字段名改一下
    tensors = [d["dielectric_tensor"] for d in data]
    return torch.tensor(tensors, dtype=torch.float32)  # [N,3,3]

train_t = load_tensors(path_train)
val_t = load_tensors(path_val)

mean_t = train_t.mean(dim=0)          # [3,3]
mae_baseline = (val_t - mean_t).abs().mean().item()

print("Baseline MAE (always predict mean tensor):", mae_baseline)
