#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch

from matten.dataset.structure_scalar_tensor import TensorDataModule
from matten.model_factory.tfn_scalar_tensor import ScalarTensorModel

# 你之前用过这个（MatTen 项目里一般都有）
from matten.utils import CartesianTensorWrapper


def load_yaml(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_3x3_from_irreps(x: torch.Tensor, formula="ij=ji") -> torch.Tensor:
    """
    x: (B, 6) irreps of symmetric rank-2 tensor (0e+2e)
    return: (B, 3, 3) cartesian symmetric tensor
    """
    wrapper = CartesianTensorWrapper(formula=formula)
    # wrapper.to_cartesian 接受 (B,6) -> (B,3,3)（MatTen里通常就是这样）
    A = wrapper.to_cartesian(x)
    # 防御：强制对称
    A = 0.5 * (A + A.transpose(-1, -2))
    return A


def mae_ten_from_cartesian(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    pred,true: (B,3,3) symmetric
    compute paper's MAE_ten: avg over 6 independent components and batch
    """
    err = (pred - true).abs()
    # 6 independent components: (0,0)(1,1)(2,2)(0,1)(0,2)(1,2)
    s = (
        err[:, 0, 0] + err[:, 1, 1] + err[:, 2, 2] +
        err[:, 0, 1] + err[:, 0, 2] + err[:, 1, 2]
    )
    return s.mean() / 6.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    task_name = cfg["data"]["tensor_target_name"]  # dielectric_tensor

    # datamodule：测试别shuffle
    cfg_data = dict(cfg["data"])
    lk = dict(cfg_data.get("loader_kwargs", {}))
    lk["shuffle"] = False
    cfg_data["loader_kwargs"] = lk

    dm = TensorDataModule(
        **cfg_data,
        normalize_tensor_target=False,
        compute_dataset_statistics=False,
    )
    dm.prepare_data()
    try:
        dm.setup(stage="test")
    except TypeError:
        dm.setup()
    loader = dm.test_dataloader()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = ScalarTensorModel.load_from_checkpoint(args.ckpt, map_location="cpu", strict=False)
    model = model.to(device).eval()

    # 累计 MAE_ten
    total = 0.0
    count = 0

    for batch in loader:
        # 关键：显式 task_name，避免默认 elastic_tensor_full
        preds, labels = model(batch, task_name=task_name)

        y_pred = preds[task_name]
        y_true = labels[task_name]

        # 放到 device
        y_pred = y_pred.to(device)
        y_true = y_true.to(device)

        # 统一到 3x3 再算 MAE_ten
        if y_pred.ndim == 2 and y_pred.shape[1] == 6:
            P = to_3x3_from_irreps(y_pred, formula="ij=ji")
        elif y_pred.ndim == 3 and y_pred.shape[1:] == (3, 3):
            P = 0.5 * (y_pred + y_pred.transpose(-1, -2))
        else:
            raise RuntimeError(f"Unsupported pred shape: {tuple(y_pred.shape)}")

        if y_true.ndim == 2 and y_true.shape[1] == 6:
            T = to_3x3_from_irreps(y_true, formula="ij=ji")
        elif y_true.ndim == 3 and y_true.shape[1:] == (3, 3):
            T = 0.5 * (y_true + y_true.transpose(-1, -2))
        else:
            raise RuntimeError(f"Unsupported true shape: {tuple(y_true.shape)}")

        batch_mae_ten = mae_ten_from_cartesian(P, T)  # scalar tensor
        bsz = P.shape[0]
        total += batch_mae_ten.item() * bsz
        count += bsz

    mae_ten = total / max(count, 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "ckpt": args.ckpt,
        "task_name": task_name,
        "device": str(device),
        "num_graphs": int(count),
        "MAE_ten": float(mae_ten),
        "definition": "MAE_ten = (1/(6B)) * sum_b sum_{i<=j} |eps_ij(true)-eps_ij(pred)|",
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("Saved:", str(out_path))
    print("MAE_ten =", mae_ten)


if __name__ == "__main__":
    main()
