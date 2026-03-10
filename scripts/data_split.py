#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
from collections import defaultdict

# ================== 路径配置 ==================
IN_PATH = Path("/home/qygao/matten/datasets/XequiNet/jarvis_dft/bec/bec_from_hdf5_2.json")

OUT_TRAIN = Path("/home/qygao/matten/datasets/XequiNet/jarvis_dft/bec/bec_from_hdf5_20_train.json")
OUT_VAL   = Path("/home/qygao/matten/datasets/XequiNet/jarvis_dft/bec/bec_from_hdf5_20_val.json")
OUT_TEST  = Path("/home/qygao/matten/datasets/XequiNet/jarvis_dft/bec/bec_from_hdf5_20_test.json")

OUT_SPLIT_META = Path("/home/qygao/matten/datasets/XequiNet/jarvis_dft/bec/bec_from_hdf5_20_split.json")
# ==============================================

# ======== 可调参数 ========
seed = 35
train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1
# =========================
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

# ------------------------------------------------------------
# 结构元素提取：优先 pymatgen，失败则用 dict 解析兜底
# ------------------------------------------------------------
def get_elements_from_structure_dict(struct_d: dict) -> set:
    # 1) 尝试 pymatgen（最稳）
    try:
        from pymatgen.core import Structure
        s = Structure.from_dict(struct_d)
        return {str(el) for el in s.composition.elements}
    except Exception:
        pass

    # 2) 兜底：解析 sites -> species
    elems = set()
    sites = struct_d.get("sites", [])
    for site in sites:
        sp = site.get("species", None)
        if sp is None:
            sp = site.get("species_and_occu", None)

        # 常见：list[{"element":"Si","occu":1}, ...]
        if isinstance(sp, list):
            for item in sp:
                if isinstance(item, dict):
                    if "element" in item:
                        elems.add(str(item["element"]))
                    elif "symbol" in item:
                        elems.add(str(item["symbol"]))
                    elif "name" in item:
                        elems.add(str(item["name"]))
        # 也可能：dict {"Si":1}
        elif isinstance(sp, dict):
            for k in sp.keys():
                # k 可能是元素符号或包含更多信息
                elems.add(str(k))
        # 也可能：字符串 "Si"
        elif isinstance(sp, str):
            elems.add(sp)

    return elems

# ------------------------------------------------------------
# split 工具
# ------------------------------------------------------------
def subset_by_ids(full_data: dict, ids_subset: list, split_name: str) -> dict:
    ids_set = set(ids_subset)
    out = {}
    for k, v in full_data.items():
        if isinstance(v, dict):
            out[k] = {i: v[i] for i in v.keys() if i in ids_set}
        else:
            out[k] = v
    out["split"] = {i: split_name for i in ids_subset}
    return out

def union_elems(ids_list, elems_per_id):
    u = set()
    for i in ids_list:
        u |= elems_per_id[i]
    return u

def train_element_counts(train_ids, elems_per_id):
    cnt = defaultdict(int)
    for i in train_ids:
        for e in elems_per_id[i]:
            cnt[e] += 1
    return cnt

def check_subset_constraint(train_ids, val_ids, test_ids, elems_per_id):
    E_train = union_elems(train_ids, elems_per_id)
    E_val   = union_elems(val_ids, elems_per_id)
    E_test  = union_elems(test_ids, elems_per_id)
    return (E_val.issubset(E_train) and E_test.issubset(E_train)), (E_train, E_val, E_test)

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
data = json.load(open(IN_PATH, "r", encoding="utf-8"))

# 用 structure.keys() 作为参与划分的 id，并过滤掉 structure 缺失
all_ids = sorted(list(data.get("structure", {}).keys()))
all_ids = [i for i in all_ids if data["structure"].get(i) is not None]
N = len(all_ids)
assert N > 0, "No valid structures found in data['structure']"

# 预计算每个样本包含的元素集合
elems_per_id = {}
bad = []
for i in all_ids:
    try:
        elems = get_elements_from_structure_dict(data["structure"][i])
        if not elems:
            bad.append(i)
        elems_per_id[i] = elems
    except Exception:
        bad.append(i)

if bad:
    print(f"[WARN] {len(bad)} samples have empty/unparsable elements. They will be dropped from splitting.")
    all_ids = [i for i in all_ids if i not in set(bad)]
    N = len(all_ids)
    assert N > 0, "All samples were dropped due to unparsable structures."

rng = np.random.default_rng(seed)
perm = rng.permutation(N)

# 目标数量（尽量）
n_train_t = int(round(N * train_ratio))
n_val_t   = int(round(N * val_ratio))
n_test_t  = N - n_train_t - n_val_t
assert n_test_t >= 0

train_ids = [all_ids[i] for i in perm[:n_train_t]]
val_ids   = [all_ids[i] for i in perm[n_train_t:n_train_t + n_val_t]]
test_ids  = [all_ids[i] for i in perm[n_train_t + n_val_t:]]

# ------------------------------------------------------------
# Step A: 强制覆盖：把 val/test 中 train 未覆盖的元素对应样本搬回 train
# ------------------------------------------------------------
def enforce_coverage(train_ids, val_ids, test_ids, elems_per_id):
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    # 反复搬运直到满足 E_val/E_test ⊆ E_train
    changed = True
    while changed:
        changed = False
        E_train = union_elems(list(train_set), elems_per_id)

        # val -> train
        for i in list(val_set):
            if not elems_per_id[i].issubset(E_train):
                val_set.remove(i)
                train_set.add(i)
                changed = True
                # 更新 E_train（加速）
                E_train |= elems_per_id[i]

        # test -> train
        for i in list(test_set):
            if not elems_per_id[i].issubset(E_train):
                test_set.remove(i)
                train_set.add(i)
                changed = True
                E_train |= elems_per_id[i]

    return list(train_set), list(val_set), list(test_set)

train_ids, val_ids, test_ids = enforce_coverage(train_ids, val_ids, test_ids, elems_per_id)

ok, (E_train, E_val, E_test) = check_subset_constraint(train_ids, val_ids, test_ids, elems_per_id)
assert ok, "Coverage constraint failed after enforcement (this should not happen)."

# ------------------------------------------------------------
# Step B: 尽量把比例拉回 8/1/1（但绝不破坏覆盖约束）
#
# 关键点：从 train 往外挪样本时，必须保证挪出去的样本元素在 train 里仍至少出现一次
# 否则它一挪到 val/test，就会引入 train 未覆盖元素，违反 E_val/E_test ⊆ E_train
# ------------------------------------------------------------
def rebalance_sizes(train_ids, val_ids, test_ids, elems_per_id, n_train_t, n_val_t, n_test_t, rng):
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    # helper: 选择一个可从 train 移出的“安全样本”
    def pick_removable_from_train(train_set, elems_per_id, rng):
        cnt = train_element_counts(list(train_set), elems_per_id)
        candidates = []
        for i in train_set:
            # i 可移出条件：它包含的每个元素在 train 中出现次数都 > 1
            # 这样移出后 train 仍保有这些元素 -> 覆盖不会被破坏
            good = True
            for e in elems_per_id[i]:
                if cnt[e] <= 1:
                    good = False
                    break
            if good:
                candidates.append(i)
        if not candidates:
            return None
        return candidates[int(rng.integers(0, len(candidates)))]

    # 1) 若 train 过大，往 val/test 挪
    #    优先补齐 val/test 到 target，其次随便分配
    while len(train_set) > n_train_t:
        # 目标：尽量让 val/test 接近目标
        need_val  = n_val_t  - len(val_set)
        need_test = n_test_t - len(test_set)
        if need_val > need_test:
            target = "val"
        else:
            target = "test"

        move_id = pick_removable_from_train(train_set, elems_per_id, rng)
        if move_id is None:
            # 没有可安全移出的样本 -> 无法维持 8/1/1，只能接受 train 偏大
            break

        train_set.remove(move_id)
        if target == "val":
            val_set.add(move_id)
        else:
            test_set.add(move_id)

    # 2) 若 train 过小（较少见：一般只会因你设定目标导致）
    #    从 val/test 随机搬回 train（这不会破坏覆盖，只会更“安全”）
    def move_back(src_set):
        if not src_set:
            return False
        i = list(src_set)[int(rng.integers(0, len(src_set)))]
        src_set.remove(i)
        train_set.add(i)
        return True

    while len(train_set) < n_train_t:
        # 从更“超额”的集合搬
        if len(val_set) > n_val_t:
            if not move_back(val_set):
                break
        elif len(test_set) > n_test_t:
            if not move_back(test_set):
                break
        else:
            # 都不超额，随便搬一个（比例会略偏）
            if len(val_set) >= len(test_set):
                if not move_back(val_set):
                    break
            else:
                if not move_back(test_set):
                    break

    return list(train_set), list(val_set), list(test_set)

train_ids, val_ids, test_ids = rebalance_sizes(train_ids, val_ids, test_ids, elems_per_id,
                                               n_train_t, n_val_t, n_test_t, rng)

# 最终校验
ok, (E_train, E_val, E_test) = check_subset_constraint(train_ids, val_ids, test_ids, elems_per_id)
assert ok, "Final coverage constraint failed (should not happen)."

# 打乱各 split 内部顺序（可选）
rng.shuffle(train_ids)
rng.shuffle(val_ids)
rng.shuffle(test_ids)

# 输出文件
for p in [OUT_TRAIN, OUT_VAL, OUT_TEST, OUT_SPLIT_META]:
    p.parent.mkdir(parents=True, exist_ok=True)

train_data = subset_by_ids(data, train_ids, "train")
val_data   = subset_by_ids(data, val_ids, "val")
test_data  = subset_by_ids(data, test_ids, "test")

json.dump(train_data, open(OUT_TRAIN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(val_data,   open(OUT_VAL,   "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(test_data,  open(OUT_TEST,  "w", encoding="utf-8"), ensure_ascii=False, indent=2)

meta = {
    "seed": seed,
    "targets": {"train": n_train_t, "val": n_val_t, "test": n_test_t},
    "final_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids), "N": N},
    "final_ratios": {"train": len(train_ids)/N, "val": len(val_ids)/N, "test": len(test_ids)/N},
    "coverage_check": {
        "E_val_subset_E_train": union_elems(val_ids, elems_per_id).issubset(union_elems(train_ids, elems_per_id)),
        "E_test_subset_E_train": union_elems(test_ids, elems_per_id).issubset(union_elems(train_ids, elems_per_id)),
        "num_elements_train": len(union_elems(train_ids, elems_per_id)),
        "num_elements_val": len(union_elems(val_ids, elems_per_id)),
        "num_elements_test": len(union_elems(test_ids, elems_per_id)),
        "elements_train": sorted(list(union_elems(train_ids, elems_per_id))),
        "elements_val_only": sorted(list(union_elems(val_ids, elems_per_id) - union_elems(train_ids, elems_per_id))),
        "elements_test_only": sorted(list(union_elems(test_ids, elems_per_id) - union_elems(train_ids, elems_per_id))),
    }
}
json.dump(meta, open(OUT_SPLIT_META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("saved:")
print("  train ->", OUT_TRAIN, "N =", len(train_ids))
print("  val   ->", OUT_VAL,   "N =", len(val_ids))
print("  test  ->", OUT_TEST,  "N =", len(test_ids))
print("target counts:", n_train_t, n_val_t, n_test_t)
print("final ratios:", len(train_ids)/N, len(val_ids)/N, len(test_ids)/N)
print("coverage: E_val ⊆ E_train =", meta["coverage_check"]["E_val_subset_E_train"],
      ", E_test ⊆ E_train =", meta["coverage_check"]["E_test_subset_E_train"])
if meta["coverage_check"]["elements_val_only"] or meta["coverage_check"]["elements_test_only"]:
    # 理论上应为空
    print("[WARN] Some elements appear in val/test but not in train:",
          meta["coverage_check"]["elements_val_only"],
          meta["coverage_check"]["elements_test_only"])
else:
    print("OK: no unseen elements in val/test relative to train.")
print("meta ->", OUT_SPLIT_META)
