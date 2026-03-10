#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

TARGET_Z = 89  # Ac

# 如果你有 pymatgen，能更稳地从元素符号拿到原子序数
try:
    from pymatgen.core.periodic_table import Element
    HAVE_PYMATGEN = True
except Exception:
    HAVE_PYMATGEN = False


def load_json_any(path: Path):
    """
    兼容三种格式：
    1) records: list[dict]
    2) jsonlines: 每行一个 dict
    3) pandas columns: dict-of-columns（{"col":{"0":...}}）
    """
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []

    # jsonlines（每行一个 JSON 对象）
    if "\n" in txt and not txt.lstrip().startswith("[") and not txt.lstrip().startswith("{"):
        rows = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    obj = json.loads(txt)

    # records
    if isinstance(obj, list):
        return obj

    # dict-of-columns -> records
    if isinstance(obj, dict):
        # 判断是否像 {"col":{"0":...,"1":...}, ...}
        # 取任意一列看看是不是 dict 且 key 像数字字符串
        any_val = next(iter(obj.values()), None)
        if isinstance(any_val, dict):
            keys = list(any_val.keys())
            if keys and all(str(k).isdigit() for k in keys[: min(20, len(keys))]):
                # 按行号组装 records
                idxs = sorted(any_val.keys(), key=lambda x: int(x))
                rows = []
                for i in idxs:
                    row = {}
                    for col, colmap in obj.items():
                        if isinstance(colmap, dict) and i in colmap:
                            row[col] = colmap[i]
                    rows.append(row)
                return rows

    raise ValueError(f"{path} 无法识别的 JSON 结构：{type(obj)}")



def dump_json_records(path: Path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def site_to_Z(site):
    """
    从一个 pymatgen-style site dict 里尽量提取原子序数
    常见结构：
      site["species"][0]["element"] -> "Si"
      site["species"][0]["atomic_number"] -> 14
      site["atomic_number"] -> 14
    """
    if isinstance(site, dict):
        if "atomic_number" in site:
            try:
                return int(site["atomic_number"])
            except Exception:
                pass

        sp = site.get("species", None)
        if isinstance(sp, list) and len(sp) > 0:
            sp0 = sp[0]
            if isinstance(sp0, dict):
                if "atomic_number" in sp0:
                    try:
                        return int(sp0["atomic_number"])
                    except Exception:
                        pass
                if "element" in sp0 and HAVE_PYMATGEN:
                    try:
                        return int(Element(sp0["element"]).Z)
                    except Exception:
                        pass
    return None


def get_atomic_numbers(row):
    """
    尽可能从 row 中拿到原子序数列表
    优先：
      row["atomic_numbers"]
      row["structure"]["sites"][*]
    """
    # 1) 直接给了 atomic_numbers
    if isinstance(row, dict) and "atomic_numbers" in row:
        z = row["atomic_numbers"]
        if isinstance(z, list):
            out = []
            for x in z:
                try:
                    out.append(int(x))
                except Exception:
                    pass
            return out

    # 2) pymatgen Structure dict
    struct = row.get("structure", None) if isinstance(row, dict) else None
    if isinstance(struct, dict):
        sites = struct.get("sites", None)
        if isinstance(sites, list):
            zs = []
            for s in sites:
                z = site_to_Z(s)
                if z is not None:
                    zs.append(z)
            if zs:
                return zs

    return []


def filter_file(in_path: Path, out_path: Path):
    rows = load_json_any(in_path)
    kept, removed = [], []
    for i, row in enumerate(rows):
        zs = get_atomic_numbers(row)
        if TARGET_Z in zs:
            removed.append(i)
        else:
            kept.append(row)

    dump_json_records(out_path, kept)
    print(f"[{in_path.name}] total={len(rows)} kept={len(kept)} removed={len(removed)}")
    if removed[:10]:
        print(f"  removed indices (first 10): {removed[:10]}")


def main():
    base = Path("/home/qygao/matten/datasets/di_pizeoelectric_tensor")  # <<< 改这里
    # 按你的真实文件名改这里
    files = [
        ("dielectric_tensor_train_max_300.json", "dielectric_tensor_train_no89.json"),
        ("dielectric_tensor_val_max_300.json",   "dielectric_tensor_val_no89.json"),
        ("dielectric_tensor_test_max_300.json",  "dielectric_tensor_test_no89.json"),
    ]

    for a, b in files:
        in_path = base / a
        out_path = base / b
        if not in_path.exists():
            raise FileNotFoundError(f"找不到：{in_path}")
        filter_file(in_path, out_path)

    print("\nDone. 记得在训练配置里把数据文件路径改成 *_no89.json")


if __name__ == "__main__":
    main()
