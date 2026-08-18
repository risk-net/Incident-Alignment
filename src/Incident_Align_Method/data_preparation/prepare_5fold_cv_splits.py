#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五折交叉验证数据划分脚本。

以 incident 为单位将 eval_structure.json 中的全部 incident
划分为 5 折，确保同一 incident 的所有 case 都在同一折中。

输出: outputs/prepared_5fold_cv/
  ├── fold_splits.json       # 总体划分定义
  ├── fold_01/test_incidents.json, dev_incidents.json, train_incidents.json
  ├── fold_02/...
  └── fold_05/...

用法:
  python src/Incident_Align_Method/data_preparation/prepare_5fold_cv_splits.py

配置 (可直接修改顶部常量):
  STRUCTURE_FILE: eval_structure.json 路径
  OUTPUT_DIR:    输出目录
  SEED:          随机种子

划分策略 (per fold i):
  Test  = fold i            (~350 incidents, 20%)
  Dev   = fold (i+1) % 5    (~350 incidents, 20%)
  Train = 其余 3 折          (~1052 incidents, 60%)

Train 部分供完整方法 (Deep+Wide) 使用；基线方法仅需 Dev + Test。

与现有 repeat 机制的区别:
  - repeat: 同一划分方式下不同随机种子 → 检验模型训练稳定性
  - 5-fold CV: 不同数据划分 → 检验方法对数据划分的鲁棒性
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from typing import Dict, List


# ── 配置 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STRUCTURE_FILE = PROJECT_ROOT / "data" / "eval_structure.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prepared_5fold_cv"
SEED = 42
N_FOLDS = 5
# ──────────────────────────────────────────────────────────────────


def norm_id(x):
    # type: (str) -> str
    return str(x).strip()


def load_incidents(structure_file):
    # type: (Path) -> Dict[str, List[str]]
    """加载 eval_structure.json，返回 {incident_id: [case_id, ...]}"""
    with open(structure_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    inc2cases = {}  # type: Dict[str, List[str]]
    for ev in data["events"]:
        inc_id = norm_id(ev["incident_id"])
        ids = [norm_id(x) for x in ev["ids"]]
        inc2cases[inc_id] = ids

    return inc2cases


def create_folds(inc2cases, n_folds=5, seed=42):
    # type: (Dict, int, int) -> List[List[str]]
    """将 incident 划分为 n_folds 折，返回 fold 定义列表"""
    inc_ids = sorted(inc2cases.keys())
    random.seed(seed)
    random.shuffle(inc_ids)

    # 尽可能均匀分配
    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for inc_id in inc_ids:
        case_count = len(inc2cases[inc_id])
        # 贪心：把当前 incident 分配给总 case 数最少的 fold
        min_idx = min(range(n_folds), key=lambda i: fold_sizes[i])
        folds[min_idx].append(inc_id)
        fold_sizes[min_idx] += case_count

    return folds


def build_split(folds, test_fold_idx):
    # type: (List[List[str]], int) -> Dict
    """为指定的 test fold 构建 train/dev/test 分配。

    Test  = folds[test_fold_idx]
    Dev   = folds[(test_fold_idx + 1) % len(folds)]
    Train = 其余 folds
    """
    n = len(folds)
    test_incidents = list(folds[test_fold_idx])
    dev_incidents = list(folds[(test_fold_idx + 1) % n])
    train_incidents = []
    for i in range(n):
        if i != test_fold_idx and i != (test_fold_idx + 1) % n:
            train_incidents.extend(folds[i])

    return {
        "test_incidents": sorted(test_incidents),
        "dev_incidents": sorted(dev_incidents),
        "train_incidents": sorted(train_incidents),
    }


def main():
    print("=" * 60)
    print("五折交叉验证数据划分")
    print("=" * 60)

    # 加载
    print(f"\n[1] 加载 {STRUCTURE_FILE}")
    inc2cases = load_incidents(STRUCTURE_FILE)
    n_incidents = len(inc2cases)
    n_cases = sum(len(v) for v in inc2cases.values())
    print(f"    Incidents: {n_incidents}")
    print(f"    Cases:     {n_cases}")

    # 统计 incident 大小分布
    sizes = [len(v) for v in inc2cases.values()]
    print(f"    Incident size: min={min(sizes)}, max={max(sizes)}, "
          f"mean={sum(sizes)/len(sizes):.1f}, median={sorted(sizes)[len(sizes)//2]}")

    # 划分
    print(f"\n[2] 划分为 {N_FOLDS} 折 (seed={SEED})")
    folds = create_folds(inc2cases, n_folds=N_FOLDS, seed=SEED)

    for i, fold in enumerate(folds):
        fold_cases = sum(len(inc2cases[inc_id]) for inc_id in fold)
        print(f"    Fold {i+1}: {len(fold)} incidents, {fold_cases} cases")

    # 构建每折的 split 并保存
    print(f"\n[3] 构建 train/dev/test 划分并保存到 {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fold_summary = {
        "description": "5-fold cross-validation splits at incident level",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "total_incidents": n_incidents,
        "total_cases": n_cases,
        "structure_file": str(STRUCTURE_FILE.resolve()),
        "strategy": "Per fold i: Test=fold_i, Dev=fold_(i+1)%5, Train=rest",
        "folds": [],
    }

    for i in range(N_FOLDS):
        fold_name = f"fold_{i+1:02d}"
        fold_dir = OUTPUT_DIR / fold_name
        os.makedirs(fold_dir, exist_ok=True)

        split = build_split(folds, i)

        # 计算 case 数
        test_cases = sum(len(inc2cases[inc_id]) for inc_id in split["test_incidents"])
        dev_cases = sum(len(inc2cases[inc_id]) for inc_id in split["dev_incidents"])
        train_cases = sum(len(inc2cases[inc_id]) for inc_id in split["train_incidents"])

        # 保存每折
        for key in ["test_incidents", "dev_incidents", "train_incidents"]:
            file_path = fold_dir / f"{key}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(split[key], f, ensure_ascii=False, indent=2)

        fold_info = {
            "fold_name": fold_name,
            "test_fold_idx": i,
            "n_test_incidents": len(split["test_incidents"]),
            "n_dev_incidents": len(split["dev_incidents"]),
            "n_train_incidents": len(split["train_incidents"]),
            "n_test_cases": test_cases,
            "n_dev_cases": dev_cases,
            "n_train_cases": train_cases,
        }
        fold_summary["folds"].append(fold_info)

        print(f"    {fold_name}: test={len(split['test_incidents'])} incidents "
              f"({test_cases} cases), "
              f"dev={len(split['dev_incidents'])} incidents ({dev_cases} cases), "
              f"train={len(split['train_incidents'])} incidents ({train_cases} cases)")

    # 保存总体描述
    summary_path = OUTPUT_DIR / "fold_splits.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, ensure_ascii=False, indent=2)
    print(f"\n    Summary → {summary_path}")

    # 验证
    print(f"\n[4] 验证划分正确性")
    all_test_incidents = set()
    for i in range(N_FOLDS):
        split = build_split(folds, i)
        all_test_incidents.update(split["test_incidents"])

        # 检查 train/dev/test 互斥
        train_set = set(split["train_incidents"])
        dev_set = set(split["dev_incidents"])
        test_set = set(split["test_incidents"])
        assert train_set.isdisjoint(dev_set), f"Fold {i}: train ∩ dev ≠ ∅"
        assert train_set.isdisjoint(test_set), f"Fold {i}: train ∩ test ≠ ∅"
        assert dev_set.isdisjoint(test_set), f"Fold {i}: dev ∩ test ≠ ∅"

    # 每个 incident 恰好出现在 test 中一次
    assert all_test_incidents == set(inc2cases.keys()), \
        f"Union of test folds ≠ all incidents! diff={all_test_incidents ^ set(inc2cases.keys())}"
    print(f"    ✅ 所有 {n_incidents} 个 incident 恰好被测试一次")
    print(f"    ✅ train/dev/test 互斥")

    print(f"\n{'=' * 60}")
    print("完成。下一步:")
    print(f"  逻辑回归基线 五折CV:")
    print(f"    python src/Incident_Align_Evaluation/baseline_logistic_regression_cv.py")
    print(f"  完整方法 五折CV:")
    print(f"    python src/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise_5fold.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
