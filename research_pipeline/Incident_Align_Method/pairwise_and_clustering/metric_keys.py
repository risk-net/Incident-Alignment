#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指标字段名的唯一事实来源（single source of truth）。

本项目统一使用 ``src/incident_alignment/`` 核心库的命名：

    hungarian_f1_g2p / hungarian_f1_p2g / hungarian_f1_symmetric

（旧训练链路里的 ``event_macro_f1_hungarian`` 等命名已全部废弃，不再保留别名。）

其余指标与核心库同名，未做改动：
    b_cubed_f1 / b_cubed_precision / b_cubed_recall
    induced_pair_f1 / induced_pair_precision / induced_pair_recall
    ari / nmi / pair_* / num_* / threshold
"""
from __future__ import annotations

# 报告里展示的规范指标（顺序即表格列顺序）
CANONICAL_METRICS = [
    "hungarian_f1_g2p",             # 主指标：Hungarian G→P event macro-F1
    "hungarian_f1_p2g",
    "hungarian_f1_symmetric",
    "b_cubed_f1",
    "b_cubed_precision",
    "b_cubed_recall",
    "induced_pair_f1",
    "induced_pair_precision",
    "induced_pair_recall",
    "ari",
    "pair_macro_f1",
    "pair_f1_pos",
    "pair_precision_pos",
    "pair_recall_pos",
    "num_pred_clusters",
    "num_gold_events",
    "threshold",
]


def require_metric(metrics: dict, name: str):
    """取规范指标；缺失即报错（本项目不保留旧键，不存在兜底）。"""
    if name not in metrics:
        raise KeyError(
            f"指标 {name!r} 不存在。已有键：{sorted(metrics)[:20]}"
        )
    return metrics[name]
