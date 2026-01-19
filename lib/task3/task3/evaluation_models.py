# -*- coding: utf-8 -*-
"""
evaluation_models.py
负责评估端：读取问题二权重（若存在） + TOPSIS 逐年计算得分与排名。

重要原则：
- 问题三必须继承问题二的评估逻辑（权重 + TOPSIS）
- 若找不到权重文件，则用等权作为兜底（并在日志中提示）
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def load_weights(outputs_dir, indicators):
    """
    读取问题二输出的熵权文件 weights_entropy.csv
    文件格式：
        Indicator, Entropy, Redundancy, Weight
    """

    # === 关键修复：强制转换为 Path（防御式） ===
    outputs_dir = Path(outputs_dir)

    weight_file = outputs_dir / "weights_entropy.csv"

    if not weight_file.exists():
        print("[Warn] 未找到 weights_entropy.csv，已使用等权。")
        return pd.Series(
            [1.0 / len(indicators)] * len(indicators),
            index=indicators,
            name="weight"
        )

    df = pd.read_csv(weight_file, encoding="utf-8-sig")

    required_cols = {"Indicator", "Weight"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"weights_entropy.csv 必须包含列 {required_cols}，当前列为 {df.columns.tolist()}"
        )

    weight_map = dict(zip(df["Indicator"], df["Weight"]))

    missing = [ind for ind in indicators if ind not in weight_map]
    if missing:
        raise ValueError(
            f"weights_entropy.csv 缺少以下指标权重：{missing}"
        )

    weights = pd.Series(
        [float(weight_map[ind]) for ind in indicators],
        index=indicators,
        name="weight"
    )

    # 再归一一次（防御性）
    weights = weights / weights.sum()

    return weights



def topsis_score(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    TOPSIS 得分计算（默认所有指标为效益型：越大越好）
    输入：
    - matrix: shape (n_country, n_indicator)
    - weights: shape (n_indicator,)
    输出：
    - scores: shape (n_country,) in [0,1]
    """
    X = matrix.astype(float)

    # 向量归一化
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1e-12
    R = X / denom

    # 加权
    W = weights.reshape(1, -1)
    Z = R * W

    # 正负理想解
    z_pos = np.max(Z, axis=0)
    z_neg = np.min(Z, axis=0)

    # 距离
    d_pos = np.sqrt(((Z - z_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((Z - z_neg) ** 2).sum(axis=1))

    # 贴近度
    scores = d_neg / (d_pos + d_neg + 1e-12)
    return scores


def rank_desc(scores: np.ndarray) -> np.ndarray:
    """
    按得分从大到小排名，返回名次（1为第一）
    """
    order = np.argsort(-scores)  # 降序
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks
