# -*- coding: utf-8 -*-
"""
forecast_models.py
负责预测端：GM(1,1) 默认预测 + 回测诊断 + 兜底线性趋势预测
并提供统一的边界约束与诊断信息输出。

核心设计原则：
- 不允许“按国家手工修正参数”
- 只依赖历史数据与统一规则（可复现）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class ForecastResult:
    """单条序列预测结果（某国家-某指标）"""
    model_used: str                 # "GM11" or "LINEAR"
    mape_2025: float                # 回测误差（2016-2024 -> 2025）
    gm_a: Optional[float] = None
    gm_b: Optional[float] = None
    shift_c: float = 0.0            # 若做平移，则记录平移量
    bounds: Optional[Tuple[float, float]] = None
    hist_years: Optional[List[int]] = None
    hist_values: Optional[List[float]] = None
    pred_years: Optional[List[int]] = None
    pred_values: Optional[List[float]] = None


def infer_bounds(indicator_name: str, series: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    自动推断指标边界，用于截断预测值，避免物理不合理。
    规则（尽量保守、统一）：
    - 若数据最大值 <= 1.5 且最小值 >= 0：认为是比例型 [0,1]
    - 若指标名包含“率/覆盖/普及/渗透/信任”且最大值 <= 100 且最小值 >= 0：认为是百分比 [0,100]
    - 其他：默认非负 [0, +inf)（通过下界截断实现）
    """
    smin = float(np.nanmin(series))
    smax = float(np.nanmax(series))

    if smin >= 0 and smax <= 1.5:
        return (0.0, 1.0)

    key = indicator_name
    if any(k in key for k in ["率", "覆盖", "普及", "渗透", "信任"]) and smin >= 0 and smax <= 100.0:
        return (0.0, 100.0)

    # 非负型：只给下界；上界用 None 表示
    return (0.0, float("inf"))


def apply_bounds(x: np.ndarray, bounds: Optional[Tuple[float, float]]) -> np.ndarray:
    """对预测值进行边界截断。"""
    if bounds is None:
        return x
    lo, hi = bounds
    x = np.maximum(x, lo)
    if np.isfinite(hi):
        x = np.minimum(x, hi)
    return x


def mape(y_true: float, y_pred: float) -> float:
    """单点 MAPE（避免除0）"""
    if y_true is None or np.isnan(y_true):
        return np.nan
    denom = abs(y_true) if abs(y_true) > 1e-12 else 1e-12
    return float(abs(y_true - y_pred) / denom)


def gm11_fit_forecast(series: np.ndarray, n_ahead: int) -> Tuple[np.ndarray, float, float, float]:
    """
    GM(1,1) 拟合与预测。
    输入：
    - series: 长度 n 的历史序列（要求正值更稳定）
    - n_ahead: 预测步数
    输出：
    - pred: 长度 n_ahead 的预测值（对应未来 n_ahead 年）
    - a, b: GM参数
    - shift_c: 若为保证正值而平移的 c（最终预测已反平移）
    """
    x0 = series.astype(float).copy()
    n = len(x0)
    if n < 4:
        raise ValueError("GM(1,1) 需要至少 4 个点。")

    # 统一正值化：若存在 <=0，则做平移，保证序列严格为正
    minv = float(np.min(x0))
    shift_c = 0.0
    if minv <= 0:
        shift_c = abs(minv) + 1e-6
        x0 = x0 + shift_c

    # AGO
    x1 = np.cumsum(x0)

    # 紧邻均值
    z1 = 0.5 * (x1[1:] + x1[:-1])

    # 构造 B, Y
    B = np.column_stack((-z1, np.ones(n - 1)))
    Y = x0[1:].reshape(-1, 1)

    # 最小二乘
    # theta = (B^T B)^{-1} B^T Y
    BtB = B.T @ B
    if np.linalg.cond(BtB) > 1e12:
        # 病态矩阵，GM可能不稳定
        raise ValueError("GM(1,1) 矩阵病态，拟合不稳定。")

    theta = np.linalg.inv(BtB) @ B.T @ Y
    a = float(theta[0, 0])
    b = float(theta[1, 0])

    # 时间响应函数 x1_hat(k) for k=0..n+n_ahead-1 (对应 1..n+n_ahead)
    # 这里用 k 从 0 开始的等价写法
    x0_1 = x0[0]

    def x1_hat(k: int) -> float:
        return (x0_1 - b / a) * np.exp(-a * k) + b / a

    # 生成到未来的 x0_hat：差分还原
    # x0_hat(k+1) = x1_hat(k+1) - x1_hat(k), k>=0
    total = n + n_ahead
    x1h = np.array([x1_hat(k) for k in range(total)], dtype=float)
    x0h = np.diff(x1h, prepend=x1h[0])  # prepend使长度一致，但第一个点不使用

    # 未来预测部分：对应历史之后的 n_ahead 点
    pred = x0h[n:n + n_ahead].copy()

    # 反平移：GM在 x0 上做了平移，预测也需减回（差分后平移量不直接相减）
    # 注意：对x0做平移，AGO与差分后，预测的“增量”不需要直接减shift_c；
    # 但因为我们对原序列整体加了常数，GM拟合的动态会改变。工程上常用做法：
    # - 对原序列加常数后建模，输出的 x0_hat 仍在“平移空间”，需减回 shift_c 的影响。
    # 这里采用近似：将预测结果减去 shift_c*(1 - exp(-a)) 的等效项不稳定。
    # 更稳妥的简化：仅当 shift_c>0 时，使用“线性兜底”避免引入不透明修正。
    if shift_c > 0:
        raise ValueError("序列含非正值触发平移，GM结果不稳，建议兜底模型。")

    return pred, a, b, shift_c


def linear_trend_forecast(years: np.ndarray, series: np.ndarray, future_years: np.ndarray) -> np.ndarray:
    """
    兜底预测：线性趋势回归（最简、可解释、可复现）
    y = p1*year + p0
    """
    x = years.astype(float)
    y = series.astype(float)

    # 使用一阶多项式拟合
    p1, p0 = np.polyfit(x, y, deg=1)
    pred = p1 * future_years.astype(float) + p0
    return pred.astype(float)


def forecast_one_series(
    indicator_name: str,
    years: List[int],
    values: List[float],
    future_years: List[int],
    mape_threshold: float = 0.20,
) -> ForecastResult:
    """
    对单条序列（某国家-某指标）预测未来若干年。
    策略：
    1) 先用 GM(1,1) 进行单步回测（2016-2024 -> 2025）
    2) 若 GM 回测误差 <= 阈值 且拟合稳定，则用 GM 预测 2026-2035
    3) 否则启用线性趋势兜底预测
    4) 统一边界截断（按指标推断）
    """
    y = np.array(values, dtype=float)
    t = np.array(years, dtype=int)
    fy = np.array(future_years, dtype=int)

    # 推断边界
    bounds = infer_bounds(indicator_name, y)

    # ---------- GM 回测 ----------
    # 训练 2016-2024，预测 2025
    # 假设 years 已覆盖 2016..2025 连续
    try:
        train_mask = t <= 2024
        test_mask = t == 2025
        y_train = y[train_mask]
        y_true_2025 = float(y[test_mask][0])

        # GM预测一步
        pred_2025, a, b, shift_c = gm11_fit_forecast(y_train, n_ahead=1)
        y_pred_2025 = float(pred_2025[0])

        # 边界截断
        y_pred_2025 = float(apply_bounds(np.array([y_pred_2025]), bounds)[0])

        err = mape(y_true_2025, y_pred_2025)

        # 若回测表现合格，再用 GM 全训练预测未来
        if np.isfinite(err) and err <= mape_threshold:
            full_pred, a2, b2, shift_c2 = gm11_fit_forecast(y, n_ahead=len(fy))
            full_pred = apply_bounds(full_pred, bounds)

            return ForecastResult(
                model_used="GM11",
                mape_2025=float(err),
                gm_a=float(a2),
                gm_b=float(b2),
                shift_c=float(shift_c2),
                bounds=bounds,
                hist_years=list(years),
                hist_values=[float(v) for v in values],
                pred_years=list(future_years),
                pred_values=[float(v) for v in full_pred.tolist()],
            )

        # 否则走兜底
        raise ValueError("GM回测不合格，切换兜底模型。")

    except Exception:
        # ---------- 兜底：线性趋势 ----------
        pred = linear_trend_forecast(t, y, fy)
        pred = apply_bounds(pred, bounds)

        # 兜底模型也输出一个回测误差（用同样方式）
        # 用线性模型训练2016-2024预测2025
        train_mask = t <= 2024
        test_mask = t == 2025
        y_train = y[train_mask]
        t_train = t[train_mask]
        y_true_2025 = float(y[test_mask][0])

        y_pred_2025 = float(linear_trend_forecast(
            t_train, y_train, np.array([2025]))[0])
        y_pred_2025 = float(apply_bounds(np.array([y_pred_2025]), bounds)[0])
        err = mape(y_true_2025, y_pred_2025)

        return ForecastResult(
            model_used="LINEAR",
            mape_2025=float(err) if np.isfinite(err) else float("nan"),
            gm_a=None,
            gm_b=None,
            shift_c=0.0,
            bounds=bounds,
            hist_years=list(years),
            hist_values=[float(v) for v in values],
            pred_years=list(future_years),
            pred_values=[float(v) for v in pred.tolist()],
        )
