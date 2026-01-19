# -*- coding: utf-8 -*-
"""
mainpip.py
问题三主入口：读取数据 -> 预测 2026-2035 指标 -> TOPSIS逐年得分与排名 -> 输出 outputs/

运行方式：
    python mainpip.py

目录要求：
- mainpip.py 与 DATA/ 同级
- 结果输出至 outputs/
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from data_loader import load_all_data, default_indicator_specs, COUNTRIES_CN
from forecast_models import forecast_one_series
from evaluation_models import load_weights, topsis_score, rank_desc

import os

DATADIR = os.path.dirname(os.path.abspath(__file__))

def ensure_outputs_dir(base_dir: Path) -> Path:
    out = base_dir / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_panel_wide(panel_long: pd.DataFrame, year: int, indicators: List[str], countries: List[str]) -> pd.DataFrame:
    """
    构造某一年 t 的 10x24 宽表矩阵：
    行=国家，列=指标
    """
    df = panel_long[panel_long["year"] == year].copy()
    pivot = df.pivot_table(
        index="country", columns="indicator", values="value", aggfunc="first")

    # 对齐顺序
    pivot = pivot.reindex(index=countries, columns=indicators)

    # 若存在缺失（理论上你现在数据完整），这里仍做兜底填充为0，避免TOPSIS报错
    pivot = pivot.fillna(0.0)
    return pivot


def main():
    base_dir = Path(__file__).resolve().parent
    outputs_dir = ensure_outputs_dir(base_dir)

    # ---------------------------
    # Step 0: 指标清单
    # ---------------------------
    specs = default_indicator_specs()
    indicators = [s.name for s in specs]
    countries = COUNTRIES_CN

    print("=== Task 3: AI Competitiveness Ranking Forecast (2026-2035) ===")
    print(f"[Info] Base dir: {base_dir}")
    print(
        f"[Info] Indicators: {len(indicators)} | Countries: {len(countries)}")

    # ---------------------------
    # Step 1: 读取 2016-2025 原始面板数据
    # ---------------------------
    panel_long, indicator_files = load_all_data(
        base_dir, specs=specs, countries=countries)

    # 保存原始长表（未来可视化直接用）
    panel_long.to_csv(outputs_dir / "panel_raw_2016_2025_long.csv",
                      index=False, encoding="utf-8-sig")

    # 检查年份范围
    years_hist = sorted(panel_long["year"].unique().tolist())
    if years_hist != list(range(2016, 2026)):
        print(f"[Warn] 历史年份不是严格 2016-2025：{years_hist}")

    print("[Step 1] Data loaded.")
    print(f"         Total rows: {len(panel_long)}")
    print("         Indicator files:")
    for k, v in indicator_files.items():
        print(f"           - {k}: {v.relative_to(base_dir)}")

    # ---------------------------
    # Step 2: 逐国家×指标预测 2026-2035
    # ---------------------------
    future_years = list(range(2026, 2036))

    pred_records = []        # 用于 predicted_indicators_2026_2035.csv
    diag_records = []        # 用于 forecast_diagnostics.csv

    # 为后续做“2016-2035全期面板”保留预测拼接
    panel_pred_long = panel_long.copy()

    # 逐序列预测
    for ind in indicators:
        df_ind = panel_long[panel_long["indicator"] == ind]
        for c in countries:
            df_seq = df_ind[df_ind["country"] == c].sort_values("year")
            years = df_seq["year"].astype(int).tolist()
            values = df_seq["value"].astype(float).tolist()

            # 预测并诊断
            res = forecast_one_series(
                indicator_name=ind,
                years=years,
                values=values,
                future_years=future_years,
                mape_threshold=0.20,   # 阈值可在此统一调整（敏感性分析可后做）
            )

            # 写预测记录（长表）
            for y, v in zip(res.pred_years, res.pred_values):
                pred_records.append({
                    "year": int(y),
                    "country": c,
                    "indicator": ind,
                    "value_pred": float(v),
                    "model_used": res.model_used,
                })

            # 写诊断记录（每序列一行）
            lo, hi = res.bounds if res.bounds is not None else (None, None)
            diag_records.append({
                "country": c,
                "indicator": ind,
                "model_used": res.model_used,
                "mape_backtest_2025": res.mape_2025,
                "gm_a": res.gm_a,
                "gm_b": res.gm_b,
                "bounds_lo": lo,
                "bounds_hi": hi if (hi is not None and np.isfinite(hi)) else None,
            })

            # 将预测拼接到全期面板（未来可视化/分析直接用）
            for y, v in zip(res.pred_years, res.pred_values):
                panel_pred_long = pd.concat([
                    panel_pred_long,
                    pd.DataFrame([{
                        "year": int(y),
                        "country": c,
                        "indicator": ind,
                        "value": float(v),
                    }])
                ], ignore_index=True)

    # 输出预测结果与诊断
    pred_df = pd.DataFrame(pred_records)
    pred_df.sort_values(["indicator", "country", "year"], inplace=True)
    pred_df.to_csv(outputs_dir / "predicted_indicators_2026_2035.csv",
                   index=False, encoding="utf-8-sig")

    diag_df = pd.DataFrame(diag_records)
    diag_df.sort_values(["indicator", "country"], inplace=True)
    diag_df.to_csv(outputs_dir / "forecast_diagnostics.csv",
                   index=False, encoding="utf-8-sig")

    # 输出“2016-2035全期面板长表”（未来做可视化非常方便）
    panel_pred_long.sort_values(["indicator", "country", "year"], inplace=True)
    panel_pred_long.to_csv(
        outputs_dir / "panel_full_2016_2035_long.csv", index=False, encoding="utf-8-sig")

    print("[Step 2] Forecast completed.")
    print(f"         Pred rows: {len(pred_df)}")
    print(f"         Diagnostics rows: {len(diag_df)}")

    # ---------------------------
    # Step 3: 读取问题二权重（若无则等权），逐年TOPSIS得分与排名
    # ---------------------------
    weights_s = load_weights(outputs_dir=outputs_dir, indicators=indicators)
    # 保存本次实际使用的权重（便于复现核查）
    weights_s.rename("weight").to_csv(
        outputs_dir / "task3_weights_used.csv", encoding="utf-8-sig")

    weights = weights_s.values.astype(float)

    score_records = []
    rank_records = []

    for year in future_years:
        mat_df = build_panel_wide(
            panel_pred_long, year=year, indicators=indicators, countries=countries)
        scores = topsis_score(mat_df.values, weights)
        ranks = rank_desc(scores)

        # 记录得分与排名（长表）
        for idx, country in enumerate(countries):
            score_records.append({
                "year": year,
                "country": country,
                "score": float(scores[idx]),
            })
            rank_records.append({
                "year": year,
                "country": country,
                "rank": int(ranks[idx]),
            })

        # 同时保存该年的原始评估输入矩阵（便于未来可视化/解释）
        mat_df_out = mat_df.copy()
        mat_df_out.insert(0, "country", mat_df_out.index)
        mat_df_out.to_csv(
            outputs_dir / f"topsis_input_matrix_{year}.csv", index=False, encoding="utf-8-sig")

    scores_df = pd.DataFrame(score_records).sort_values(
        ["year", "score"], ascending=[True, False])
    ranks_df = pd.DataFrame(rank_records).sort_values(
        ["year", "rank"], ascending=[True, True])

    scores_df.to_csv(outputs_dir / "topsis_scores_2026_2035.csv",
                     index=False, encoding="utf-8-sig")
    ranks_df.to_csv(outputs_dir / "rankings_2026_2035.csv",
                    index=False, encoding="utf-8-sig")

    print("[Step 3] TOPSIS scoring & ranking completed.")
    print(f"         Scores rows: {len(scores_df)}")
    print(f"         Ranks rows: {len(ranks_df)}")

    # ---------------------------
    # Step 4: 额外中间结果（可选但推荐保留）
    # ---------------------------
    # 生成“每年排名宽表”（方便后续画演化图/河流图/热力图）
    rank_wide = ranks_df.pivot(
        index="country", columns="year", values="rank").reindex(countries)
    rank_wide.to_csv(
        outputs_dir / "rankings_2026_2035_wide.csv", encoding="utf-8-sig")

    score_wide = scores_df.pivot(
        index="country", columns="year", values="score").reindex(countries)
    score_wide.to_csv(
        outputs_dir / "scores_2026_2035_wide.csv", encoding="utf-8-sig")

    print("[Done] Outputs saved to outputs/ directory.")


if __name__ == "__main__":
    main()
