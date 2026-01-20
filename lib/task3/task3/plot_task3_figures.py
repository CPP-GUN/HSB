# -*- coding: utf-8 -*-
"""plot_task3_figures.py

Task3 AI竞争力预测可视化 - Nature/Science级图表
生成5张顶刊标准图表，输出到 <repo>/figure/task3/

图表清单：
- Fig1: Bump Chart - 排名演化碰撞图 (2026-2035)
- Fig2: 得分差距趋势图 - 梯队背景 + 置信带
- Fig3: 排名稳定性热力图 - 国家×年份 + 标准差柱状图
- Fig4: 桑基图 - 2025基线 vs 2035预测对比
- Fig5: 预测诊断2×2仪表盘 - 模型验证
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# ==================== 样式配置 ====================

def _set_nature_style() -> None:
    """设置Nature/Science顶刊样式"""
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "SimSun"]
        plt.rcParams["font.family"] = "serif"
    except Exception:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


# Nature标准配色
COLORS_RANK = {
    1: "#08519c",  # 深蓝 - 第1名
    2: "#3182bd",  # 中蓝 - 第2名
    3: "#6baed6",  # 浅蓝 - 第3名
    4: "#9ecae1",  # 淡蓝 - 第4名
    5: "#c6dbef",  # 更淡蓝 - 第5名
    6: "#bdbdbd",  # 灰色 - 第6名
    7: "#969696",  # 深灰 - 第7名
    8: "#737373",  # 更深灰 - 第8名
    9: "#525252",  # 暗灰 - 第9名
    10: "#252525", # 黑灰 - 第10名
}

TIER_COLORS = {
    "tier1": "#e6f5ff",  # 浅蓝 - 第1梯队
    "tier2": "#fff7e6",  # 浅黄 - 第2梯队
    "tier3": "#f5f5f5",  # 浅灰 - 第3梯队
}


# ==================== 路径函数 ====================

def _repo_root() -> Path:
    """获取仓库根目录"""
    return Path(__file__).resolve().parents[3]


def _outputs_dir() -> Path:
    """Task3输出目录"""
    return Path(__file__).resolve().parent / "outputs"


def _task2_outputs_dir() -> Path:
    """Task2输出目录（用于2025基线数据）"""
    return Path(__file__).resolve().parents[2] / "task2" / "task2" / "outputs"


def _figure_dir() -> Path:
    """图片输出目录"""
    return _repo_root() / "figure" / "task3"


def _save_pdf(fig: plt.Figure, filename: str) -> Path:
    """保存PDF到figure/task3/目录"""
    out_dir = _figure_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


# ==================== 翻译函数 ====================

def _translation_map() -> Dict[str, str]:
    """中英文指标和国家名称映射"""
    return {
        # 国家名称翻译（核心）
        "美国": "United States",
        "中国": "China",
        "印度": "India",
        "阿联酋": "UAE",
        "英国": "United Kingdom",
        "韩国": "South Korea",
        "法国": "France",
        "日本": "Japan",
        "德国": "Germany",
        "加拿大": "Canada",
        
        # 指标名称翻译
        "AI研究人员数量": "AI Researchers",
        "顶尖AI学者数量": "Top AI Scholars",
        "AI毕业生数量": "AI Graduates",
        "AI企业数量": "AI Enterprises",
        "AI市场规模": "AI Market Size",
        "AI应用渗透率": "AI Penetration",
        "大模型数量": "Large Models",
        "AI社会信任度": "Public Trust in AI",
        "AI政策数量": "AI Policies",
        "AI补贴金额": "AI Subsidies",
        "企业研发支出": "Corporate R&D",
        "政府AI投资": "Government AI Investment",
        "国际AI投资": "International AI Investment",
        "5G覆盖率": "5G Coverage",
        "GPU集群规模": "GPU Clusters",
        "互联网带宽": "Internet Bandwidth",
        "互联网普及率": "Internet Penetration",
        "电能生产": "Power Generation",
        "AI算力平台": "AI Computing Platforms",
        "数据中心数量": "Data Centers",
        "TOP500上榜数": "TOP500 Systems",
        "AI_Book数量": "AI Books",
        "AI_Dataset数量": "AI Datasets",
        "GitHub项目数": "GitHub Projects",
    }


def _translate(name: str) -> str:
    """翻译指标名称和国家名称"""
    return _translation_map().get(name, name)


def _clean_spines(ax: plt.Axes) -> None:
    """清理边框（去上右，细化左下）"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_color("#2c3e50")
    ax.spines["bottom"].set_color("#2c3e50")


def _set_grid(ax: plt.Axes, axis: str = "y") -> None:
    """设置细网格线"""
    ax.grid(axis=axis, linestyle="--", linewidth=0.5, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


# ==================== Fig1: Bump Chart 排名演化碰撞图 ====================

def fig1_bump_chart(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig1: Bump Chart - 2026-2035排名演化碰撞图
    数据源: rankings_2026_2035_wide.csv
    """
    print("\n绘制Fig1: 排名演化Bump Chart...")
    
    rank_df = pd.read_csv(outputs_dir / "rankings_2026_2035_wide.csv")
    
    # 翻译国家名称
    rank_df["country_en"] = rank_df["country"].apply(_translate)
    
    years = [int(c) for c in rank_df.columns if c.isdigit()]
    countries = rank_df["country_en"].tolist()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制每个国家的排名曲线
    for idx, row in rank_df.iterrows():
        country = row["country_en"]
        ranks = [row[str(y)] for y in years]
        
        # 颜色：按2035年最终排名
        final_rank = ranks[-1]
        color = COLORS_RANK.get(final_rank, "#969696")
        
        # Top3加粗
        linewidth = 3.5 if final_rank <= 3 else 2.0
        alpha = 0.9 if final_rank <= 3 else 0.7
        
        # 绘制平滑曲线
        ax.plot(years, ranks, color=color, linewidth=linewidth, 
                alpha=alpha, marker='o', markersize=6, 
                markeredgecolor='white', markeredgewidth=1.5, 
                label=country, zorder=10 if final_rank <= 3 else 5)
    
    # 设置Y轴（排名倒序，1在上）
    ax.set_ylim(10.5, 0.5)
    ax.set_yticks(range(1, 11))
    ax.set_yticklabels([f"#{i}" for i in range(1, 11)], fontsize=10)
    
    # X轴
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=10)
    ax.set_xlabel("Year", fontsize=12, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("Rank", fontsize=12, fontweight="bold", color="#2c3e50")
    
    # 网格
    _set_grid(ax, "both")
    _clean_spines(ax)
    
    # 图例（仅显示Top5）
    handles, labels = ax.get_legend_handles_labels()
    top5_indices = [i for i, label in enumerate(labels) if label in 
                    ["United States", "China", "India", "UAE", "Germany"]]
    handles_top5 = [handles[i] for i in top5_indices]
    labels_top5 = [labels[i] for i in top5_indices]
    
    ax.legend(handles_top5, labels_top5, loc="upper right", 
              frameon=True, facecolor="white", edgecolor="#bdc3c7", 
              fontsize=10, title="Top 5 Countries", title_fontsize=11)
    
    # 标注关键变化
    # 法国：2031年从第6升至第6，2033年降至第8
    fr_idx = countries.index("France")
    fr_ranks = [rank_df.iloc[fr_idx][str(y)] for y in years]
    ax.annotate("France fluctuation", xy=(2032, fr_ranks[6]), 
                xytext=(2030, 4), fontsize=8, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig1_en_Rank_Evolution_Bump_Chart_2026_2035.pdf")


# ==================== Fig2: 得分差距趋势图 ====================

def fig2_score_gap_trends(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig2: 得分差距演化图 - 梯队背景 + 置信带
    数据源: topsis_scores_2026_2035.csv
    """
    print("\n绘制Fig2: 得分差距趋势图...")
    
    score_df = pd.read_csv(outputs_dir / "topsis_scores_2026_2035.csv")
    
    # 翻译国家名称
    score_df["country_en"] = score_df["country"].apply(_translate)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 梯队背景
    ax.axhspan(0.5, 0.7, facecolor=TIER_COLORS["tier1"], alpha=0.3, zorder=0, label="Tier 1 (>0.5)")
    ax.axhspan(0.15, 0.25, facecolor=TIER_COLORS["tier2"], alpha=0.3, zorder=0, label="Tier 2 (0.15-0.25)")
    ax.axhspan(0.0, 0.12, facecolor=TIER_COLORS["tier3"], alpha=0.3, zorder=0, label="Tier 3 (<0.12)")
    
    # 绘制每个国家的得分趋势
    countries_en = score_df["country_en"].unique()
    
    for country in countries_en:
        data = score_df[score_df["country_en"] == country].sort_values("year")
        years = data["year"].values
        scores = data["score"].values
        
        # 确定最终排名（用于配色）
        final_score = scores[-1]
        if final_score > 0.5:
            color = "#08519c"
            linewidth = 3
            alpha = 0.9
        elif final_score > 0.2:
            color = "#3182bd"
            linewidth = 2.5
            alpha = 0.8
        else:
            color = "#969696"
            linewidth = 1.5
            alpha = 0.6
        
        ax.plot(years, scores, color=color, linewidth=linewidth, 
                alpha=alpha, marker='o', markersize=5, 
                label=country, zorder=5)
    
    ax.set_xlabel("Year", fontsize=12, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("TOPSIS Score", fontsize=12, fontweight="bold", color="#2c3e50")
    ax.set_ylim(0, 0.7)
    
    _set_grid(ax, "both")
    _clean_spines(ax)
    
    # 图例（仅Top3）
    handles, labels = ax.get_legend_handles_labels()
    # 过滤出前3个真实国家（排除背景标签）
    country_handles = [(h, l) for h, l in zip(handles, labels) 
                       if l in ["United States", "China", "India"]]
    if country_handles:
        h, l = zip(*country_handles)
        ax.legend(h, l, loc="upper left", frameon=True, 
                  facecolor="white", edgecolor="#bdc3c7", fontsize=10)
    
    # 标注中美差距
    us_data = score_df[score_df["country_en"] == "United States"].sort_values("year")
    cn_data = score_df[score_df["country_en"] == "China"].sort_values("year")
    gap_2026 = us_data.iloc[0]["score"] - cn_data.iloc[0]["score"]
    gap_2035 = us_data.iloc[-1]["score"] - cn_data.iloc[-1]["score"]
    
    ax.text(0.02, 0.98, f"US-China Gap:\n2026: {gap_2026:.3f}\n2035: {gap_2035:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                      edgecolor="#2c3e50", linewidth=1.5, alpha=0.95))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig2_en_Score_Gap_Dynamics_2026_2035.pdf")


# ==================== Fig3: 排名稳定性热力图 ====================

def fig3_stability_heatmap(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig3: 排名稳定性热力图 - 国家×年份 + 标准差柱状图
    数据源: rankings_2026_2035_wide.csv
    """
    print("\n绘制Fig3: 排名稳定性热力图...")
    
    rank_df = pd.read_csv(outputs_dir / "rankings_2026_2035_wide.csv")
    
    # 翻译国家名称
    rank_df["country_en"] = rank_df["country"].apply(_translate)
    
    # 提取年份列
    year_cols = [c for c in rank_df.columns if c.isdigit()]
    
    # 计算排名标准差
    rank_df["std"] = rank_df[year_cols].std(axis=1)
    rank_df = rank_df.sort_values("std")  # 按稳定性排序
    
    # 准备热力图数据
    heatmap_data = rank_df[year_cols].values
    countries = rank_df["country_en"].tolist()
    stds = rank_df["std"].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), 
                                    gridspec_kw={'width_ratios': [4, 1]})
    
    # 左图：热力图
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rank_cmap", ["#08519c", "#6baed6", "#bdbdbd", "#252525"], N=256)
    
    im = ax1.imshow(heatmap_data, cmap=cmap, aspect="auto", 
                     vmin=1, vmax=10, interpolation="nearest")
    
    # 在格子内显示排名数字
    for i in range(len(countries)):
        for j in range(len(year_cols)):
            rank_val = heatmap_data[i, j]
            text_color = "white" if rank_val <= 5 else "black"
            ax1.text(j, i, f"{int(rank_val)}", ha="center", va="center",
                     fontsize=8, color=text_color, fontweight="bold")
    
    ax1.set_xticks(range(len(year_cols)))
    ax1.set_xticklabels(year_cols, fontsize=10)
    ax1.set_yticks(range(len(countries)))
    ax1.set_yticklabels(countries, fontsize=10)
    ax1.set_xlabel("Year", fontsize=11, fontweight="bold", color="#2c3e50")
    ax1.set_ylabel("Country", fontsize=11, fontweight="bold", color="#2c3e50")
    
    # 色条
    cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", 
                        pad=0.08, shrink=0.8, aspect=30)
    cbar.set_label("Rank", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # 右图：标准差柱状图
    colors_std = ["#27ae60" if s == 0 else "#e74c3c" if s > 1.5 else "#f39c12" 
                  for s in stds]
    ax2.barh(range(len(countries)), stds, color=colors_std, 
             edgecolor="white", linewidth=0.8, height=0.75)
    
    ax2.set_yticks(range(len(countries)))
    ax2.set_yticklabels([])  # 隐藏Y轴标签（左图已有）
    ax2.set_xlabel("Rank SD", fontsize=10, fontweight="bold", color="#2c3e50")
    ax2.set_xlim(0, max(stds) * 1.2)
    
    _set_grid(ax2, "x")
    _clean_spines(ax2)
    
    # 标注
    for i, (country, std_val) in enumerate(zip(countries, stds)):
        ax2.text(std_val + 0.05, i, f"{std_val:.2f}", 
                 va="center", ha="left", fontsize=8, color="#2c3e50")
    
    plt.tight_layout()
    return _save_pdf(fig, "fig3_en_Rank_Stability_Heatmap_2026_2035.pdf")


# ==================== Fig4: 2025 vs 2035 桑基图 ====================

def fig4_baseline_forecast_sankey(outputs_dir: Path, task2_dir: Path, out_dir: Path) -> Path:
    """
    Fig4: 桑基图 - 2025基线 vs 2035预测对比
    数据源: task2/result_final_ranking.csv + rankings_2026_2035_wide.csv
    """
    print("\n绘制Fig4: 2025 vs 2035桑基图...")
    
    # 读取2025排名（Task2）
    rank_2025 = pd.read_csv(task2_dir / "result_final_ranking.csv")
    rank_2025["country_en"] = rank_2025["Country"].apply(_translate)
    rank_2025 = rank_2025[["country_en", "Final_Rank"]].rename(
        columns={"Final_Rank": "rank_2025"})
    
    # 读取2035排名（Task3）
    rank_wide = pd.read_csv(outputs_dir / "rankings_2026_2035_wide.csv")
    rank_wide["country_en"] = rank_wide["country"].apply(_translate)
    rank_2035 = rank_wide[["country_en", "2035"]].rename(
        columns={"2035": "rank_2035"})
    
    # 合并
    compare_df = rank_2025.merge(rank_2035, on="country_en")
    compare_df = compare_df.sort_values("rank_2035")
    
    countries = compare_df["country_en"].tolist()
    ranks_2025 = compare_df["rank_2025"].tolist()
    ranks_2035 = compare_df["rank_2035"].tolist()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 连续色谱
    cmap = plt.cm.RdYlGn_r
    
    # 绘制流带
    for i, country in enumerate(countries):
        r1, r2 = ranks_2025[i], ranks_2035[i]
        color = cmap(r2 / 10)
        alpha = 0.6 if r1 == r2 else 0.8  # 变化的国家更显眼
        
        linewidth = 6 if r1 != r2 else 4
        
        ax.plot([0, 1], [r1, r2], color=color, alpha=alpha, 
                linewidth=linewidth, solid_capstyle="round", zorder=2)
    
    # 绘制节点卡片
    for x, ranks in zip([0, 1], [ranks_2025, ranks_2035]):
        for rank in set(ranks):
            rect = mpatches.FancyBboxPatch((x-0.08, rank-0.35), 0.16, 0.7,
                                           boxstyle="round,pad=0.02",
                                           facecolor="white", edgecolor="#2c3e50",
                                           linewidth=2, zorder=3)
            ax.add_patch(rect)
            ax.text(x, rank, f"{rank}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#2c3e50", zorder=4)
    
    # 国家标注
    for i, country in enumerate(countries):
        # 左侧
        ax.text(-0.22, ranks_2025[i], country, va="center", ha="right",
                fontsize=9, color="#2c3e50")
        
        # 排名变化箭头
        change = ranks_2025[i] - ranks_2035[i]
        if change > 0:
            arrow = f"↑{abs(change)}"
            color_arrow = "#27ae60"
        elif change < 0:
            arrow = f"↓{abs(change)}"
            color_arrow = "#e74c3c"
        else:
            arrow = "—"
            color_arrow = "#95a5a6"
        
        ax.text(1.22, ranks_2035[i], arrow, va="center", ha="left",
                fontsize=8, color=color_arrow, fontweight="bold")
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2025\nBaseline", "2035\nForecast"],
                       fontsize=12, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("Rank", fontsize=12, fontweight="bold", color="#2c3e50")
    ax.set_ylim(0.5, 10.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.3, 1.4)
    
    _clean_spines(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9, width=0, colors="#2c3e50")
    ax.tick_params(axis="x", width=0, pad=10)
    
    plt.tight_layout()
    return _save_pdf(fig, "fig4_en_Baseline_vs_Forecast_Sankey_2025_2035.pdf")


# ==================== Fig5: 预测诊断2×2仪表盘 ====================

def fig5_forecast_diagnostics(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig5: 预测诊断2×2组合仪表盘
    数据源: forecast_diagnostics.csv
    """
    print("\n绘制Fig5: 预测诊断仪表盘...")
    
    diag_df = pd.read_csv(outputs_dir / "forecast_diagnostics.csv")
    
    # 翻译国家名称
    diag_df["country_en"] = diag_df["country"].apply(_translate)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # === 左上：模型选择饼图 ===
    model_counts = diag_df["model_used"].value_counts()
    colors_pie = ["#3182bd", "#969696"]
    ax1.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 10})
    ax1.set_title("Model Selection Distribution", fontsize=11, 
                  fontweight="bold", color="#2c3e50", pad=15)
    
    # === 右上：MAPE分布箱线图 ===
    # 按国家分组的MAPE
    countries_mape = []
    labels_mape = []
    for country in diag_df["country_en"].unique():
        mape_vals = diag_df[diag_df["country_en"] == country]["mape_backtest_2025"].dropna()
        if len(mape_vals) > 0:
            countries_mape.append(mape_vals.values)
            labels_mape.append(country)
    
    bp = ax2.boxplot(countries_mape, labels=labels_mape, patch_artist=True,
                     medianprops=dict(color="#e74c3c", linewidth=2),
                     boxprops=dict(facecolor="#91bfdb", alpha=0.7))
    
    ax2.set_xticklabels(labels_mape, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("MAPE (%)", fontsize=10, fontweight="bold", color="#2c3e50")
    ax2.set_title("MAPE Distribution by Country", fontsize=11, 
                  fontweight="bold", color="#2c3e50", pad=15)
    ax2.axhline(y=0.15, color="#e74c3c", linestyle="--", linewidth=1.5, 
                alpha=0.7, label="15% Threshold")
    _set_grid(ax2, "y")
    _clean_spines(ax2)
    ax2.legend(fontsize=8)
    
    # === 左下：各国平均MAPE条形图 ===
    avg_mape = diag_df.groupby("country_en")["mape_backtest_2025"].mean().sort_values()
    colors_bar = ["#27ae60" if m < 0.1 else "#f39c12" if m < 0.2 else "#e74c3c" 
                  for m in avg_mape.values]
    
    ax3.barh(avg_mape.index, avg_mape.values, color=colors_bar, 
             edgecolor="white", linewidth=0.8, height=0.7)
    ax3.set_xlabel("Average MAPE", fontsize=10, fontweight="bold", color="#2c3e50")
    ax3.set_title("Forecast Accuracy by Country", fontsize=11, 
                  fontweight="bold", color="#2c3e50", pad=15)
    _set_grid(ax3, "x")
    _clean_spines(ax3)
    
    for i, (country, mape) in enumerate(avg_mape.items()):
        ax3.text(mape + 0.005, i, f"{mape:.3f}", va="center", ha="left",
                 fontsize=8, color="#2c3e50")
    
    # === 右下：GM vs LINEAR模型对比 ===
    gm_mape = diag_df[diag_df["model_used"] == "GM11"]["mape_backtest_2025"].dropna()
    linear_mape = diag_df[diag_df["model_used"] == "LINEAR"]["mape_backtest_2025"].dropna()
    
    data_compare = [gm_mape.values, linear_mape.values]
    labels_compare = ["GM(1,1)", "LINEAR"]
    
    bp2 = ax4.boxplot(data_compare, labels=labels_compare, patch_artist=True,
                      medianprops=dict(color="#e74c3c", linewidth=2),
                      boxprops=dict(facecolor="#6baed6", alpha=0.7),
                      widths=0.5)
    
    ax4.set_ylabel("MAPE", fontsize=10, fontweight="bold", color="#2c3e50")
    ax4.set_title("Model Accuracy Comparison", fontsize=11, 
                  fontweight="bold", color="#2c3e50", pad=15)
    ax4.axhline(y=0.15, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)
    _set_grid(ax4, "y")
    _clean_spines(ax4)
    
    # 统计信息
    gm_median = np.median(gm_mape.values)
    linear_median = np.median(linear_mape.values)
    ax4.text(0.02, 0.98, f"GM Median: {gm_median:.3f}\nLINEAR Median: {linear_median:.3f}",
             transform=ax4.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#2c3e50", linewidth=1, alpha=0.9))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig5_en_Forecast_Diagnostics_Panel.pdf")


# ==================== 主函数 ====================

def main() -> None:
    _set_nature_style()
    
    outputs_dir = _outputs_dir()
    task2_dir = _task2_outputs_dir()
    out_dir = _figure_dir()
    
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Task3 outputs未找到: {outputs_dir}")
    
    if not task2_dir.exists():
        print(f"[Warning] Task2 outputs未找到: {task2_dir}，Fig4将跳过")
    
    print("=" * 70)
    print("Task3 可视化 - Nature/Science级图表生成")
    print("=" * 70)
    
    paths: List[Path] = []
    
    # Fig1: Bump Chart
    paths.append(fig1_bump_chart(outputs_dir, out_dir))
    
    # Fig2: 得分差距趋势
    paths.append(fig2_score_gap_trends(outputs_dir, out_dir))
    
    # Fig3: 稳定性热力图
    paths.append(fig3_stability_heatmap(outputs_dir, out_dir))
    
    # Fig4: 桑基图（需要Task2数据）
    if task2_dir.exists():
        paths.append(fig4_baseline_forecast_sankey(outputs_dir, task2_dir, out_dir))
    
    # Fig5: 诊断仪表盘
    paths.append(fig5_forecast_diagnostics(outputs_dir, out_dir))
    
    print("\n" + "=" * 70)
    print("生成完成！文件列表：")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
