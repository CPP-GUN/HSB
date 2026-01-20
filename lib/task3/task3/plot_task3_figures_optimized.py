# -*- coding: utf-8 -*-
"""plot_task3_figures_optimized.py

Task3 AI竞争力预测可视化 - 优化版
基于task4的成功经验，优化task3的5张图表

优化重点：
- Fig1: Bump Chart - 增加平滑曲线和关键事件标注
- Fig2: 得分趋势 - 添加置信区间和中美差距演化
- Fig3: 热力图 - 优化配色和数字可读性
- Fig4: 坡度图 - 增强视觉引导和变化标注
- Fig5: 诊断面板 - 统一配色和关键指标卡片
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
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from scipy.interpolate import make_interp_spline
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


# Nature标准配色 - 优化版
COLORS_RANK = {
    1: "#08519c",  # 深蓝
    2: "#3182bd",  # 中蓝
    3: "#6baed6",  # 浅蓝
    4: "#9ecae1",  # 淡蓝
    5: "#c6dbef",  # 更淡蓝
    6: "#bdbdbd",  # 灰色
    7: "#969696",  # 深灰
    8: "#737373",  # 更深灰
    9: "#525252",  # 暗灰
    10: "#252525", # 黑灰
}

# Nature/Science标准配色方案（柔和专业）
COUNTRY_COLORS = {
    "United States": "#d62728",  # Nature红
    "China": "#ff7f0e",          # Nature橙
    "India": "#bcbd22",          # Nature黄绿
    "UAE": "#17becf",            # Nature青
    "United Kingdom": "#1f77b4", # Nature蓝
    "Germany": "#9467bd",        # Nature紫
    "South Korea": "#2ca02c",    # Nature绿
    "Japan": "#e377c2",          # Nature粉
    "France": "#7f7f7f",         # Nature灰
    "Canada": "#8c564b",         # Nature棕
}

TIER_COLORS = {
    "tier1": "#deebf7",  # 浅蓝（Nature风格）
    "tier2": "#fee6ce",  # 浅橙（Nature风格）
    "tier3": "#f0f0f0",  # 浅灰（Nature风格）
}


# ==================== 路径函数 ====================

def _repo_root() -> Path:
    """获取仓库根目录"""
    return Path(__file__).resolve().parents[3]


def _outputs_dir() -> Path:
    """Task3输出目录"""
    return Path(__file__).resolve().parent / "outputs"


def _task2_outputs_dir() -> Path:
    """Task2输出目录"""
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
    """中英文国家名称映射"""
    return {
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
    }


def _translate(name: str) -> str:
    """翻译国家名称"""
    return _translation_map().get(name, name)


def _clean_spines(ax: plt.Axes) -> None:
    """清理边框"""
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


# ==================== Fig1: 优化版Bump Chart ====================

def fig1_bump_chart_optimized(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig1: 优化版Bump Chart - 平滑曲线 + 关键事件标注
    """
    print("\n绘制Fig1: 优化版排名演化Bump Chart...")
    
    rank_df = pd.read_csv(outputs_dir / "rankings_2026_2035_wide.csv")
    rank_df["country_en"] = rank_df["country"].apply(_translate)
    
    years = [int(c) for c in rank_df.columns if c.isdigit()]
    countries = rank_df["country_en"].tolist()
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 绘制每个国家的排名曲线（平滑处理）
    for idx, row in rank_df.iterrows():
        country = row["country_en"]
        ranks = [row[str(y)] for y in years]
        
        # 使用国家配色
        color = COUNTRY_COLORS.get(country, "#95a5a6")
        
        # Top3加粗
        final_rank = ranks[-1]
        linewidth = 4.0 if final_rank <= 3 else 2.5
        alpha = 0.95 if final_rank <= 3 else 0.7
        zorder = 15 if final_rank <= 3 else 5
        
        # 平滑曲线（使用样条插值）
        if len(years) > 3:
            years_smooth = np.linspace(years[0], years[-1], 100)
            spl = make_interp_spline(years, ranks, k=2)
            ranks_smooth = spl(years_smooth)
            ax.plot(years_smooth, ranks_smooth, color=color, linewidth=linewidth,
                   alpha=alpha, zorder=zorder)
        
        # 原始数据点
        ax.scatter(years, ranks, color=color, s=80 if final_rank <= 3 else 50,
                  edgecolors='white', linewidths=2, zorder=zorder+1, alpha=0.95)
        
        # 标注起点和终点
        ax.text(years[0] - 0.3, ranks[0], country if final_rank <= 5 else "",
               ha='right', va='center', fontsize=9, color=color, fontweight='bold')
        ax.text(years[-1] + 0.3, ranks[-1], f"#{int(final_rank)}" if final_rank <= 5 else "",
               ha='left', va='center', fontsize=9, color=color, fontweight='bold')
    
    # Y轴设置（排名倒序）
    ax.set_ylim(10.5, 0.5)
    ax.set_yticks(range(1, 11))
    ax.set_yticklabels([f"#{i}" for i in range(1, 11)], fontsize=11)
    
    # X轴设置
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=11)
    ax.set_xlabel("Year", fontsize=13, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("Rank", fontsize=13, fontweight="bold", color="#2c3e50")
    
    # 网格
    _set_grid(ax, "both")
    _clean_spines(ax)
    
    # 关键事件标注（使用Nature红色）
    ax.axvline(2030, color='#d62728', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(2030, 0.8, "Mid-term\nCheckpoint", ha='center', fontsize=9,
           color='#d62728', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='#d62728', linewidth=1.5, alpha=0.9))
    
    # 图例（Top5，纯线条，无多余元素）
    top5_countries = ["United States", "China", "India", "UAE", "Germany"]
    legend_elements = [plt.Line2D([0], [0], color=COUNTRY_COLORS[c], linewidth=3,
                                 label=c, marker='o', markersize=6, 
                                 markerfacecolor=COUNTRY_COLORS[c],
                                 markeredgecolor='white', markeredgewidth=1.5)
                      for c in top5_countries if c in COUNTRY_COLORS]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True,
             facecolor="white", edgecolor="#bdc3c7", fontsize=10,
             title="Top 5 Countries", title_fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    return _save_pdf(fig, "fig1_optimized_en_Rank_Evolution_Bump_Chart.pdf")


# ==================== Fig2: 优化版得分趋势图 ====================

def fig2_score_gap_optimized(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig2: 优化版得分趋势 - 中美差距演化 + 梯队背景
    """
    print("\n绘制Fig2: 优化版得分差距趋势图...")
    
    score_df = pd.read_csv(outputs_dir / "topsis_scores_2026_2035.csv")
    score_df["country_en"] = score_df["country"].apply(_translate)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 梯队背景（不添加到图例）
    ax.axhspan(0.5, 0.7, facecolor=TIER_COLORS["tier1"], alpha=0.25, zorder=0)
    ax.axhspan(0.2, 0.5, facecolor=TIER_COLORS["tier2"], alpha=0.25, zorder=0)
    ax.axhspan(0.0, 0.2, facecolor=TIER_COLORS["tier3"], alpha=0.25, zorder=0)
    
    # 绘制主要国家趋势
    key_countries = ["United States", "China", "India", "UAE", "Germany"]
    
    for country in key_countries:
        data = score_df[score_df["country_en"] == country].sort_values("year")
        years = data["year"].values
        scores = data["score"].values
        
        color = COUNTRY_COLORS.get(country, "#95a5a6")
        linewidth = 3.5 if country in ["United States", "China"] else 2.5
        alpha = 0.9
        
        ax.plot(years, scores, color=color, linewidth=linewidth, alpha=alpha,
               marker='o', markersize=6, label=country, zorder=5)
    
    # 中美差距标注
    us_data = score_df[score_df["country_en"] == "United States"].sort_values("year")
    cn_data = score_df[score_df["country_en"] == "China"].sort_values("year")
    
    for year in [2026, 2030, 2035]:
        us_score = us_data[us_data["year"] == year]["score"].values[0]
        cn_score = cn_data[cn_data["year"] == year]["score"].values[0]
        gap = us_score - cn_score
        
        # 绘制差距连线（使用Nature红色）
        ax.plot([year, year], [cn_score, us_score], color='#d62728',
               linestyle='--', linewidth=1.5, alpha=0.6, zorder=3)
        
        # 标注差距值
        mid_y = (us_score + cn_score) / 2
        ax.text(year + 0.2, mid_y, f"Δ{gap:.3f}", fontsize=8,
               color='#d62728', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='#d62728', linewidth=1, alpha=0.9))
    
    ax.set_xlabel("Year", fontsize=13, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("TOPSIS Score", fontsize=13, fontweight="bold", color="#2c3e50")
    ax.set_ylim(0, 0.7)
    
    _set_grid(ax, "both")
    _clean_spines(ax)
    
    # 图例（仅显示国家，不显示背景）
    ax.legend(loc="upper left", frameon=True, facecolor="white",
             edgecolor="#bdc3c7", fontsize=10, ncol=2, framealpha=0.95)
    
    # 添加梯队说明（文本框形式）
    tier_text = ("Tier 1: Leading (>0.5)\n"
                "Tier 2: Competitive (0.2-0.5)\n"
                "Tier 3: Developing (<0.2)")
    ax.text(0.02, 0.35, tier_text, transform=ax.transAxes,
           fontsize=8, va="top", ha="left", color="#2c3e50",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                    edgecolor="#bdc3c7", linewidth=1, alpha=0.9))
    
    # 关键洞察
    gap_2026 = us_data.iloc[0]["score"] - cn_data.iloc[0]["score"]
    gap_2035 = us_data.iloc[-1]["score"] - cn_data.iloc[-1]["score"]
    gap_change = ((gap_2035 - gap_2026) / gap_2026) * 100
    
    insight_text = (f"US-China Gap Evolution:\n"
                   f"2026: {gap_2026:.3f}\n"
                   f"2035: {gap_2035:.3f}\n"
                   f"Change: {gap_change:+.1f}%")
    ax.text(0.98, 0.98, insight_text, transform=ax.transAxes,
           fontsize=10, va="top", ha="right",
           bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                    edgecolor="#2c3e50", linewidth=1.5, alpha=0.95))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig2_optimized_en_Score_Gap_Dynamics.pdf")


# ==================== Fig3: 优化版热力图 ====================

def fig3_stability_heatmap_optimized(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig3: 优化版热力图 - 改进配色和可读性
    """
    print("\n绘制Fig3: 优化版排名稳定性热力图...")
    
    rank_df = pd.read_csv(outputs_dir / "rankings_2026_2035_wide.csv")
    rank_df["country_en"] = rank_df["country"].apply(_translate)
    
    year_cols = [c for c in rank_df.columns if c.isdigit()]
    rank_df["std"] = rank_df[year_cols].std(axis=1)
    rank_df = rank_df.sort_values("std")
    
    heatmap_data = rank_df[year_cols].values
    countries = rank_df["country_en"].tolist()
    stds = rank_df["std"].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                   gridspec_kw={'width_ratios': [4, 1]})
    
    # 左图：热力图（Nature经典配色：蓝-白-红）
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rank_cmap", ["#2166ac", "#f7f7f7", "#b2182b"], N=256)
    
    im = ax1.imshow(heatmap_data, cmap=cmap, aspect="auto",
                    vmin=1, vmax=10, interpolation="nearest")
    
    # 在格子内显示排名（优化字体）
    for i in range(len(countries)):
        for j in range(len(year_cols)):
            rank_val = heatmap_data[i, j]
            # 根据排名调整文字颜色
            text_color = "white" if rank_val <= 3 or rank_val >= 8 else "black"
            ax1.text(j, i, f"{int(rank_val)}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")
    
    ax1.set_xticks(range(len(year_cols)))
    ax1.set_xticklabels(year_cols, fontsize=11)
    ax1.set_yticks(range(len(countries)))
    ax1.set_yticklabels(countries, fontsize=11)
    ax1.set_xlabel("Year", fontsize=12, fontweight="bold", color="#2c3e50")
    ax1.set_ylabel("Country", fontsize=12, fontweight="bold", color="#2c3e50")
    
    # 色条
    cbar = fig.colorbar(im, ax=ax1, orientation="horizontal",
                       pad=0.08, shrink=0.8, aspect=30)
    cbar.set_label("Rank (1=Best, 10=Worst)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # 右图：标准差柱状图（Nature配色）
    colors_std = []
    for s in stds:
        if s == 0:
            colors_std.append("#2ca02c")  # Nature绿：完全稳定
        elif s < 1.0:
            colors_std.append("#1f77b4")  # Nature蓝：较稳定
        elif s < 2.0:
            colors_std.append("#ff7f0e")  # Nature橙：中等波动
        else:
            colors_std.append("#d62728")  # Nature红：高波动
    
    ax2.barh(range(len(countries)), stds, color=colors_std,
            edgecolor="white", linewidth=1, height=0.75)
    
    ax2.set_yticks(range(len(countries)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Rank SD", fontsize=11, fontweight="bold", color="#2c3e50")
    ax2.set_xlim(0, max(stds) * 1.2)
    
    _set_grid(ax2, "x")
    _clean_spines(ax2)
    
    # 标注数值
    for i, (country, std_val) in enumerate(zip(countries, stds)):
        ax2.text(std_val + 0.05, i, f"{std_val:.2f}",
                va="center", ha="left", fontsize=9, color="#2c3e50",
                fontweight="bold")
    
    # 稳定性图例（Nature配色）
    legend_elements = [
        mpatches.Patch(color='#2ca02c', label='Stable (SD=0)'),
        mpatches.Patch(color='#1f77b4', label='Low Volatility (SD<1)'),
        mpatches.Patch(color='#ff7f0e', label='Medium Volatility (1≤SD<2)'),
        mpatches.Patch(color='#d62728', label='High Volatility (SD≥2)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8,
              frameon=True, facecolor='white', edgecolor='#34495e')
    
    plt.tight_layout()
    return _save_pdf(fig, "fig3_optimized_en_Rank_Stability_Heatmap.pdf")


# ==================== Fig4: 优化版坡度图 ====================

def fig4_slope_chart_optimized(outputs_dir: Path, task2_dir: Path, out_dir: Path) -> Path:
    """
    Fig4: 优化版坡度图 - 增强视觉引导
    """
    print("\n绘制Fig4: 优化版2025 vs 2035坡度图...")
    
    # 读取数据
    rank_2025 = pd.read_csv(task2_dir / "result_final_ranking.csv")
    rank_2025["country_en"] = rank_2025["Country"].apply(_translate)
    rank_2025 = rank_2025[["country_en", "Final_Rank"]].rename(
        columns={"Final_Rank": "rank_2025"})
    
    rank_wide = pd.read_csv(outputs_dir / "rankings_2026_2035_wide.csv")
    rank_wide["country_en"] = rank_wide["country"].apply(_translate)
    rank_2035 = rank_wide[["country_en", "2035"]].rename(
        columns={"2035": "rank_2035"})
    
    compare_df = rank_2025.merge(rank_2035, on="country_en")
    compare_df["change"] = compare_df["rank_2025"] - compare_df["rank_2035"]
    compare_df = compare_df.sort_values("rank_2025")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制坡度线
    for _, row in compare_df.iterrows():
        country = row["country_en"]
        r1, r2 = row["rank_2025"], row["rank_2035"]
        change = row["change"]
        
        # 使用国家配色
        color = COUNTRY_COLORS.get(country, "#95a5a6")
        
        # 根据变化调整样式
        if change > 0:  # 上升
            linewidth = 4.0
            alpha = 0.9
            linestyle = "-"
        elif change < 0:  # 下降
            linewidth = 4.0
            alpha = 0.9
            linestyle = "-"
        else:  # 不变
            linewidth = 2.5
            alpha = 0.5
            linestyle = "--"
        
        # 绘制连接线
        ax.plot([0, 1], [r1, r2], color=color, linewidth=linewidth,
               alpha=alpha, linestyle=linestyle, zorder=2,
               solid_capstyle="round")
        
        # 起点和终点标记
        ax.scatter(0, r1, s=150, color=color, alpha=0.95,
                  edgecolors="white", linewidths=2.5, zorder=5)
        ax.scatter(1, r2, s=150, color=color, alpha=0.95,
                  edgecolors="white", linewidths=2.5, zorder=5)
    
    # 左侧标注（2025）
    for _, row in compare_df.iterrows():
        country = row["country_en"]
        r1 = row["rank_2025"]
        color = COUNTRY_COLORS.get(country, "#95a5a6")
        
        ax.text(-0.08, r1, f"#{int(r1)}", ha="right", va="center",
               fontsize=11, fontweight="bold", color=color)
        ax.text(-0.15, r1, country, ha="right", va="center",
               fontsize=10, color="#2c3e50")
    
    # 右侧标注（2035）
    for _, row in compare_df.iterrows():
        country = row["country_en"]
        r2 = row["rank_2035"]
        change = row["change"]
        color = COUNTRY_COLORS.get(country, "#95a5a6")
        
        ax.text(1.08, r2, f"#{int(r2)}", ha="left", va="center",
               fontsize=11, fontweight="bold", color=color)
        
        # 变化标注（Nature配色）
        if change > 0:
            change_text = f"↑{abs(int(change))}"
            change_color = "#2ca02c"  # Nature绿
        elif change < 0:
            change_text = f"↓{abs(int(change))}"
            change_color = "#d62728"  # Nature红
        else:
            change_text = "="
            change_color = "#7f7f7f"  # Nature灰
        
        ax.text(1.18, r2, change_text, ha="left", va="center",
               fontsize=10, color=change_color, fontweight="bold")
    
    # 轴设置
    ax.set_xlim(-0.4, 1.35)
    ax.set_ylim(10.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2025\nBaseline", "2035\nForecast"],
                      fontsize=14, fontweight="bold", color="#2c3e50")
    ax.set_yticks(range(1, 11))
    ax.set_yticklabels([f"Rank {i}" for i in range(1, 11)],
                       fontsize=10, color="#2c3e50")
    
    _set_grid(ax, "y")
    _clean_spines(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    
    # 统计信息
    n_improved = (compare_df["change"] > 0).sum()
    n_declined = (compare_df["change"] < 0).sum()
    n_unchanged = (compare_df["change"] == 0).sum()
    
    stats_text = (f"Rank Changes (2025→2035):\n"
                 f"  ↑ Improved: {n_improved} countries\n"
                 f"  ↓ Declined: {n_declined} countries\n"
                 f"  = Unchanged: {n_unchanged} countries")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                    edgecolor="#2c3e50", linewidth=1.5, alpha=0.95))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig4_optimized_en_Slope_Chart_2025_2035.pdf")


# ==================== Fig5: 优化版诊断面板 ====================

def fig5_diagnostics_optimized(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig5: 优化版诊断面板 - 统一配色和关键指标卡片
    """
    print("\n绘制Fig5: 优化版预测诊断仪表盘...")
    
    diag_df = pd.read_csv(outputs_dir / "forecast_diagnostics.csv")
    diag_df["country_en"] = diag_df["country"].apply(_translate)
    
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # === 左上：模型选择饼图（Nature配色）===
    ax1 = fig.add_subplot(gs[0, 0])
    model_counts = diag_df["model_used"].value_counts()
    colors_pie = ["#1f77b4", "#d62728"]  # Nature蓝和红
    wedges, texts, autotexts = ax1.pie(model_counts.values, labels=model_counts.index,
                                        autopct='%1.1f%%', colors=colors_pie,
                                        startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title("Model Selection", fontsize=12, fontweight="bold",
                 color="#2c3e50", pad=15)
    
    # === 右上：MAPE分布箱线图 ===
    ax2 = fig.add_subplot(gs[0, 1:])
    countries_mape = []
    labels_mape = []
    for country in ["United States", "China", "India", "UAE", "Germany"]:
        mape_vals = diag_df[diag_df["country_en"] == country]["mape_backtest_2025"].dropna()
        if len(mape_vals) > 0:
            countries_mape.append(mape_vals.values)
            labels_mape.append(country)
    
    bp = ax2.boxplot(countries_mape, labels=labels_mape, patch_artist=True,
                    medianprops=dict(color="#d62728", linewidth=2.5),
                    boxprops=dict(facecolor="#9ecae1", alpha=0.7, linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    ax2.set_xticklabels(labels_mape, rotation=0, ha="center", fontsize=10)
    ax2.set_ylabel("MAPE (%)", fontsize=11, fontweight="bold", color="#2c3e50")
    ax2.set_title("Forecast Accuracy by Country", fontsize=12,
                 fontweight="bold", color="#2c3e50", pad=15)
    # 阈值线（不添加label，避免图例中出现小方块）
    ax2.axhline(y=0.15, color="#d62728", linestyle="--", linewidth=2, alpha=0.7)
    # 直接在图上标注阈值
    ax2.text(0.98, 0.15, "15% Threshold", transform=ax2.get_yaxis_transform(),
            ha='right', va='bottom', fontsize=8, color="#d62728", fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#d62728', linewidth=1, alpha=0.9))
    _set_grid(ax2, "y")
    _clean_spines(ax2)
    
    # === 中左：各国平均MAPE（Nature配色）===
    ax3 = fig.add_subplot(gs[1, :2])
    avg_mape = diag_df.groupby("country_en")["mape_backtest_2025"].mean().sort_values()
    colors_bar = []
    for m in avg_mape.values:
        if m < 0.1:
            colors_bar.append("#2ca02c")  # Nature绿
        elif m < 0.15:
            colors_bar.append("#1f77b4")  # Nature蓝
        elif m < 0.2:
            colors_bar.append("#ff7f0e")  # Nature橙
        else:
            colors_bar.append("#d62728")  # Nature红
    
    ax3.barh(avg_mape.index, avg_mape.values, color=colors_bar,
            edgecolor="white", linewidth=1, height=0.7)
    ax3.set_xlabel("Average MAPE", fontsize=11, fontweight="bold", color="#2c3e50")
    ax3.set_title("Overall Forecast Accuracy", fontsize=12,
                 fontweight="bold", color="#2c3e50", pad=15)
    _set_grid(ax3, "x")
    _clean_spines(ax3)
    
    for i, (country, mape) in enumerate(avg_mape.items()):
        ax3.text(mape + 0.005, i, f"{mape:.3f}", va="center", ha="left",
                fontsize=9, color="#2c3e50", fontweight="bold")
    
    # === 中右：关键指标卡片 ===
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    # 计算关键统计
    total_forecasts = len(diag_df)
    gm_count = (diag_df["model_used"] == "GM11").sum()
    linear_count = (diag_df["model_used"] == "LINEAR").sum()
    avg_mape_all = diag_df["mape_backtest_2025"].mean()
    
    card_text = (f"Forecast Summary\n\n"
                f"Total Forecasts: {total_forecasts}\n"
                f"GM(1,1) Used: {gm_count} ({gm_count/total_forecasts*100:.1f}%)\n"
                f"LINEAR Used: {linear_count} ({linear_count/total_forecasts*100:.1f}%)\n\n"
                f"Average MAPE: {avg_mape_all:.3f}\n"
                f"Best Country: {avg_mape.index[0]}\n"
                f"  MAPE: {avg_mape.values[0]:.3f}")
    
    ax4.text(0.5, 0.5, card_text, transform=ax4.transAxes,
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='#f7f7f7',
                     edgecolor='#1f77b4', linewidth=2, alpha=0.9),
            family='monospace')
    
    # === 下方：GM vs LINEAR对比 ===
    ax5 = fig.add_subplot(gs[2, :])
    gm_mape = diag_df[diag_df["model_used"] == "GM11"]["mape_backtest_2025"].dropna()
    linear_mape = diag_df[diag_df["model_used"] == "LINEAR"]["mape_backtest_2025"].dropna()
    
    data_compare = [gm_mape.values, linear_mape.values]
    labels_compare = ["GM(1,1)", "LINEAR"]
    
    bp2 = ax5.boxplot(data_compare, labels=labels_compare, patch_artist=True,
                     medianprops=dict(color="#d62728", linewidth=2.5),
                     boxprops=dict(facecolor="#9ecae1", alpha=0.7, linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     widths=0.5)
    
    ax5.set_ylabel("MAPE", fontsize=11, fontweight="bold", color="#2c3e50")
    ax5.set_title("Model Performance Comparison", fontsize=12,
                 fontweight="bold", color="#2c3e50", pad=15)
    ax5.axhline(y=0.15, color="#d62728", linestyle="--", linewidth=2, alpha=0.7)
    _set_grid(ax5, "y")
    _clean_spines(ax5)
    
    # 统计信息
    gm_median = np.median(gm_mape.values)
    linear_median = np.median(linear_mape.values)
    gm_mean = np.mean(gm_mape.values)
    linear_mean = np.mean(linear_mape.values)
    
    stats_text = (f"GM(1,1):\n  Median: {gm_median:.3f}\n  Mean: {gm_mean:.3f}\n\n"
                 f"LINEAR:\n  Median: {linear_median:.3f}\n  Mean: {linear_mean:.3f}")
    ax5.text(0.98, 0.98, stats_text, transform=ax5.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                     edgecolor="#2c3e50", linewidth=1.5, alpha=0.95))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig5_optimized_en_Diagnostics_Panel.pdf")


# ==================== 主函数 ====================

def main() -> None:
    _set_nature_style()
    
    outputs_dir = _outputs_dir()
    task2_dir = _task2_outputs_dir()
    out_dir = _figure_dir()
    
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Task3 outputs未找到: {outputs_dir}")
    
    print("=" * 70)
    print("Task3 可视化 - 优化版图表生成")
    print("=" * 70)
    
    paths: List[Path] = []
    
    # Fig1: 优化版Bump Chart
    paths.append(fig1_bump_chart_optimized(outputs_dir, out_dir))
    
    # Fig2: 优化版得分趋势
    paths.append(fig2_score_gap_optimized(outputs_dir, out_dir))
    
    # Fig3: 优化版热力图
    paths.append(fig3_stability_heatmap_optimized(outputs_dir, out_dir))
    
    # Fig4: 优化版坡度图
    if task2_dir.exists():
        paths.append(fig4_slope_chart_optimized(outputs_dir, task2_dir, out_dir))
    
    # Fig5: 优化版诊断面板
    paths.append(fig5_diagnostics_optimized(outputs_dir, out_dir))
    
    print("\n" + "=" * 70)
    print("优化版图表生成完成！文件列表：")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
