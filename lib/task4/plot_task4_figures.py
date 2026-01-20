# -*- coding: utf-8 -*-
"""plot_task4_figures.py

Task4 中国AI投资优化可视化 - Nature/Science级图表
生成5张顶刊标准图表，输出到 <repo>/figure/task4/

图表清单：
- Fig1: 饼图 - 六大维度投资分布（唯一饼图）
- Fig2: 横向条形图 - Top10投资重点+排名徽章
- Fig3: 哑铃图 - 24指标投资前后对比
- Fig4: 树状图 - 24指标投资全景
- Fig5: 气泡图 - 投资效率四象限分析
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
import squarify  # 树状图库

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


# 维度颜色映射
DIMENSION_COLORS = {
    "I_基础设施": "#1f77b4",  # 深蓝
    "T_人才": "#2ca02c",      # 绿色
    "P_政策": "#ff7f0e",      # 橙色
    "R_研发": "#9467bd",      # 紫色
    "A_应用": "#e377c2",      # 粉色
    "O_产出": "#8c564b",      # 棕色
}


# ==================== 路径函数 ====================

def _repo_root() -> Path:
    """获取仓库根目录"""
    return Path(__file__).resolve().parents[2]


def _outputs_dir() -> Path:
    """Task4输出目录"""
    return Path(__file__).resolve().parent / "outputs"


def _figure_dir() -> Path:
    """图片输出目录"""
    return _repo_root() / "figure" / "task4"


def _save_pdf(fig: plt.Figure, filename: str) -> Path:
    """保存PDF到figure/task4/目录"""
    out_dir = _figure_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


# ==================== 翻译函数 ====================

def _translation_map() -> Dict[str, str]:
    """中英文指标和维度映射"""
    return {
        # 维度翻译
        "I_基础设施": "Infrastructure",
        "T_人才": "Talent",
        "P_政策": "Policy",
        "R_研发": "R&D",
        "A_应用": "Application",
        "O_产出": "Output",
        
        # 指标翻译
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
    """翻译指标名称和维度名称"""
    return _translation_map().get(name, name)


def _clean_spines(ax: plt.Axes) -> None:
    """清理边框（去上右，细化左下）"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_color("#2c3e50")
    ax.spines["bottom"].set_color("#2c3e50")


# ==================== Fig1: 饼图 - 六大维度投资分布 ====================

def fig1_dimension_pie_chart(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig1: 饼图 - 六大维度投资分布（唯一饼图）
    数据源: dimension_distribution.csv
    """
    print("\n绘制Fig1: 六大维度投资分布饼图...")
    
    dim_df = pd.read_csv(outputs_dir / "dimension_distribution.csv")
    
    # 翻译维度名称
    dim_df["dimension_en"] = dim_df["维度"].apply(_translate)
    
    # 按占比排序
    dim_df = dim_df.sort_values("占比_%", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 颜色映射
    colors = [DIMENSION_COLORS.get(d, "#95a5a6") for d in dim_df["维度"]]
    
    # 突出最大扇形（基础设施）
    explode = [0.08 if i == 0 else 0 for i in range(len(dim_df))]
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        dim_df["投资额_亿元"].values,
        labels=dim_df["dimension_en"].values,
        autopct=lambda pct: f'{pct:.1f}%\n({pct*100:.0f}B¥)',
        startangle=90,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 11, 'fontweight': 'bold', 'color': 'white'},
        wedgeprops=dict(edgecolor='white', linewidth=2.5)
    )
    
    # 优化百分比标签样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # 优化维度标签样式
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
        text.set_color('#2c3e50')
    
    # 中心标注总金额
    ax.text(0, 0, "Total\n10,000B¥", ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='circle,pad=0.4', facecolor='white', 
                     edgecolor='#3498db', linewidth=2.5, alpha=0.95))
    
    # 图例（带投资额详情）
    legend_labels = [f"{row['dimension_en']}: {row['投资额_亿元']:.0f}B¥ ({row['占比_%']:.1f}%)" 
                     for _, row in dim_df.iterrows()]
    ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1.05, 1),
              frameon=True, facecolor='white', edgecolor='#34495e',
              fontsize=10, title="Investment Breakdown", title_fontsize=11,
              shadow=True, framealpha=0.95)
    
    # 添加洞察文本框
    insight_text = ("Infrastructure dominates\nwith 44% investment,\n"
                   "reflecting hardware-first\nstrategy")
    ax.text(-1.8, -1.3, insight_text, fontsize=10, color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#e3f2fd',
                     edgecolor='#3498db', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig1_en_Dimension_Investment_Pie_Chart.pdf")


# ==================== Fig2: 横向条形图 - Top10投资重点 ====================

def fig2_top10_bar_chart(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig2: 横向条形图 - Top10投资重点+排名徽章
    数据源: investment_allocation.csv
    """
    print("\n绘制Fig2: Top10投资重点条形图...")
    
    inv_df = pd.read_csv(outputs_dir / "investment_allocation.csv")
    
    # 翻译指标名称
    inv_df["indicator_en"] = inv_df["指标"].apply(_translate)
    
    # Top 10
    top10 = inv_df.head(10).copy()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 反转顺序（排名1在上）
    top10 = top10.iloc[::-1]
    y_pos = np.arange(len(top10))
    
    # 渐变颜色（金→银→铜→蓝）
    colors = []
    for rank in top10["排名"].values:
        if rank == 1:
            colors.append("#FFD700")  # 金色
        elif rank == 2:
            colors.append("#C0C0C0")  # 银色
        elif rank == 3:
            colors.append("#CD7F32")  # 铜色
        else:
            colors.append("#3498db")  # 蓝色
    
    # 绘制条形图
    bars = ax.barh(y_pos, top10["投资额_亿元"].values, color=colors,
                   edgecolor='white', linewidth=1.5, height=0.7)
    
    # 渐变透明度
    for i, bar in enumerate(bars):
        bar.set_alpha(0.85)
    
    # 左侧排名徽章
    for i, (idx, row) in enumerate(top10.iterrows()):
        rank = row["排名"]
        if rank == 1:
            badge_color = "#FFD700"
            badge_text = "🥇"
        elif rank == 2:
            badge_color = "#C0C0C0"
            badge_text = "🥈"
        elif rank == 3:
            badge_color = "#CD7F32"
            badge_text = "🥉"
        else:
            badge_color = "#95a5a6"
            badge_text = f"#{rank}"
        
        ax.text(-80, i, badge_text, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white",
                bbox=dict(boxstyle="circle,pad=0.4", facecolor=badge_color,
                         edgecolor="white", linewidth=2))
    
    # 右侧数值标签
    for i, (idx, row) in enumerate(top10.iterrows()):
        inv = row["投资额_亿元"]
        pct = row["占比_%"]
        ax.text(inv + 30, i, f"{inv:.0f}B¥ ({pct:.1f}%)",
                va="center", ha="left", fontsize=10, color="#2c3e50", fontweight="bold")
    
    # Y轴标签（指标名称）
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10["indicator_en"].values, fontsize=11)
    
    ax.set_xlabel("Investment Amount (Billion RMB)", fontsize=12,
                  fontweight="bold", color="#2c3e50", labelpad=10)
    ax.set_xlim(0, max(top10["投资额_亿元"]) * 1.25)
    
    # 网格
    ax.grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    _clean_spines(ax)
    
    # 累计占比标注
    cumsum = inv_df.head(10)["占比_%"].sum()
    ax.text(0.98, 0.02, f"Top 10 accounts for {cumsum:.1f}% of total budget",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
            color="#2c3e50", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd",
                                       edgecolor="#f39c12", linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig2_en_Top10_Investment_Bar_Chart.pdf")


# ==================== Fig3: 哑铃图 - 投资前后对比 ====================

def fig3_dumbbell_comparison(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig3: 增长率条形图 - 24指标投资效果排序
    数据源: before_after_comparison.csv
    """
    print("\n绘制Fig3: 投资效果增长率条形图...")
    
    comp_df = pd.read_csv(outputs_dir / "before_after_comparison.csv")
    
    # 翻译指标名称
    comp_df["indicator_en"] = comp_df["指标"].apply(_translate)
    
    # 按增长率排序
    comp_df = comp_df.sort_values("增长率_%", ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 11))
    
    y_pos = np.arange(len(comp_df))
    growth_rates = comp_df["增长率_%"].values
    
    # 颜色映射（根据增长率）
    colors = []
    for rate in growth_rates:
        if rate > 1000:
            colors.append("#27ae60")  # 深绿
        elif rate > 500:
            colors.append("#2ecc71")  # 浅绿
        elif rate > 100:
            colors.append("#3498db")  # 蓝色
        else:
            colors.append("#95a5a6")  # 灰色
    
    # 绘制横向条形图
    bars = ax.barh(y_pos, growth_rates, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5, height=0.7)
    
    # 数值标签
    for i, (rate, bar) in enumerate(zip(growth_rates, bars)):
        ax.text(rate + max(growth_rates) * 0.02, i, f"{rate:.0f}%",
                va='center', ha='left', fontsize=9, fontweight='bold', color='#2c3e50')
    
    # Y轴标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp_df["indicator_en"].values, fontsize=9)
    
    ax.set_xlabel("Growth Rate (%)", fontsize=12, fontweight="bold",
                  color="#2c3e50", labelpad=10)
    
    # 网格
    ax.grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    _clean_spines(ax)
    
    # 图例
    legend_elements = [
        mpatches.Patch(color='#27ae60', label='Extreme Growth (>1000%)'),
        mpatches.Patch(color='#2ecc71', label='High Growth (500-1000%)'),
        mpatches.Patch(color='#3498db', label='Medium Growth (100-500%)'),
        mpatches.Patch(color='#95a5a6', label='Low Growth (<100%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, facecolor='white', edgecolor='#34495e', framealpha=0.95)
    
    # 标注Top3
    top3_indices = [len(comp_df) - 1, len(comp_df) - 2, len(comp_df) - 3]
    for rank, idx in enumerate(top3_indices, 1):
        badge_color = "#FFD700" if rank == 1 else "#C0C0C0" if rank == 2 else "#CD7F32"
        badge_text = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        ax.text(-max(growth_rates) * 0.08, idx, badge_text, ha="center", va="center",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.35", facecolor=badge_color,
                         edgecolor="white", linewidth=2))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig3_en_Growth_Rate_Bar_Chart.pdf")


# ==================== Fig4: 树状图 - 24指标投资全景 ====================

def fig4_treemap_all_indicators(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig4: 分组条形图 - 六大维度Top指标对比
    数据源: investment_allocation.csv + dimension_distribution.csv
    """
    print("\n绘制Fig4: 六大维度Top指标对比条形图...")
    
    inv_df = pd.read_csv(outputs_dir / "investment_allocation.csv")
    dim_df = pd.read_csv(outputs_dir / "dimension_distribution.csv")
    
    # 翻译
    inv_df["indicator_en"] = inv_df["指标"].apply(_translate)
    dim_df["dimension_en"] = dim_df["维度"].apply(_translate)
    
    # 维度映射
    dimension_map = {
        "GPU Clusters": "Infrastructure", "Internet Bandwidth": "Infrastructure",
        "TOP500 Systems": "Infrastructure", "Data Centers": "Infrastructure",
        "Power Generation": "Infrastructure", "AI Computing Platforms": "Infrastructure",
        "Internet Penetration": "Infrastructure", "5G Coverage": "Infrastructure",
        "AI Researchers": "Talent", "Top AI Scholars": "Talent", "AI Graduates": "Talent",
        "AI Policies": "Policy", "AI Subsidies": "Policy", "Public Trust in AI": "Policy",
        "Corporate R&D": "R&D", "Government AI Investment": "R&D", "International AI Investment": "R&D",
        "AI Market Size": "Application", "AI Enterprises": "Application",
        "AI Penetration": "Application", "Large Models": "Application",
        "GitHub Projects": "Output", "AI Books": "Output", "AI Datasets": "Output",
    }
    inv_df["dimension"] = inv_df["indicator_en"].map(dimension_map)
    
    # 每个维度取Top3指标
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    dimension_order = ["Infrastructure", "Talent", "Policy", "R&D", "Application", "Output"]
    
    for idx, dim in enumerate(dimension_order):
        ax = axes[idx]
        dim_data = inv_df[inv_df["dimension"] == dim].head(3)
        
        if len(dim_data) == 0:
            ax.axis('off')
            continue
        
        y_pos = np.arange(len(dim_data))
        investments = dim_data["投资额_亿元"].values
        
        # 颜色
        color_map = {
            "Infrastructure": "#1f77b4", "Talent": "#2ca02c",
            "Policy": "#ff7f0e", "R&D": "#9467bd",
            "Application": "#e377c2", "Output": "#8c564b"
        }
        color = color_map.get(dim, "#95a5a6")
        
        # 反转顺序（Top1在上）
        dim_data_rev = dim_data.iloc[::-1]
        
        bars = ax.barh(y_pos, dim_data_rev["投资额_亿元"].values, color=color,
                       alpha=0.85, edgecolor='white', linewidth=1.5, height=0.6)
        
        # 数值标签
        for i, (_, row) in enumerate(dim_data_rev.iterrows()):
            inv = row["投资额_亿元"]
            ax.text(inv + max(investments) * 0.05, i, f"{inv:.0f}B¥",
                   va='center', ha='left', fontsize=9, fontweight='bold', color='#2c3e50')
        
        # Y轴标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.replace(' ', '\n') for name in dim_data_rev["indicator_en"].values],
                          fontsize=8)
        
        # 子图标题
        dim_total = dim_df[dim_df["dimension_en"] == dim]["投资额_亿元"].values
        dim_pct = dim_df[dim_df["dimension_en"] == dim]["占比_%"].values
        if len(dim_total) > 0:
            ax.set_title(f"{dim}\nTotal: {dim_total[0]:.0f}B¥ ({dim_pct[0]:.1f}%)",
                        fontsize=11, fontweight='bold', color='#2c3e50', pad=10)
        
        ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        _clean_spines(ax)
    
    plt.tight_layout()
    return _save_pdf(fig, "fig4_en_Dimension_Top_Indicators_Chart.pdf")


# ==================== Fig5: 气泡图 - 投资效率分析 ====================

def fig5_bubble_efficiency(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig5: 热力图 - 24指标×3维度综合分析
    数据源: investment_allocation.csv + before_after_comparison.csv
    """
    print("\n绘制Fig5: 24指标综合分析热力图...")
    
    import seaborn as sns
    
    inv_df = pd.read_csv(outputs_dir / "investment_allocation.csv")
    comp_df = pd.read_csv(outputs_dir / "before_after_comparison.csv")
    
    # 合并数据
    merged = inv_df.merge(comp_df, on="指标", how="inner")
    merged["indicator_en"] = merged["指标"].apply(_translate)
    
    # 选择关键指标（Top12 + 高增长6个）
    top12 = merged.head(12)
    high_growth = merged.nlargest(6, "增长率_%")
    selected = pd.concat([top12, high_growth]).drop_duplicates(subset="指标")
    
    # 构建热力图数据（标准化到0-100）
    heatmap_data = pd.DataFrame({
        "Indicator": selected["indicator_en"].values,
        "Investment\n(Normalized)": (selected["投资额_亿元"] / selected["投资额_亿元"].max() * 100).values,
        "Growth Rate\n(Log Scale)": np.log10(selected["增长率_%"] + 1) / np.log10(selected["增长率_%"].max() + 1) * 100,
        "Growth Amount\n(Normalized)": (selected["增长量"] / selected["增长量"].max() * 100).values
    })
    
    heatmap_data = heatmap_data.set_index("Indicator")
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # 绘制热力图
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="RdYlGn",
                linewidths=2, linecolor='white', cbar_kws={'label': 'Normalized Score (0-100)'},
                ax=ax, vmin=0, vmax=100, square=False)
    
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    # 色带标签
    cbar = ax.collections[0].colorbar
    cbar.set_label('Normalized Score (0-100)', fontsize=11, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    return _save_pdf(fig, "fig5_en_Comprehensive_Heatmap.pdf")


# ==================== 主函数 ====================

def main() -> None:
    _set_nature_style()
    
    outputs_dir = _outputs_dir()
    out_dir = _figure_dir()
    
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Task4 outputs未找到: {outputs_dir}")
    
    print("=" * 70)
    print("Task4 可视化 - Nature/Science级图表生成")
    print("=" * 70)
    
    paths: List[Path] = []
    
    # Fig1: 饼图（六大维度投资分布）
    paths.append(fig1_dimension_pie_chart(outputs_dir, out_dir))
    
    # Fig2: 条形图（Top10投资重点）
    paths.append(fig2_top10_bar_chart(outputs_dir, out_dir))
    
    # Fig3: 哑铃图（投资前后对比）
    paths.append(fig3_dumbbell_comparison(outputs_dir, out_dir))
    
    # Fig4: 树状图（24指标全景）
    paths.append(fig4_treemap_all_indicators(outputs_dir, out_dir))
    
    # Fig5: 气泡图（投资效率分析）
    paths.append(fig5_bubble_efficiency(outputs_dir, out_dir))
    
    print("\n" + "=" * 70)
    print("生成完成！文件列表：")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
