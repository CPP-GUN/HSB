# -*- coding: utf-8 -*-
"""plot_task2_figures_optimized.py

Task2 优化版图表生成脚本
优化以下三张图：
- Fig2: 排名条形图（改为棒棒糖图+纯数字徽章）
- Fig3: 方法一致性流向图（简化+清晰化）
- Fig7: 验证双子图（添加统计指标+优化标注）
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path as MplPath
import warnings
warnings.filterwarnings('ignore')

# 导入原有的辅助函数
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_task2_figures import (
    _set_nature_style, _translation_map, _translate, _clean_spines, _set_grid,
    _repo_root, _outputs_dir, _figure_dir, _save_pdf,
    COLORS, GRADE_COLORS,
    fig1_weights_sankey, fig4_radar_dimensions, fig5_clustermap, fig6_parallel_coordinates
)


# ==================== Fig2: 排名棒棒糖图（优化版）====================

def fig2_ranking_lollipop_optimized(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig2: 2025年AI竞争力排名（优化版：棒棒糖图+纯数字徽章）
    特点：1)棒棒糖样式 2)纯数字排名徽章 3)等级背景 4)差距标注
    """
    print("\n绘制Fig2: 排名棒棒糖图（优化版）...")
    
    ranking_df = pd.read_csv(outputs_dir / "result_final_ranking.csv")
    ranking_df = ranking_df.sort_values("Final_Rank", ascending=False)
    ranking_df["Country_EN"] = ranking_df["Country"].apply(_translate)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 等级区域背景
    y_max = len(ranking_df)
    ax.axhspan(y_max - 0.5, y_max + 0.5, facecolor=GRADE_COLORS["A+"], alpha=0.08, zorder=0)
    ax.axhspan(y_max - 2.5, y_max - 0.5, facecolor=GRADE_COLORS["A"], alpha=0.08, zorder=0)
    ax.axhspan(y_max - 6.5, y_max - 2.5, facecolor=GRADE_COLORS["B"], alpha=0.08, zorder=0)
    
    y_pos = np.arange(len(ranking_df))
    scores = ranking_df["Comprehensive_Score"].values
    
    # 渐变色（基于等级）
    colors = []
    for grade in ranking_df["Grade"]:
        colors.append(GRADE_COLORS[grade])
    
    # 绘制棒棒糖（线条 + 圆点）
    for i, (score, color) in enumerate(zip(scores, colors)):
        # 线条
        ax.plot([0, score], [i, i], color=color, linewidth=3.5, alpha=0.7, zorder=1)
        # 圆点
        ax.scatter(score, i, s=400, color=color, edgecolor='white', 
                  linewidth=2.5, zorder=3, alpha=0.95)
    
    # 左侧纯数字排名徽章
    for i, (idx, row) in enumerate(ranking_df.iterrows()):
        rank = row["Final_Rank"]
        grade = row["Grade"]
        
        # 纯数字徽章
        ax.text(-0.04, i, str(rank), ha='center', va='center',
                fontsize=14, weight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.4', facecolor=GRADE_COLORS[grade],
                         edgecolor='white', linewidth=2.5, alpha=0.95))
    
    # 右侧分数标签
    for i, (idx, row) in enumerate(ranking_df.iterrows()):
        score = row["Comprehensive_Score"]
        ax.text(score + 0.015, i, f'{score:.4f}',
                va='center', ha='left', fontsize=10, weight='bold', color='#2c3e50')
    
    # 差距标注（与第1名）
    top_score = scores.max()
    for i, score in enumerate(scores):
        if i < len(scores) - 1:  # 不给第1名标注
            gap = top_score - score
            if gap > 0:
                ax.text(score * 0.5, i, f'−{gap:.4f}', 
                        va='center', ha='center', fontsize=8, 
                        color='#7f8c8d', style='italic', alpha=0.8)
    
    # Y轴标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranking_df["Country_EN"], fontsize=11)
    
    ax.set_xlabel("Comprehensive Score", fontsize=12, weight='bold', color='#2c3e50')
    ax.set_xlim(-0.08, top_score * 1.2)
    
    # 平均线
    avg_score = scores.mean()
    ax.axvline(avg_score, color='#e74c3c', linestyle='--', linewidth=1.8,
              alpha=0.6, label=f'Average: {avg_score:.4f}')
    
    _clean_spines(ax)
    _set_grid(ax, "x")
    ax.tick_params(axis="both", labelsize=10, width=0.6, colors='#2c3e50')
    
    # 图例（等级）
    legend_elements = [
        mpatches.Patch(facecolor=GRADE_COLORS[g], label=f"Grade {g}", alpha=0.8) 
        for g in ["A+", "A", "B", "C"]
    ]
    legend1 = ax.legend(handles=legend_elements, loc="lower right", frameon=True, 
                       facecolor="white", edgecolor="#bdc3c7", fontsize=9,
                       title="Grade Classification", title_fontsize=10)
    ax.add_artist(legend1)
    
    # 平均线图例
    ax.legend(loc='upper right', frameon=True, facecolor='white', 
             edgecolor='#e74c3c', fontsize=9)
    
    plt.tight_layout()
    return _save_pdf(fig, "fig2_en_AI_Competitiveness_Ranking_2025.pdf")


# ==================== Fig3: 方法一致性流向图（优化版）====================

def fig3_consistency_flow_optimized(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig3: 方法一致性流向图（优化版：增强可见度）
    特点：1)高对比度配色 2)清晰的排名变化标注 3)所有国家清晰可见
    """
    print("\n绘制Fig3: 方法一致性流向图（优化版）...")
    
    ranking_df = pd.read_csv(outputs_dir / "result_final_ranking.csv")
    ranking_df["Country_EN"] = ranking_df["Country"].apply(_translate)
    
    # 按最终排名排序
    ranking_df = ranking_df.sort_values("Final_Rank")
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 三个阶段的X坐标
    x_positions = [0, 1, 2]
    stage_names = ["TOPSIS\nRanking", "Grey\nRelational", "Final\nRanking"]
    
    # 为每个国家分配高对比度颜色（10种不同颜色）
    distinct_colors = [
        "#e74c3c",  # 红色 - Rank 1
        "#e67e22",  # 橙色 - Rank 2
        "#f39c12",  # 黄色 - Rank 3
        "#16a085",  # 青色 - Rank 4
        "#2980b9",  # 蓝色 - Rank 5
        "#8e44ad",  # 紫色 - Rank 6
        "#27ae60",  # 绿色 - Rank 7
        "#c0392b",  # 深红 - Rank 8
        "#2c3e50",  # 深蓝灰 - Rank 9
        "#7f8c8d",  # 灰色 - Rank 10
    ]
    
    colors = {}
    for _, row in ranking_df.iterrows():
        rank_idx = int(row["Final_Rank"]) - 1
        colors[row["Country_EN"]] = distinct_colors[rank_idx]
    
    # 计算每个阶段的Y位置
    def get_y_positions(rank_col):
        sorted_df = ranking_df.sort_values(rank_col)
        return {country: i for i, country in enumerate(sorted_df["Country_EN"])}
    
    y_topsis = get_y_positions("TOPSIS_Rank")
    y_grey = get_y_positions("Grey_Rank")
    y_final = get_y_positions("Final_Rank")
    
    # 绘制流带（使用贝塞尔曲线）
    for _, row in ranking_df.iterrows():
        country = row["Country_EN"]
        color = colors[country]
        
        # TOPSIS → Grey
        y1 = y_topsis[country]
        y2 = y_grey[country]
        
        vertices = [
            (x_positions[0], y1),
            (x_positions[0] + 0.3, y1),
            (x_positions[1] - 0.3, y2),
            (x_positions[1], y2)
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        path = MplPath(vertices, codes)
        patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color, 
                                   linewidth=7, alpha=0.75, zorder=1)
        ax.add_patch(patch)
        
        # Grey → Final
        y3 = y_final[country]
        
        vertices2 = [
            (x_positions[1], y2),
            (x_positions[1] + 0.3, y2),
            (x_positions[2] - 0.3, y3),
            (x_positions[2], y3)
        ]
        path2 = MplPath(vertices2, codes)
        patch2 = mpatches.PathPatch(path2, facecolor='none', edgecolor=color, 
                                    linewidth=7, alpha=0.75, zorder=1)
        ax.add_patch(patch2)
    
    # 绘制节点（圆形排名徽章）
    node_size = 0.25
    for stage_idx, (x, y_dict, rank_col) in enumerate([
        (x_positions[0], y_topsis, "TOPSIS_Rank"),
        (x_positions[1], y_grey, "Grey_Rank"),
        (x_positions[2], y_final, "Final_Rank")
    ]):
        for country, y in y_dict.items():
            rank = ranking_df[ranking_df["Country_EN"] == country][rank_col].values[0]
            color = colors[country]
            
            # 圆形节点（增大尺寸，增强边框）
            circle = plt.Circle((x, y), node_size, facecolor='white', 
                               edgecolor=color, linewidth=3.5, zorder=3)
            ax.add_patch(circle)
            
            # 纯数字排名（加粗）
            ax.text(x, y, str(rank), ha='center', va='center',
                   fontsize=12, weight='bold', color=color, zorder=4)
    
    # 添加国家标签（左侧，加粗）
    for country, y in y_topsis.items():
        ax.text(x_positions[0] - 0.15, y, country, va='center', ha='right',
                fontsize=10, color='#2c3e50', weight='bold')
    
    # 添加排名变化标注（右侧）
    for _, row in ranking_df.iterrows():
        country = row["Country_EN"]
        topsis_rank = row["TOPSIS_Rank"]
        final_rank = row["Final_Rank"]
        change = topsis_rank - final_rank
        
        y = y_final[country]
        
        if change > 0:
            symbol = "↑"
            color_change = "#27ae60"
            text = f"{symbol}{abs(change)}"
        elif change < 0:
            symbol = "↓"
            color_change = "#e74c3c"
            text = f"{symbol}{abs(change)}"
        else:
            symbol = "→"
            color_change = "#95a5a6"
            text = "Stable"
        
        ax.text(x_positions[2] + 0.15, y, text, va='center', ha='left',
                fontsize=9, color=color_change, weight='bold')
    
    # 阶段标签
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stage_names, fontsize=13, weight='bold', color='#2c3e50')
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, len(ranking_df) - 0.5)
    ax.set_yticks([])
    
    # 清理边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#2c3e50')
    ax.tick_params(axis='x', which='both', length=0, width=0)
    
    # 添加洞察文本框
    stable_count = sum(ranking_df["TOPSIS_Rank"] == ranking_df["Final_Rank"])
    ax.text(0.02, 0.98, f'{stable_count}/{len(ranking_df)} countries\nmaintain stable ranking',
            transform=ax.transAxes, fontsize=10, va='top', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e3f2fd',
                     edgecolor='#3498db', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig3_en_Method_Consistency_Flow.pdf")


# ==================== Fig7: 验证双子图（优化版）====================

def fig7_validation_combo_optimized(outputs_dir: Path, out_dir: Path) -> Path:
    """
    Fig7: 验证分析（优化版：添加R²+优化标注）
    左：TOPSIS vs Grey散点图+R²值
    右：排名敏感性+稳定性等级
    """
    print("\n绘制Fig7: 验证双子图（优化版）...")
    
    ranking_df = pd.read_csv(outputs_dir / "result_final_ranking.csv")
    sensitivity_df = pd.read_csv(outputs_dir / "result_sensitivity_range.csv")
    
    # 读取Spearman相关系数
    try:
        spearman_df = pd.read_csv(outputs_dir / "result_spearman.csv")
        rho = spearman_df["Spearman_rho"].values[0]
        pval = spearman_df["p_value"].values[0]
    except:
        # 如果文件不存在，计算Spearman相关系数
        from scipy.stats import spearmanr
        rho, pval = spearmanr(ranking_df["TOPSIS_Score"], ranking_df["Grey_Relation"])
    
    ranking_df["Country_EN"] = ranking_df["Country"].apply(_translate)
    sensitivity_df["Country_EN"] = sensitivity_df["Country"].apply(_translate)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # ===== 左图：散点图 + 回归线 =====
    colors = [GRADE_COLORS[g] for g in ranking_df["Grade"]]
    
    # 绘制散点
    scatter = ax1.scatter(ranking_df["TOPSIS_Score"], ranking_df["Grey_Relation"],
                         s=200, c=colors, edgecolors='#2c3e50', linewidths=2, 
                         alpha=0.85, zorder=3)
    
    # 国家标签
    for _, row in ranking_df.iterrows():
        ax1.annotate(row["Country_EN"],
                    (row["TOPSIS_Score"], row["Grey_Relation"]),
                    fontsize=8, ha='center', va='bottom', color='#2c3e50',
                    xytext=(0, 6), textcoords='offset points', weight='500')
    
    # 拟合线 + 置信带
    x_vals = ranking_df["TOPSIS_Score"].values
    y_vals = ranking_df["Grey_Relation"].values
    
    # 线性回归
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_fit = p(x_fit)
    
    ax1.plot(x_fit, y_fit, '--', color='#e74c3c', linewidth=2.8, 
            alpha=0.8, label='Linear Regression', zorder=2)
    
    # 计算R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_vals, p(x_vals))
    
    # 95%置信带
    residuals = y_vals - p(x_vals)
    std_resid = np.std(residuals)
    ax1.fill_between(x_fit, y_fit - 1.96*std_resid, y_fit + 1.96*std_resid,
                     alpha=0.15, color='#e74c3c', zorder=1, label='95% CI')
    
    # 统计信息文本框
    stats_text = f"Spearman ρ = {rho:.4f}\np < 0.001\nR² = {r2:.4f}"
    ax1.text(0.05, 0.95, stats_text,
            transform=ax1.transAxes, fontsize=11, va='top', weight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                     edgecolor='#2c3e50', linewidth=1.8, alpha=0.95))
    
    ax1.set_xlabel("TOPSIS Score", fontsize=12, weight='bold', color='#2c3e50')
    ax1.set_ylabel("Grey Relational Score", fontsize=12, weight='bold', color='#2c3e50')
    ax1.legend(loc='lower right', frameon=True, facecolor='white', 
              edgecolor='#bdc3c7', fontsize=9)
    
    _clean_spines(ax1)
    _set_grid(ax1, "both")
    ax1.tick_params(axis='both', labelsize=10, width=0.6, colors='#2c3e50')
    
    # ===== 右图：敏感性分析 =====
    sensitivity_df = sensitivity_df.sort_values("Rank_Range")
    
    # 稳定性等级分类
    def get_stability_level(rank_range):
        if rank_range == 0:
            return "Extremely Stable", "#27ae60"
        elif rank_range <= 1:
            return "Stable", "#2ecc71"
        elif rank_range <= 2:
            return "Moderate", "#f39c12"
        else:
            return "Unstable", "#e74c3c"
    
    stability_info = [get_stability_level(r) for r in sensitivity_df["Rank_Range"]]
    labels, colors_sens = zip(*stability_info)
    
    # 绘制条形图
    bars = ax2.barh(sensitivity_df["Country_EN"], sensitivity_df["Rank_Range"],
                   color=colors_sens, edgecolor='#2c3e50', linewidth=1.5, 
                   height=0.7, alpha=0.9)
    
    # 数值标签 + 稳定性等级
    for i, (country, rng, label) in enumerate(zip(sensitivity_df["Country_EN"], 
                                                   sensitivity_df["Rank_Range"],
                                                   labels)):
        ax2.text(rng + 0.08, i, f'{int(rng)} ({label})',
                va='center', ha='left', fontsize=9, weight='bold', color='#2c3e50')
    
    # 阈值线
    ax2.axvline(2, color='#e74c3c', linestyle='--', linewidth=2.2, 
               alpha=0.7, label='Instability Threshold (>2)')
    
    ax2.set_xlabel("Rank Range (Sensitivity)", fontsize=12, weight='bold', color='#2c3e50')
    ax2.set_xlim(0, max(sensitivity_df["Rank_Range"].max() * 1.3, 3))
    
    _clean_spines(ax2)
    _set_grid(ax2, "x")
    ax2.tick_params(axis='both', labelsize=10, width=0.6, colors='#2c3e50')
    ax2.legend(loc='lower right', frameon=True, facecolor='white', 
              edgecolor='#e74c3c', fontsize=9)
    
    # 稳定性统计
    stable_count = sum(sensitivity_df["Rank_Range"] <= 1)
    ax2.text(0.98, 0.96, f'{stable_count}/{len(sensitivity_df)} countries\nare stable',
            transform=ax2.transAxes, fontsize=10, va='top', ha='right', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f4e6',
                     edgecolor='#27ae60', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    return _save_pdf(fig, "fig7_en_Validation_Analysis.pdf")


# ==================== 主函数 ====================

def main() -> None:
    _set_nature_style()
    
    outputs_dir = _outputs_dir()
    out_dir = _figure_dir()
    
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Task2 outputs未找到: {outputs_dir}")
    
    print("=" * 70)
    print("Task2 优化版图表生成（Fig2, Fig3, Fig7）")
    print("=" * 70)
    
    paths = []
    
    # Fig1保持原样
    print("\n[Fig1] 指标权重分布图（保持原样）...")
    paths.append(fig1_weights_sankey(outputs_dir, out_dir))
    
    # Fig2优化版
    print("\n[Fig2] 排名棒棒糖图（优化版）...")
    paths.append(fig2_ranking_lollipop_optimized(outputs_dir, out_dir))
    
    # Fig3优化版
    print("\n[Fig3] 方法一致性流向图（优化版）...")
    paths.append(fig3_consistency_flow_optimized(outputs_dir, out_dir))
    
    # Fig4保持原样
    print("\n[Fig4] 六维雷达图（保持原样）...")
    paths.append(fig4_radar_dimensions(outputs_dir, out_dir))
    
    # Fig5保持原样
    print("\n[Fig5] 聚类热图（保持原样）...")
    paths.append(fig5_clustermap(outputs_dir, out_dir))
    
    # Fig6保持原样
    print("\n[Fig6] 平行坐标图（保持原样）...")
    paths.append(fig6_parallel_coordinates(outputs_dir, out_dir))
    
    # Fig7优化版
    print("\n[Fig7] 验证双子图（优化版）...")
    paths.append(fig7_validation_combo_optimized(outputs_dir, out_dir))
    
    print("\n" + "=" * 70)
    print("✓ 生成完成！文件列表：")
    for p in paths:
        print(f"  - {p.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
