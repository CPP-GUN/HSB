# -*- coding: utf-8 -*-
"""plot_task1_figures_optimized.py

优化版Task1图表生成脚本
- Fig2: 层次聚类树状图（添加分组颜色+阈值线）
- Fig3: PCA方差解释图（添加85%阈值+突出前3PC+数值标签）
- Fig4: 要素重要性排名（添加排名徽章+百分比+简化载荷图）
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# 导入原有的辅助函数
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_task1_figures import (
    _set_mcm_style, _translation_map, _t, _repo_root,
    _task1_outputs_dir, _figure_dir, _save_pdf,
    fig1_correlation_heatmap, fig5_strong_correlation_chord
)


def fig2_hierarchical_dendrogram_optimized(outputs_dir: Path, out_dir: Path) -> Path:
    """Fig2：层次聚类树状图（优化版：分组颜色+阈值线+簇标注）"""
    R = pd.read_csv(outputs_dir / "correlation_matrix.csv", index_col=0)

    D = 1 - np.abs(R.values)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)

    condensed = squareform(D, checks=True)
    Z = linkage(condensed, method="average")

    labels = [_t(c) for c in R.columns]

    fig, ax = plt.subplots(figsize=(12, 6.6))

    # 优化1：使用颜色阈值自动分组（设置为0.3，约85%相似度）
    threshold = 0.3
    
    dend = dendrogram(
        Z,
        labels=labels,
        leaf_rotation=60,
        leaf_font_size=7,
        ax=ax,
        color_threshold=threshold,
        above_threshold_color='#999999'
    )

    # 优化2：添加阈值参考线
    ax.axhline(y=threshold, color='#e74c3c', linestyle='--', linewidth=1.8, 
              alpha=0.7, label=f'Threshold (d={threshold:.2f})')

    ax.set_ylabel("Distance (1 - |r|)", fontsize=10, weight='bold')

    # 轻量 y 轴网格
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.3)

    # 清理边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", which="major", width=0.3, length=0, pad=2)
    ax.tick_params(axis="y", which="major", labelsize=9, width=0.3, length=3)

    # 优化3：添加图例
    ax.legend(loc='upper right', fontsize=9, frameon=True, 
             edgecolor='#d0d0d0', framealpha=0.9)

    # 优化4：添加洞察文本框
    n_clusters = len(set([c for c in dend['color_list'] if c != '#999999']))
    ax.text(0.02, 0.96, f'{n_clusters} clusters identified\nat threshold {threshold:.2f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e3f2fd',
                     edgecolor='#3498db', linewidth=1.5, alpha=0.9))

    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.38)

    out_path = out_dir / "fig2_en_Hierarchical Clustering Dendrogram of AI Development Factors.pdf"
    _save_pdf(fig, out_path)
    return out_path


def fig3_variance_explained_optimized(outputs_dir: Path, out_dir: Path) -> Path:
    """Fig3：PCA 方差贡献率（优化版：阈值线+突出前3PC+数值标签+洞察）"""
    df = pd.read_csv(outputs_dir / "pca_variance.csv", index_col=0)

    pcs = list(df.index)
    var = (df["Variance_Ratio"].values * 100).astype(float)
    cum = (df["Cumulative_Ratio"].values * 100).astype(float)

    x = np.arange(len(pcs))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # 优化1：前3个PC用深色，其他用浅色
    colors = ['#2c7bb6' if i < 3 else '#abd9e9' for i in range(len(pcs))]
    
    # 绘制柱状图
    bars = ax.bar(x, var, color=colors, alpha=0.9, label="Variance Explained (%)", 
                  edgecolor="white", linewidth=0.8)
    
    # 优化2：添加数值标签（只在前5个PC上显示）
    for i, (bar, v) in enumerate(zip(bars, var)):
        if i < 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{v:.1f}%', ha='center', va='bottom', fontsize=8, weight='bold')
    
    # 绘制累积折线
    line = ax.plot(x, cum, color='#d6604d', marker='o', markersize=6, 
                   linewidth=2.2, label="Cumulative (%)", alpha=0.95, zorder=10)

    # 优化3：添加85%阈值线
    threshold_85 = 85
    ax.axhline(y=threshold_85, color='#e74c3c', linestyle='--', linewidth=2, 
              alpha=0.7, label='85% Threshold', zorder=5)
    
    # 找到累积方差超过85%的第一个PC
    pc_85_idx = np.argmax(cum >= threshold_85)
    ax.axvline(x=pc_85_idx, color='#e74c3c', linestyle=':', linewidth=1.5, 
              alpha=0.5, zorder=5)
    
    # 标注交点
    ax.scatter(pc_85_idx, cum[pc_85_idx], s=120, color='#e74c3c', 
              edgecolor='white', linewidth=2, zorder=15)
    ax.text(pc_85_idx + 0.3, cum[pc_85_idx] + 2, 
           f'PC{pc_85_idx+1}\n{cum[pc_85_idx]:.1f}%',
           fontsize=9, weight='bold', color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels(pcs, rotation=0, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percentage (%)", fontsize=11, weight='bold')
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True, 
             edgecolor='#d0d0d0', framealpha=0.95)
    
    # 清理边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=9, width=0.3, length=3)

    # 优化4：添加洞察文本框
    ax.text(0.02, 0.96, f'First 3 PCs explain\n{cum[2]:.2f}% of variance',
            transform=ax.transAxes, fontsize=10, va='top', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f4e6',
                     edgecolor='#27ae60', linewidth=1.5, alpha=0.9))

    out_path = out_dir / "fig3_en_Variance Explained Plot for Principal Components.pdf"
    _save_pdf(fig, out_path)
    return out_path


def fig4_importance_and_loadings_optimized(outputs_dir: Path, out_dir: Path, top_k: int = 10) -> Path:
    """Fig4：Top-k 要素重要性条形图（优化版：排名徽章+百分比+简化载荷图）"""
    imp = pd.read_csv(outputs_dir / "factor_importance.csv", index_col=0)
    loadings = pd.read_csv(outputs_dir / "pca_loadings.csv", index_col=0)

    # 只使用前3个主成分（简化）
    m = 3

    imp_sorted = imp.sort_values("Importance", ascending=False).head(top_k).copy()
    imp_sorted.index = [_t(x) for x in imp_sorted.index]

    # 载荷热力图：只展示 Top-k 指标在前3个PC上的载荷
    pc_cols = [f"PC{i+1}" for i in range(m)]
    topk_cn = imp.sort_values("Importance", ascending=False).head(top_k).index.tolist()
    load_sub = loadings.loc[topk_cn, pc_cols].copy()
    load_sub.index = [_t(x) for x in load_sub.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # 左：重要性条形图（水平）- 优化版
    values = imp_sorted["Importance"].values[::-1]
    contrib = imp_sorted["Contribution_%"].values[::-1]
    
    # 优化1：使用渐变色（深绿→浅黄）
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    colors = plt.cm.RdYlGn(norm(values))
    
    bars = ax1.barh(range(len(values)), values, color=colors, 
                    edgecolor="white", linewidth=1.2, alpha=0.9)
    
    # 优化2：添加排名徽章（纯数字1、2、3）
    for i, (rank, val, cont) in enumerate(zip(range(top_k, 0, -1), values, contrib)):
        # 左侧排名徽章
        if rank <= 3:
            badge_colors = {1: '#FFD700', 2: '#C0C0C0', 3: '#CD7F32'}
            badge_color = badge_colors[rank]
            ax1.text(-0.003, i, str(rank), ha='center', va='center',
                    fontsize=13, weight='bold', color='white',
                    bbox=dict(boxstyle='circle,pad=0.35', facecolor=badge_color,
                             edgecolor='white', linewidth=2))
        
        # 优化3：右侧百分比标签
        ax1.text(val + 0.0005, i, f'{cont:.1f}%',
                va='center', ha='left', fontsize=8.5, weight='bold', color='#2c3e50')
    
    ax1.set_yticks(range(len(values)))
    ax1.set_yticklabels(imp_sorted.index[::-1], fontsize=9.5)
    ax1.set_xlabel("Importance Score", fontsize=11, weight='bold')
    ax1.set_xlim(-0.005, max(values) * 1.15)
    ax1.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.8)
    ax1.spines["bottom"].set_linewidth(0.8)
    ax1.tick_params(axis="both", which="major", labelsize=9, width=0.3, length=3)

    # 右：载荷热力图（简化版：只显示前3个PC）
    vmax = float(np.nanmax(np.abs(load_sub.values))) if load_sub.size else 1.0
    im = ax2.imshow(load_sub.values, cmap="coolwarm", vmin=-vmax, vmax=vmax, 
                   aspect="auto", interpolation="nearest")
    
    ax2.set_xticks(range(len(pc_cols)))
    ax2.set_xticklabels(pc_cols, rotation=0, fontsize=10, weight='bold')
    ax2.set_yticks(range(len(load_sub.index)))
    ax2.set_yticklabels(load_sub.index, fontsize=9)
    ax2.tick_params(axis="both", which="major", width=0.3, length=2)
    
    # 添加数值标签（载荷值）
    for i in range(len(load_sub.index)):
        for j in range(len(pc_cols)):
            val = load_sub.values[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color, weight='bold')
    
    # 弱化边框
    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#d0d0d0')
    
    # 色条
    fig.subplots_adjust(left=0.14, right=0.88, top=0.96, bottom=0.10, wspace=0.40)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.50])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Loading", fontsize=9, weight='bold')
    cbar.ax.tick_params(labelsize=8, width=0.3, length=2)
    cbar.outline.set_linewidth(0.5)

    # 添加标题
    ax1.text(0.5, 1.02, 'Factor Importance Ranking', transform=ax1.transAxes,
            ha='center', fontsize=11, weight='bold')
    ax2.text(0.5, 1.02, 'PCA Loadings (Top 3 Components)', transform=ax2.transAxes,
            ha='center', fontsize=11, weight='bold')

    out_path = out_dir / "fig4_en_Factor Importance Ranking Bar Chart.pdf"
    _save_pdf(fig, out_path)
    return out_path


def main() -> None:
    _set_mcm_style()

    outputs_dir = _task1_outputs_dir()
    out_dir = _figure_dir()

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Task1 outputs not found: {outputs_dir}")

    print("=" * 70)
    print("Task1 优化版图表生成")
    print("=" * 70)

    paths = []
    
    # Fig1保持原样
    print("\n[Fig1] 相关性热力图（保持原样）...")
    paths.append(fig1_correlation_heatmap(outputs_dir, out_dir))
    
    # Fig2优化版
    print("\n[Fig2] 层次聚类树状图（优化版）...")
    paths.append(fig2_hierarchical_dendrogram_optimized(outputs_dir, out_dir))
    
    # Fig3优化版
    print("\n[Fig3] PCA方差解释图（优化版）...")
    paths.append(fig3_variance_explained_optimized(outputs_dir, out_dir))
    
    # Fig4优化版
    print("\n[Fig4] 要素重要性排名（优化版）...")
    paths.append(fig4_importance_and_loadings_optimized(outputs_dir, out_dir))
    
    # Fig5保持原样
    print("\n[Fig5] 强相关网络/弦图（保持原样）...")
    paths.append(fig5_strong_correlation_chord(outputs_dir, out_dir))

    print("\n" + "=" * 70)
    print("✓ 生成完成！文件列表：")
    for p in paths:
        print(f"  - {p.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
