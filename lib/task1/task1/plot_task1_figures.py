# -*- coding: utf-8 -*-
"""plot_task1_figures.py

按论文 body/5task1.tex 的要求，从 Task1 outputs 的 CSV 复现生成 5 张图：
- Fig1 相关性热力图（correlation_matrix.csv）
- Fig2 层次聚类树状图（correlation_matrix.csv -> d=1-|r| -> average linkage）
- Fig3 PCA 方差贡献（pca_variance.csv：柱状 + 累积折线）
- Fig4 要素重要性 Top-k + PCA 载荷热图（factor_importance.csv + pca_loadings.csv）
- Fig5 强相关网络/弦图（strong_correlations.csv）

输出目录：<repo>/figure/task1/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
import matplotlib.colors as mcolors

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def _set_mcm_style() -> None:
    """统一论文风格：Times New Roman 优先 + serif。"""
    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    except Exception:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.unicode_minus"] = False


def _translation_map() -> Dict[str, str]:
    # 与 fig.py 保持一致（用于英文出图）
    return {
        "AI研究人员数量": "No. of AI Researchers",
        "顶尖AI学者数量": "Top AI Scholars",
        "AI毕业生数量": "No. of AI Graduates",
        "AI企业数量": "No. of AI Enterprises",
        "AI市场规模": "AI Market Size",
        "AI应用渗透率": "AI Penetration Rate",
        "大模型数量": "No. of LLMs",
        "AI社会信任度": "Public Trust in AI",
        "AI政策数量": "No. of AI Policies",
        "AI补贴金额": "AI Subsidies",
        "企业研发支出": "Corporate R&D Expenditure",
        "政府AI投资": "Government AI Investment",
        "国际AI投资": "International AI Investment",
        "5G覆盖率": "5G Coverage",
        "GPU集群规模": "GPU Cluster Scale",
        "互联网带宽": "Internet Bandwidth",
        "互联网普及率": "Internet Penetration",
        "电能生产": "Power Generation",
        "AI算力平台": "AI Computing Platforms",
        "数据中心数量": "No. of Data Centers",
        "TOP500上榜数": "No. of TOP500 Systems",
        "AI_Book数量": "No. of AI Books",
        "AI_Dataset数量": "No. of AI Datasets",
        "GitHub项目数": "GitHub Repositories",
    }


def _t(name: str) -> str:
    return _translation_map().get(name, name)


def _repo_root() -> Path:
    # <repo>/lib/task1/task1/plot_task1_figures.py
    # parents[0]=<repo>/lib/task1/task1
    # parents[1]=<repo>/lib/task1
    # parents[2]=<repo>/lib
    # parents[3]=<repo>
    return Path(__file__).resolve().parents[3]


def _task1_outputs_dir() -> Path:
    return Path(__file__).resolve().parent / "outputs"


def _figure_dir() -> Path:
    return _repo_root() / "figure" / "task1"


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def fig1_correlation_heatmap(outputs_dir: Path, out_dir: Path) -> Path:
    """Fig1：相关性热力图（数据：correlation_matrix.csv）"""
    R = pd.read_csv(outputs_dir / "correlation_matrix.csv", index_col=0)

    # ✅ 原因1：对变量做层次聚类排序（行列同序重排），让同类变量聚集成块
    # 使用距离 d=1-|r|，average linkage（与 Fig2 一致）
    R = (R + R.T) / 2
    np.fill_diagonal(R.values, 1.0)
    D = 1 - np.abs(R.values)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=True)
    Z = linkage(condensed, method="average")
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    R = R.iloc[leaves, leaves]

    # ✅ 原因3：画布用正方形；矩阵单元格保持正方形
    fig, ax = plt.subplots(figsize=(9, 9))

    # ✅ 原因2+6：按用户要求的“更柔和学术级配色”，并显式以 0 为中心
    # Nature/Science风格：深红 → 灰白 → 深蓝（更专业的科研色彩）
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sci_correlation",
        ["#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4"],
        N=256,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    im = ax.imshow(R.values, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")

    labels = [_t(c) for c in R.columns]
    n = len(labels)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # x 轴标签斜排（按用户最新要求）
    ax.set_xticklabels(labels, rotation=60, ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    # ✅ 原因4：单元格细边框（浅灰网格线）
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#d0d0d0", linestyle="-", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 去掉热力图外圈“大正方形边框”（axes spines）
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 保留刻度，但更细一点
    ax.tick_params(axis="both", which="major", width=0.3, length=2)

    # 顶部不显示图片名（标题）
    ax.set_title("")
    # 右侧色条：单独放一个轴，便于把色条往右靠
    # 先把主图留出右侧空白，再把色条放到更右的位置
    fig.subplots_adjust(left=0.20, bottom=0.26, right=0.90, top=0.98)
    cax = fig.add_axes([0.93, 0.34, 0.015, 0.54])  # [x0, y0, width, height]
    cbar = fig.colorbar(im, cax=cax)
    # 与示例一致：色条不额外加标题
    cbar.set_label("")
    cbar.ax.tick_params(labelsize=7, length=2, width=0.3)

    out_path = out_dir / "fig1_en_Correlation Heatmap of AI Development Factors.pdf"
    _save_pdf(fig, out_path)
    return out_path


def fig2_hierarchical_dendrogram(outputs_dir: Path, out_dir: Path) -> Path:
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
    
    # 使用专业配色方案
    cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    
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


def fig3_variance_explained(outputs_dir: Path, out_dir: Path) -> Path:
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


def _select_m_by_cum_ratio(variance_df: pd.DataFrame, threshold: float = 0.85) -> int:
    cum = variance_df["Cumulative_Ratio"].values
    return int(np.argmax(cum >= threshold) + 1)


def fig4_importance_and_loadings(outputs_dir: Path, out_dir: Path, top_k: int = 10) -> Path:
    """Fig4：Top-k 要素重要性条形图 + PCA 载荷热图（辅助说明）。"""
    imp = pd.read_csv(outputs_dir / "factor_importance.csv", index_col=0)
    var_df = pd.read_csv(outputs_dir / "pca_variance.csv", index_col=0)
    loadings = pd.read_csv(outputs_dir / "pca_loadings.csv", index_col=0)

    m = _select_m_by_cum_ratio(var_df, threshold=0.85)

    imp_sorted = imp.sort_values("Importance", ascending=False).head(top_k).copy()
    imp_sorted.index = [ _t(x) for x in imp_sorted.index ]

    # 载荷热图：只展示 Top-k 指标在前 m 个主成分上的载荷
    pc_cols = [f"PC{i+1}" for i in range(m)]
    topk_cn = imp.sort_values("Importance", ascending=False).head(top_k).index.tolist()
    load_sub = loadings.loc[topk_cn, pc_cols].copy()
    load_sub.index = [_t(x) for x in load_sub.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.05, 1.2]})

    # 左：重要性条形图（水平）- 使用RdYlGn渐变色谱（红黄绿）
    values = imp_sorted["Importance"].values[::-1]
    colors = plt.cm.RdYlGn(values / values.max())
    ax1.barh(imp_sorted.index[::-1], values, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Importance", fontsize=10)
    ax1.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.6)
    ax1.spines["bottom"].set_linewidth(0.6)
    ax1.tick_params(axis="both", which="major", labelsize=9, width=0.3, length=3)

    # 右：载荷热图
    vmax = float(np.nanmax(np.abs(load_sub.values))) if load_sub.size else 1.0
    im = ax2.imshow(load_sub.values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax2.set_xticks(range(len(pc_cols)))
    ax2.set_xticklabels(pc_cols, rotation=0, fontsize=9)
    ax2.set_yticks(range(len(load_sub.index)))
    ax2.set_yticklabels(load_sub.index, fontsize=8)
    ax2.tick_params(axis="both", which="major", width=0.3, length=2)
    # 弱化边框
    for spine in ax2.spines.values():
        spine.set_linewidth(0.3)
        spine.set_alpha(0.3)
    
    # 色条变小并右移
    fig.subplots_adjust(left=0.16, right=0.88, top=0.96, bottom=0.10, wspace=0.35)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.50])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Loading", fontsize=8)
    cbar.ax.tick_params(labelsize=7, width=0.25, length=1.5)
    cbar.outline.set_linewidth(0.3)

    out_path = out_dir / "fig4_en_Factor Importance Ranking Bar Chart.pdf"
    _save_pdf(fig, out_path)
    return out_path


def fig5_strong_correlation_chord(outputs_dir: Path, out_dir: Path) -> Path:
    """Fig5：强相关网络/弦图（数据：strong_correlations.csv）。"""
    df = pd.read_csv(outputs_dir / "strong_correlations.csv")
    df = df.rename(columns={"Indicator_1": "u", "Indicator_2": "v", "Correlation": "corr"})

    df["u"] = df["u"].map(_t)
    df["v"] = df["v"].map(_t)

    nodes = sorted(set(df["u"]).union(set(df["v"])))
    node_angles = {node: angle for node, angle in zip(nodes, np.linspace(0, 2 * np.pi, len(nodes), endpoint=False))}

    node_weights = {node: 0.0 for node in nodes}
    for _, row in df.iterrows():
        w = float(abs(row["corr"]))
        node_weights[row["u"]] += w
        node_weights[row["v"]] += w

    max_w = max(node_weights.values()) if node_weights else 1.0
    min_w = min(node_weights.values()) if node_weights else 0.0

    def node_size(w: float) -> float:
        if max_w == min_w:
            return 120
        return 60 + (w - min_w) / (max_w - min_w) * 180

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.axis("off")

    radius = 1.0
    label_radius = 1.18

    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    ring = plt.Circle((0, 0), radius, color="#bdbdbd", fill=False, linewidth=1.2, zorder=0, alpha=0.5)
    ax.add_artist(ring)

    df = df.assign(abs_corr=lambda x: x["corr"].abs()).sort_values("abs_corr")

    for _, row in df.iterrows():
        u, v, corr = row["u"], row["v"], float(row["corr"])
        a1, a2 = node_angles[u], node_angles[v]

        x1, y1 = radius * np.cos(a1), radius * np.sin(a1)
        x2, y2 = radius * np.cos(a2), radius * np.sin(a2)

        verts = [(x1, y1), (0, 0), (x2, y2)]
        codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
        path = MplPath(verts, codes)

        color = cmap(norm(corr))
        abs_corr = abs(corr)
        linewidth = (abs_corr**2) * 4.0
        alpha = 0.2 + (abs_corr**3) * 0.7

        patch = patches.PathPatch(path, facecolor="none", edgecolor=color, lw=linewidth, alpha=alpha, zorder=1)
        ax.add_patch(patch)

    for node, angle in node_angles.items():
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        ax.scatter(x, y, color="#404040", s=node_size(node_weights[node]), edgecolors="white", linewidth=1.5, zorder=10)

        deg = np.degrees(angle)
        if 90 < deg < 270:
            rotation = deg + 180
            ha = "right"
        else:
            rotation = deg
            ha = "left"

        lx, ly = label_radius * np.cos(angle), label_radius * np.sin(angle)
        ax.text(lx, ly, node, rotation=rotation, ha=ha, va="center", fontsize=10, family="serif")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.025, pad=0.15, shrink=0.5)
    cbar.set_label("Correlation Coefficient", fontsize=10, labelpad=8)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=9, width=0.3, length=2)

    out_path = out_dir / "fig5_en_Strong-Correlation Network or Chord Diagram.pdf"
    _save_pdf(fig, out_path)
    return out_path


def main() -> None:
    _set_mcm_style()

    outputs_dir = _task1_outputs_dir()
    out_dir = _figure_dir()

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Task1 outputs not found: {outputs_dir}")

    paths: List[Path] = []
    paths.append(fig1_correlation_heatmap(outputs_dir, out_dir))
    paths.append(fig2_hierarchical_dendrogram(outputs_dir, out_dir))
    paths.append(fig3_variance_explained(outputs_dir, out_dir))
    paths.append(fig4_importance_and_loadings(outputs_dir, out_dir, top_k=10))
    paths.append(fig5_strong_correlation_chord(outputs_dir, out_dir))

    print("Generated figures:")
    for p in paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
