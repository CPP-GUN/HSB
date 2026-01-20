# -*- coding: utf-8 -*-
"""
Task4 中国AI投资优化可视化 - Nature/Science级图表
生成5张顶刊标准图表，输出到 figure/task4/

图表清单：
- Fig1: 环形图 - 六大维度投资分布
- Fig2: 棒棒糖图 - Top10投资重点
- Fig3: 对数刻度棒棒糖图 - 24指标增长率
- Fig4: 统一坐标系群组条形图 - 六大维度对比
- Fig5: 气泡图 - 投资效率四象限分析
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')


# ==================== 全局配置 ====================

def setup_style():
    """设置Nature/Science顶刊样式"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.unicode_minus': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })


# 六大维度配色方案
DIMENSION_COLORS = {
    "I_基础设施": "#1f77b4",  # 深蓝
    "T_人才": "#2ca02c",      # 绿色
    "P_政策": "#ff7f0e",      # 橙色
    "R_研发": "#9467bd",      # 紫色
    "A_应用": "#e377c2",      # 粉色
    "O_产出": "#8c564b",      # 棕色
}

# 中英文翻译映射
TRANSLATION_MAP = {
    # 维度
    "I_基础设施": "Infrastructure",
    "T_人才": "Talent",
    "P_政策": "Policy",
    "R_研发": "R&D",
    "A_应用": "Application",
    "O_产出": "Output",
    # 指标
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


def translate(text):
    """翻译中文到英文"""
    return TRANSLATION_MAP.get(text, text)


def clean_spines(ax):
    """清理坐标轴边框"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)


def get_paths():
    """获取路径"""
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(__file__).resolve().parent / "task4" / "outputs"
    output_dir = repo_root / "figure" / "task4"
    output_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, output_dir


# ==================== Fig1: 环形图 - 六大维度投资分布 ====================

def plot_fig1_donut_chart(data_dir, output_dir):
    """
    Fig1: 环形图 - 六大维度投资分布
    优化版：更好的配色、清晰的标签、显示金额和百分比
    """
    print("\n[Fig1] 绘制六大维度投资分布环形图...")
    
    # 读取数据
    df = pd.read_csv(data_dir / "dimension_distribution.csv")
    df['dimension_en'] = df['维度'].apply(translate)
    df = df.sort_values('占比_%', ascending=False)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 优化配色方案 - 使用更协调的渐变色
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    
    # 突出最大扇形（基础设施）
    explode = [0.08 if i == 0 else 0.02 for i in range(len(df))]
    
    # 绘制饼图 - 自定义标签格式
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return f'{pct:.1f}%\n{val}B¥'
        return my_autopct
    
    wedges, texts, autotexts = ax.pie(
        df['投资额_亿元'],
        labels=None,  # 不在扇形上显示维度名
        autopct=make_autopct(df['投资额_亿元']),
        startangle=90,
        colors=colors,
        explode=explode,
        wedgeprops=dict(width=0.35, edgecolor='white', linewidth=3),
        pctdistance=0.75
    )
    
    # 优化百分比和金额标签
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
        autotext.set_ha('center')
    
    # 中心圆显示总额
    centre_circle = Circle((0, 0), 0.65, fc='white', linewidth=3, edgecolor='#2c3e50')
    ax.add_artist(centre_circle)
    
    ax.text(0, 0.05, 'Total Investment', ha='center', va='center',
            fontsize=14, weight='bold', color='#34495e')
    ax.text(0, -0.15, '10,000', ha='center', va='center',
            fontsize=28, weight='bold', color='#2c3e50')
    ax.text(0, -0.35, 'Billion RMB', ha='center', va='center',
            fontsize=12, color='#34495e')
    
    # 优化图例 - 显示维度名、金额、百分比
    legend_labels = []
    for _, row in df.iterrows():
        label = f"{row['dimension_en']}: {row['投资额_亿元']:.0f}B¥ ({row['占比_%']:.1f}%)"
        legend_labels.append(label)
    
    legend = ax.legend(
        legend_labels,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        frameon=True,
        fontsize=11,
        title='Investment Breakdown',
        title_fontsize=13,
        framealpha=0.95,
        edgecolor='#34495e',
        fancybox=True,
        shadow=True
    )
    
    # 为图例添加颜色标记
    for i, (text, color) in enumerate(zip(legend.get_texts(), colors)):
        text.set_weight('500')
    
    # 添加标题
    ax.set_title('Investment Distribution Across Six AI Dimensions',
                fontsize=15, weight='bold', pad=20, color='#2c3e50')
    
    # 保存
    output_path = output_dir / "fig1_dimension_investment_donut.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"✓ 保存: {output_path}")
    return output_path


# ==================== Fig2: 棒棒糖图 - Top10投资重点 ====================

def plot_fig2_top10_lollipop(data_dir, output_dir):
    """
    Fig2: 棒棒糖图 - Top10投资重点
    纯数字排名标记
    """
    print("\n[Fig2] 绘制Top10投资重点棒棒糖图...")
    
    # 读取数据
    df = pd.read_csv(data_dir / "investment_allocation.csv")
    df['indicator_en'] = df['指标'].apply(translate)
    top10 = df.head(10).iloc[::-1]  # 反转顺序
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top10))
    
    # 绘制棒棒糖
    for i, (idx, row) in enumerate(top10.iterrows()):
        inv = row['投资额_亿元']
        rank = row['排名']
        
        # 颜色：前3名特殊色
        if rank == 1:
            color = '#FFD700'  # 金色
        elif rank == 2:
            color = '#C0C0C0'  # 银色
        elif rank == 3:
            color = '#CD7F32'  # 铜色
        else:
            color = '#3498db'  # 蓝色
        
        # 线条
        ax.plot([0, inv], [i, i], color=color, linewidth=3, alpha=0.7, zorder=1)
        # 圆点
        ax.scatter(inv, i, s=350, color=color, edgecolor='white', 
                  linewidth=2, zorder=3, alpha=0.9)
        
        # 左侧纯数字排名
        ax.text(-80, i, str(rank), ha='center', va='center',
                fontsize=14, weight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.4', facecolor=color,
                         edgecolor='white', linewidth=2))
        
        # 右侧数值标签
        ax.text(inv + 40, i, f'{inv:.0f}B¥\n({row["占比_%"]:.1f}%)',
                va='center', ha='left', fontsize=9, weight='bold',
                color='#2c3e50')
    
    # Y轴标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10['indicator_en'], fontsize=10)
    
    # X轴
    ax.set_xlabel('Investment Amount (Billion RMB)', fontsize=12, weight='bold')
    ax.set_xlim(-120, max(top10['投资额_亿元']) * 1.25)
    
    # 平均线
    avg_inv = top10['投资额_亿元'].mean()
    ax.axvline(avg_inv, color='#e74c3c', linestyle='--', linewidth=1.5,
              alpha=0.6, label=f'Average: {avg_inv:.0f}B¥')
    ax.legend(loc='lower right', fontsize=9)
    
    # 网格和样式
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    clean_spines(ax)
    
    # 累计占比标注
    cumsum = df.head(10)['占比_%'].sum()
    ax.text(1.045, 0.97, f'Top 10: {cumsum:.1f}% of Total',
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498db',
                     edgecolor='white', linewidth=1.5, alpha=0.9),
            color='white', weight='bold')
    
    # 保存
    output_path = output_dir / "fig2_top10_investment_lollipop.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"✓ 保存: {output_path}")
    return output_path


# ==================== Fig3: 对数刻度棒棒糖图 - 24指标增长率 ====================

def plot_fig3_growth_rate_lollipop(data_dir, output_dir):
    """
    Fig3: 对数刻度棒棒糖图 - 24指标增长率
    处理极端值（4600%），纯数字Top3标记
    """
    print("\n[Fig3] 绘制24指标增长率棒棒糖图（对数刻度）...")
    
    # 读取数据
    df = pd.read_csv(data_dir / "before_after_comparison.csv")
    df['indicator_en'] = df['指标'].apply(translate)
    
    # 处理0值（对数刻度无法显示0）
    df['增长率_display'] = df['增长率_%'].replace(0, 0.01)
    df = df.sort_values('增长率_%', ascending=True)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_pos = np.arange(len(df))
    
    # 颜色分级
    colors = []
    for rate in df['增长率_%']:
        if rate > 1000:
            colors.append('#27ae60')  # 深绿
        elif rate > 500:
            colors.append('#2ecc71')  # 浅绿
        elif rate > 100:
            colors.append('#3498db')  # 蓝色
        elif rate > 0:
            colors.append('#95a5a6')  # 灰色
        else:
            colors.append('#e74c3c')  # 红色
    
    # 绘制棒棒糖
    for i, (idx, row) in enumerate(df.iterrows()):
        rate_display = row['增长率_display']
        color = colors[i]
        
        # 线条
        ax.plot([0.01, rate_display], [i, i], color=color, linewidth=2.5, alpha=0.7)
        # 圆点
        ax.scatter(rate_display, i, s=200, color=color, edgecolor='white',
                  linewidth=1.5, zorder=3, alpha=0.9)
    
    # 对数刻度
    ax.set_xscale('log')
    ax.set_xlabel('Growth Rate (%) - Logarithmic Scale', fontsize=12, weight='bold')
    ax.set_xlim(0.005, max(df['增长率_display']) * 2)
    
    # Y轴标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['indicator_en'], fontsize=9)
    
    # 数值标签
    for i, (idx, row) in enumerate(df.iterrows()):
        rate = row['增长率_%']
        rate_display = row['增长率_display']
        
        if rate == 0:
            label = '0%'
            label_color = '#e74c3c'
            label_weight = 'bold'
        elif rate > 1000:
            label = f'{rate:.0f}%'
            label_color = '#27ae60'
            label_weight = 'bold'
        else:
            label = f'{rate:.1f}%'
            label_color = '#2c3e50'
            label_weight = 'normal'
        
        ax.text(rate_display * 1.3, i, label, va='center', ha='left',
                fontsize=8, color=label_color, weight=label_weight)
    
    # Top3纯数字徽章
    top3_indices = [len(df) - 1, len(df) - 2, len(df) - 3]
    badge_colors = ['#FFD700', '#C0C0C0', '#CD7F32']
    for rank, (idx, color) in enumerate(zip(top3_indices, badge_colors), 1):
        ax.text(0.008, idx, str(rank), ha='center', va='center',
                fontsize=13, weight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.4', facecolor=color,
                         edgecolor='white', linewidth=2))
    
    # 图例
    legend_elements = [
        mpatches.Patch(color='#27ae60', label='Extreme (>1000%)'),
        mpatches.Patch(color='#2ecc71', label='High (500-1000%)'),
        mpatches.Patch(color='#3498db', label='Medium (100-500%)'),
        mpatches.Patch(color='#95a5a6', label='Low (0-100%)'),
        mpatches.Patch(color='#e74c3c', label='No Growth (0%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, title='Growth Level', title_fontsize=10)
    
    # 网格和样式
    ax.grid(axis='x', which='both', linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    clean_spines(ax)
    
    # 保存
    output_path = output_dir / "fig3_growth_rate_lollipop_log.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"✓ 保存: {output_path}")
    return output_path


# ==================== Fig4: 统一坐标系群组条形图 - 六大维度对比 ====================

def plot_fig4_unified_grouped_bar(data_dir, output_dir):
    """
    Fig4: 统一坐标系群组条形图 - 六大维度Top4指标对比
    同一Y轴，维度分隔线，顶部标签
    """
    print("\n[Fig4] 绘制六大维度统一坐标系对比图...")
    
    # 读取数据
    inv_df = pd.read_csv(data_dir / "investment_allocation.csv")
    dim_df = pd.read_csv(data_dir / "dimension_distribution.csv")
    
    inv_df['indicator_en'] = inv_df['指标'].apply(translate)
    dim_df['dimension_en'] = dim_df['维度'].apply(translate)
    
    # 维度映射
    dimension_map = {
        "GPU Clusters": "Infrastructure", "Internet Bandwidth": "Infrastructure",
        "TOP500 Systems": "Infrastructure", "Data Centers": "Infrastructure",
        "Power Generation": "Infrastructure", "AI Computing Platforms": "Infrastructure",
        "Internet Penetration": "Infrastructure", "5G Coverage": "Infrastructure",
        "AI Researchers": "Talent", "Top AI Scholars": "Talent", "AI Graduates": "Talent",
        "AI Policies": "Policy", "AI Subsidies": "Policy", "Public Trust in AI": "Policy",
        "Corporate R&D": "R&D", "Government AI Investment": "R&D", 
        "International AI Investment": "R&D",
        "AI Market Size": "Application", "AI Enterprises": "Application",
        "AI Penetration": "Application", "Large Models": "Application",
        "GitHub Projects": "Output", "AI Books": "Output", "AI Datasets": "Output",
    }
    inv_df['dimension'] = inv_df['indicator_en'].map(dimension_map)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(16, 9))
    
    dimension_order = ["Infrastructure", "Policy", "Talent", "R&D", "Application", "Output"]
    color_map = {
        "Infrastructure": "#1f77b4", "Talent": "#2ca02c",
        "Policy": "#ff7f0e", "R&D": "#9467bd",
        "Application": "#e377c2", "Output": "#8c564b"
    }
    
    x_offset = 0
    bar_width = 0.7
    gap = 1.5
    
    all_positions = []
    all_labels = []
    dimension_centers = []
    
    for dim in dimension_order:
        dim_data = inv_df[inv_df['dimension'] == dim].head(4)
        
        if len(dim_data) == 0:
            continue
        
        n_bars = len(dim_data)
        positions = np.arange(x_offset, x_offset + n_bars)
        
        # 渐变颜色
        base_color = np.array(plt.matplotlib.colors.to_rgb(color_map[dim]))
        colors_gradient = [tuple(base_color * (1 - 0.15 * i)) for i in range(n_bars)]
        
        # 绘制条形
        bars = ax.bar(positions, dim_data['投资额_亿元'], width=bar_width,
                     color=colors_gradient, edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # 数值标签
        for bar, (_, row) in zip(bars, dim_data.iterrows()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 20,
                   f'{height:.0f}', ha='center', va='bottom',
                   fontsize=9, weight='bold', color='#2c3e50')
        
        all_positions.extend(positions)
        all_labels.extend(dim_data['indicator_en'])
        dimension_centers.append((positions[0] + positions[-1]) / 2)
        
        x_offset += n_bars + gap
    
    # X轴标签
    ax.set_xticks(all_positions)
    ax.set_xticklabels([label.replace(' ', '\n') for label in all_labels],
                       fontsize=8, rotation=0, ha='center')
    
    # Y轴
    ax.set_ylabel('Investment Amount (Billion RMB)', fontsize=12, weight='bold')
    ax.set_ylim(0, max(inv_df.head(24)['投资额_亿元']) * 1.15)
    
    # 维度分隔线和标签
    for i, (dim, center) in enumerate(zip(dimension_order, dimension_centers)):
        if i > 0:
            sep_x = all_positions[sum(len(inv_df[inv_df['dimension'] == d].head(4)) 
                                     for d in dimension_order[:i])] - gap/2
            ax.axvline(sep_x, color='#34495e', linestyle='--', 
                      linewidth=1.2, alpha=0.5)
        
        # 顶部维度标签
        dim_total = dim_df[dim_df['dimension_en'] == dim]['投资额_亿元'].values
        dim_pct = dim_df[dim_df['dimension_en'] == dim]['占比_%'].values
        if len(dim_total) > 0:
            ax.text(center, ax.get_ylim()[1] * 0.95, 
                   f'{dim}\n{dim_total[0]:.0f}B¥ ({dim_pct[0]:.1f}%)',
                   ha='center', va='top', fontsize=10, weight='bold',
                   color=color_map[dim],
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=color_map[dim], linewidth=1.5, alpha=0.9))
    
    # 网格和样式
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    clean_spines(ax)
    
    # 保存
    output_path = output_dir / "fig4_unified_dimension_comparison.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"✓ 保存: {output_path}")
    return output_path


# ==================== Fig5: 气泡图 - 投资效率四象限分析 ====================

def plot_fig5_bubble_efficiency(data_dir, output_dir):
    """
    Fig5: 气泡图 - 投资效率四象限分析
    X轴=投资额, Y轴=增长率(对数), 大小=增长量, 颜色=维度
    """
    print("\n[Fig5] 绘制投资效率气泡图（四象限分析）...")
    
    # 读取数据
    inv_df = pd.read_csv(data_dir / "investment_allocation.csv")
    comp_df = pd.read_csv(data_dir / "before_after_comparison.csv")
    
    # 合并数据
    merged = inv_df.merge(comp_df, on='指标', how='inner')
    merged['indicator_en'] = merged['指标'].apply(translate)
    
    # 维度映射
    dimension_map = {
        "GPU集群规模": "Infrastructure", "互联网带宽": "Infrastructure",
        "TOP500上榜数": "Infrastructure", "数据中心数量": "Infrastructure",
        "电能生产": "Infrastructure", "AI算力平台": "Infrastructure",
        "互联网普及率": "Infrastructure", "5G覆盖率": "Infrastructure",
        "AI研究人员数量": "Talent", "顶尖AI学者数量": "Talent", "AI毕业生数量": "Talent",
        "AI政策数量": "Policy", "AI补贴金额": "Policy", "AI社会信任度": "Policy",
        "企业研发支出": "R&D", "政府AI投资": "R&D", "国际AI投资": "R&D",
        "AI市场规模": "Application", "AI企业数量": "Application",
        "AI应用渗透率": "Application", "大模型数量": "Application",
        "GitHub项目数": "Output", "AI_Book数量": "Output", "AI_Dataset数量": "Output",
    }
    merged['dimension'] = merged['指标'].map(dimension_map)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 数据
    x = merged['投资额_亿元'].values
    y = merged['增长率_%'].values
    sizes = merged['增长量'].values
    
    # 归一化气泡大小
    sizes_norm = (sizes / sizes.max()) * 2500 + 150
    
    # 颜色映射
    color_map = {
        "Infrastructure": "#1f77b4", "Talent": "#2ca02c",
        "Policy": "#ff7f0e", "R&D": "#9467bd",
        "Application": "#e377c2", "Output": "#8c564b"
    }
    colors = [color_map.get(d, '#95a5a6') for d in merged['dimension']]
    
    # 绘制气泡
    ax.scatter(x, y, s=sizes_norm, c=colors, alpha=0.6,
              edgecolors='white', linewidth=1.5)
    
    # 对数刻度Y轴
    ax.set_yscale('log')
    
    # 中位数分割线
    median_x = np.median(x)
    median_y = np.median(y)
    
    ax.axvline(median_x, color='#e74c3c', linestyle='--', linewidth=1.5,
              alpha=0.6, label=f'Median Investment: {median_x:.0f}B¥')
    ax.axhline(median_y, color='#e74c3c', linestyle='--', linewidth=1.5,
              alpha=0.6, label=f'Median Growth: {median_y:.0f}%')
    
    # 四象限标注
    quadrants = [
        (0.25, 0.95, "Low Investment\nHigh Return", '#27ae60', '#d5f4e6'),
        (0.75, 0.95, "High Investment\nHigh Return", '#3498db', '#d6eaf8'),
        (0.25, 0.05, "Low Investment\nLow Return", '#95a5a6', '#ecf0f1'),
        (0.75, 0.05, "High Investment\nLow Return", '#e67e22', '#fdebd0')
    ]
    
    for x_pos, y_pos, text, edge_color, face_color in quadrants:
        va = 'top' if y_pos > 0.5 else 'bottom'
        ax.text(x_pos, y_pos, text, transform=ax.transAxes,
                fontsize=10, ha='center', va=va, weight='bold',
                color=edge_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=face_color,
                         edgecolor=edge_color, linewidth=1.5, alpha=0.8))
    
    # 关键指标标注
    key_indicators = ["GPU Clusters", "AI Policies", "GitHub Projects", 
                     "AI Researchers", "Internet Bandwidth"]
    for _, row in merged.iterrows():
        if row['indicator_en'] in key_indicators:
            ax.annotate(row['indicator_en'].replace(' ', '\n'),
                       xy=(row['投资额_亿元'], row['增长率_%']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, weight='bold', color='#2c3e50',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#34495e', linewidth=1.2, alpha=0.9),
                       arrowprops=dict(arrowstyle='->', color='#34495e', linewidth=1))
    
    # 坐标轴
    ax.set_xlabel('Investment Amount (Billion RMB)', fontsize=12, weight='bold')
    ax.set_ylabel('Growth Rate (%) - Logarithmic Scale', fontsize=12, weight='bold')
    
    # 图例（维度）
    legend_elements = [mpatches.Patch(color=color_map[dim], label=dim, alpha=0.8)
                      for dim in color_map.keys()]
    legend1 = ax.legend(handles=legend_elements, loc='upper left',
                       fontsize=9, frameon=True, title='Dimension',
                       title_fontsize=10)
    ax.add_artist(legend1)
    
    # 图例（分割线）
    ax.legend(loc='lower left', fontsize=8, frameon=True)
    
    # 网格和样式
    ax.grid(linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    clean_spines(ax)
    
    # 保存
    output_path = output_dir / "fig5_bubble_efficiency_quadrant.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"✓ 保存: {output_path}")
    return output_path


# ==================== 主函数 ====================

def main():
    """主函数：生成所有5张图表"""
    print("=" * 70)
    print("Task4 可视化 - Nature/Science级图表生成")
    print("=" * 70)
    
    # 设置样式
    setup_style()
    
    # 获取路径
    data_dir, output_dir = get_paths()
    
    if not data_dir.exists():
        print(f"\n❌ 错误: 数据目录不存在 {data_dir}")
        return
    
    # 生成图表
    paths = []
    
    try:
        paths.append(plot_fig1_donut_chart(data_dir, output_dir))
        paths.append(plot_fig2_top10_lollipop(data_dir, output_dir))
        paths.append(plot_fig3_growth_rate_lollipop(data_dir, output_dir))
        paths.append(plot_fig4_unified_grouped_bar(data_dir, output_dir))
        paths.append(plot_fig5_bubble_efficiency(data_dir, output_dir))
        
        print("\n" + "=" * 70)
        print("✓ 生成完成！文件列表：")
        for p in paths:
            print(f"  - {p.name}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
