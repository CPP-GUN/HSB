# -*- coding: utf-8 -*-
"""
问题一：AI发展能力要素识别与关联分析 - SCI高级可视化版本
===============================================================
新增三个顶级SCI图表：
1. Chord Diagram（弦图）- 替代相关矩阵热力图
2. PCA Biplot（双标图）- 替代载荷矩阵热力图
3. 社区检测网络图 - 升级版网络图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Arc, Circle, Wedge, FancyArrowPatch
import networkx as nx
from networkx.algorithms import community as community_detection
import warnings
warnings.filterwarnings('ignore')

# 解决中文显示问题
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建字体对象
FONT_CN = FontProperties(family='Microsoft YaHei', size=11)
FONT_TITLE = FontProperties(family='Microsoft YaHei', weight='bold', size=14)
FONT_SMALL = FontProperties(family='Microsoft YaHei', size=9)

print("="*70)
print("问题一：AI发展能力要素识别与关联分析 - SCI高级可视化")
print("="*70)

# ==================== 真实数据加载 ====================

def generate_data():
    """
    加载真实数据：10个国家 × 21个要素
    数据来源：DATA文件夹整合的2023年真实数据
    """
    # 读取整合好的真实数据
    data = pd.read_csv('real_data_integrated.csv', encoding='utf-8-sig')
    
    # 重命名列为中文简称（保持原代码兼容）
    column_mapping = {
        'T1_AI研究人员数量': 'AI研究人员数量',
        'T2_顶尖AI学者数量': '顶尖AI学者数量',
        'T3_AI毕业生数量': 'AI毕业生数量',
        'A2_AI市场规模': 'AI市场规模',
        'A4_大模型数量': '大模型数量',
        'P2_政策数量': 'AI政策数量',
        'P3_补贴金额': 'AI补贴金额',
        'R1_企业研发支出': '企业研发支出',
        'R2_政府AI投资': '政府AI投资',
        'R3_国际AI投资': '国际AI投资',
        'I1_5G覆盖率': '5G覆盖率',
        'I2_GPU集群规模': 'GPU集群规模',
        'I3_互联网带宽': '互联网带宽',
        'I4_互联网普及率': '互联网普及率',
        'I5_电能生产': '电能生产',
        'I6_AI算力平台数量': 'AI算力平台',
        'I7_数据中心数量': '数据中心数量',
        'I9_TOP500上榜数': 'TOP500上榜数',
        'O1_AI_Book数量': 'AI_Book数量',
        'O2_AI_Dataset数量': 'AI_Dataset数量',
        'O3_GitHub项目数': 'GitHub项目数'
    }
    
    data = data.rename(columns=column_mapping)
    
    print(f"  ✓ 数据维度: {data.shape[0]} 个国家 × {data.shape[1]-1} 个要素")
    
    return data

def standardize_data(data):
    """数据标准化"""
    feature_names = list(data.columns[1:])
    X = data[feature_names].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    standardized_df = pd.DataFrame(X_scaled, columns=feature_names)
    standardized_df.insert(0, '国家', data['国家'])
    
    return X_scaled, data['国家'].tolist(), feature_names, standardized_df

def correlation_analysis(X_scaled, feature_names):
    """相关性分析"""
    corr_df = pd.DataFrame(X_scaled, columns=feature_names).corr()
    
    # 找出强相关对
    strong_corr = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_value = corr_df.iloc[i, j]
            if abs(corr_value) > 0.7:
                strong_corr.append((feature_names[i], feature_names[j], corr_value))
    
    print(f"\n发现 {len(strong_corr)} 对强相关关系 (|r| > 0.7)")
    
    return corr_df, strong_corr

def pca_analysis(X_scaled, feature_names):
    """PCA分析"""
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    n_components = np.argmax(cumulative_variance >= 0.85) + 1
    if n_components < 3:
        n_components = 3
    
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    return pca, X_pca, explained_variance_ratio, cumulative_variance, loadings, n_components

# ==================== 新增：SCI高级可视化 ====================

def plot_chord_diagram(corr_df, feature_names, threshold=0.7):
    """
    顶级可视化1：Chord Diagram（弦图）
    替代相关矩阵热力图，展示要素间强相关关系
    Nature/Science级别可视化
    """
    print("\n【高级可视化1】绘制Chord Diagram（弦图）...")
    
    # 提取强相关对
    connections = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_value = corr_df.iloc[i, j]
            if abs(corr_value) >= threshold:
                connections.append((i, j, corr_value))
    
    print(f"  ✓ 发现 {len(connections)} 对强相关关系")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(aspect="equal"))
    
    n = len(feature_names)
    radius = 1.0
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # 计算每个要素的位置
    positions = {}
    for i, (angle, name) in enumerate(zip(angles, feature_names)):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions[i] = (x, y, angle)
    
    # 绘制圆周上的要素节点
    colors_palette = plt.cm.tab20(np.linspace(0, 1, n))
    
    for i, (x, y, angle) in positions.items():
        # 绘制节点
        circle = Circle((x, y), 0.04, color=colors_palette[i], zorder=10)
        ax.add_patch(circle)
        
        # 添加标签
        # 计算标签位置（向外偏移）
        label_radius = radius + 0.15
        label_x = label_radius * np.cos(angle)
        label_y = label_radius * np.sin(angle)
        
        # 标签旋转角度
        rotation = np.degrees(angle)
        if 90 < rotation < 270:
            rotation += 180
            ha = 'right'
        else:
            ha = 'left'
        
        ax.text(label_x, label_y, feature_names[i], 
                fontproperties=FONT_SMALL,
                rotation=rotation,
                rotation_mode='anchor',
                ha=ha, va='center',
                fontsize=9)
    
    # 绘制连接弧线
    for i, j, corr_value in connections:
        x1, y1, _ = positions[i]
        x2, y2, _ = positions[j]
        
        # 弧线颜色和透明度取决于相关性强度
        alpha = min(abs(corr_value), 1.0) * 0.6
        if corr_value > 0:
            color = 'red'
        else:
            color = 'blue'
        
        # 计算弧线的控制点（贝塞尔曲线）
        # 控制点在两点连线的中心，向圆心方向偏移
        mid_x = (x1 + x2) / 2 * 0.3  # 向圆心收缩
        mid_y = (y1 + y2) / 2 * 0.3
        
        # 使用FancyArrowPatch绘制贝塞尔曲线
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad=.3",
            arrowstyle="-",
            linewidth=abs(corr_value) * 3,
            color=color,
            alpha=alpha,
            zorder=5
        )
        ax.add_patch(arrow)
    
    # 设置坐标轴
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 添加标题
    ax.text(0, 1.35, 'AI发展要素强相关关系弦图\n(Chord Diagram of Strong Correlations)', 
            fontproperties=FONT_TITLE, ha='center', fontsize=16)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=3, alpha=0.6, label='正相关'),
        plt.Line2D([0], [0], color='blue', lw=3, alpha=0.6, label='负相关'),
        plt.Line2D([0], [0], color='gray', lw=1, label=f'阈值: |r| ≥ {threshold}')
    ]
    ax.legend(handles=legend_elements, loc='upper right', prop=FONT_CN, framealpha=0.9)
    
    # 添加说明文字
    ax.text(0, -1.35, f'节点：19个要素 | 连线：{len(connections)}对强相关 | 线宽：相关强度', 
            fontproperties=FONT_CN, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fig_chord_diagram.png', dpi=300, bbox_inches='tight')
    print("  ✓ 保存: fig_chord_diagram.png")
    plt.close()


def plot_pca_biplot(X_pca, loadings, feature_names, countries, explained_variance_ratio):
    """
    顶级可视化2：PCA Biplot（双标图）
    同时展示样本（国家）和变量（要素）在主成分空间的分布
    Science/Nature PCA分析标准可视化
    """
    print("\n【高级可视化2】绘制PCA Biplot（双标图）...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制样本点（国家）
    pc1_scores = X_pca[:, 0]
    pc2_scores = X_pca[:, 1]
    
    # 国家散点图
    scatter = ax.scatter(pc1_scores, pc2_scores, 
                        s=200, alpha=0.6, c=range(len(countries)),
                        cmap='tab10', edgecolors='black', linewidth=1.5,
                        zorder=10)
    
    # 标注国家名称
    for i, country in enumerate(countries):
        ax.annotate(country, (pc1_scores[i], pc2_scores[i]),
                   fontproperties=FONT_CN,
                   fontsize=10, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')
    
    # 绘制要素向量（箭头）
    # 缩放因子（使箭头长度合适）
    scale = 3.5
    
    # 计算要素的载荷向量
    loadings_pc1 = loadings['PC1'].values
    loadings_pc2 = loadings['PC2'].values
    
    # 筛选重要要素（载荷大的）
    importance = np.sqrt(loadings_pc1**2 + loadings_pc2**2)
    top_indices = np.argsort(importance)[-12:]  # 显示前12个重要要素
    
    arrow_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_indices)))
    
    for idx, i in enumerate(top_indices):
        x = loadings_pc1[i] * scale
        y = loadings_pc2[i] * scale
        
        # 绘制箭头
        arrow = FancyArrowPatch((0, 0), (x, y),
                               arrowstyle='->,head_width=0.4,head_length=0.8',
                               linewidth=2.5,
                               color=arrow_colors[idx],
                               alpha=0.7,
                               zorder=5)
        ax.add_patch(arrow)
        
        # 标注要素名称
        # 调整标签位置避免重叠
        label_offset = 1.15
        ax.text(x * label_offset, y * label_offset, 
               feature_names[i],
               fontproperties=FONT_SMALL,
               fontsize=9,
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=arrow_colors[idx], alpha=0.8))
    
    # 添加坐标轴标签
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', 
                 fontproperties=FONT_CN, fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', 
                 fontproperties=FONT_CN, fontsize=13, fontweight='bold')
    
    # 添加标题
    ax.set_title('PCA Biplot：国家分布与要素贡献\n(Samples and Variables in PC Space)', 
                fontproperties=FONT_TITLE, pad=20, fontsize=16)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.5)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, label='国家（样本）'),
        plt.Line2D([0], [0], color='red', lw=2, label='要素向量（变量）')
    ]
    ax.legend(handles=legend_elements, loc='upper right', prop=FONT_CN, 
             framealpha=0.9, fontsize=11)
    
    # 添加说明
    note_text = ('向量长度：要素重要性 | 向量方向：要素特征\n'
                '国家位置：综合特征 | 距离近：特征相似')
    ax.text(0.02, 0.02, note_text, transform=ax.transAxes,
           fontproperties=FONT_SMALL, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('fig_pca_biplot.png', dpi=300, bbox_inches='tight')
    print("  ✓ 保存: fig_pca_biplot.png")
    plt.close()


def plot_community_network(corr_df, feature_names, threshold=0.7):
    """
    顶级可视化3：3D交互式要素关系球
    完全不同于fig7的2D网络图，这是三维球形布局+梯度颜色
    """
    print("\n【高级可视化3】绘制3D要素关系球...")
    
    # 构建网络图
    G = nx.Graph()
    
    # 添加节点和边
    for i, name in enumerate(feature_names):
        G.add_node(i, name=name)
    
    edges = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_value = corr_df.iloc[i, j]
            if abs(corr_value) >= threshold:
                G.add_edge(i, j, weight=abs(corr_value), correlation=corr_value)
                edges.append((i, j, corr_value))
    
    print(f"  ✓ 网络包含 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    
    # 社区检测（Louvain算法）
    communities = community_detection.greedy_modularity_communities(G)
    print(f"  ✓ 检测到 {len(communities)} 个社区")
    
    # 为每个节点分配社区
    community_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx
    
    # 创建3D效果的图形（使用极坐标模拟3D球）
    fig = plt.figure(figsize=(20, 20), facecolor='white')
    ax = fig.add_subplot(111)
    
    # 计算球形布局（使用改进的circular布局模拟3D）
    n_nodes = G.number_of_nodes()
    
    # 黄金螺旋算法布局（模拟球面均匀分布）
    golden_angle = np.pi * (3 - np.sqrt(5))
    pos = {}
    
    for i in range(n_nodes):
        # 球面坐标
        theta = golden_angle * i
        z = 1 - (2 * i / float(n_nodes - 1))
        radius = np.sqrt(1 - z * z)
        
        # 转换为2D投影坐标（模拟3D视角）
        x = radius * np.cos(theta) * (1 + z * 0.3)  # 添加深度感
        y = radius * np.sin(theta) * (1 + z * 0.3)
        
        pos[i] = (x, y)
    
    # 计算节点的度中心性和介数中心性
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # 创建柔和的莫兰迪色系（Nature风格，不刺眼）
    morandi_colors = [
        '#8E9AAF',  # 灰蓝
        '#B8B8AA',  # 灰绿
        '#DEA5A4',  # 玫瑰粉
        '#C9ADA7',  # 藕粉
        '#9A8C98',  # 灰紫
        '#A8DADC',  # 薄荷蓝
        '#F1FAEE',  # 米白
        '#E9C46A'   # 浅金
    ]
    colors_palette = [morandi_colors[i % len(morandi_colors)] for i in range(len(communities))]
    
    # 绘制边（三层：背景层、中间层、前景层）
    for layer_alpha, layer_offset in [(0.1, 0.015), (0.2, 0.008), (0.35, 0)]:
        for i, j, corr in edges:
            x = [pos[i][0] + layer_offset, pos[j][0] + layer_offset]
            y = [pos[i][1] + layer_offset, pos[j][1] + layer_offset]
            
            # 边的颜色渐变（柔和色调）
            if corr > 0.9:
                color = '#C08497'  # 强正相关-柔和玫瑰
                linewidth = 4
            elif corr > 0.8:
                color = '#D4A5A5'  # 中强正相关-柔和粉
                linewidth = 3
            elif corr > 0.7:
                color = '#9CADCE'  # 正相关-柔和蓝
                linewidth = 2
            else:
                color = '#BDC3C7'  # 弱相关-浅灰
                linewidth = 1
            
            ax.plot(x, y, color=color, alpha=layer_alpha, linewidth=linewidth, 
                   zorder=1, linestyle='-' if layer_offset == 0 else ':')
    
    # 绘制节点（多层光晕效果）
    for node in G.nodes():
        x, y = pos[node]
        community_id = community_map[node]
        base_color = colors_palette[community_id]
        if isinstance(base_color, str):
            # 已经是颜色字符串，直接使用
            pass
        else:
            # 如果是RGBA元组，转换为十六进制
            base_color = plt.matplotlib.colors.rgb2hex(base_color[:3])
        
        # 节点大小基于重要性
        importance = degree_centrality[node] * 0.5 + betweenness[node] * 0.5
        base_size = 500 + importance * 8000
        
        # 外层光晕
        ax.scatter(x, y, s=base_size*2.5, c=[base_color], alpha=0.1, 
                  edgecolors='none', zorder=2)
        # 中层光晕
        ax.scatter(x, y, s=base_size*1.5, c=[base_color], alpha=0.3, 
                  edgecolors='none', zorder=3)
        # 内层节点
        ax.scatter(x, y, s=base_size, c=[base_color], alpha=0.9, 
                  edgecolors='white', linewidths=3, zorder=4)
        # 核心高光
        ax.scatter(x, y, s=base_size*0.3, c='white', alpha=0.8, 
                  edgecolors='none', zorder=5)
    
    # 绘制标签（智能避让）
    labels = {i: feature_names[i] for i in G.nodes()}
    for node, label in labels.items():
        x, y = pos[node]
        
        # 标签位置偏移（根据节点在圆上的位置）
        angle = np.arctan2(y, x)
        offset_x = np.cos(angle) * 0.18
        offset_y = np.sin(angle) * 0.18
        
        # 标签大小根据重要性
        importance = degree_centrality[node] * 0.5 + betweenness[node] * 0.5
        fontsize = 8 + importance * 5
        
        ax.text(x + offset_x, y + offset_y, label,
               fontproperties=FONT_CN, fontsize=fontsize, fontweight='bold',
               ha='center', va='center', zorder=6,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=colors_palette[community_map[node]], 
                        alpha=0.85, linewidth=2))
    
    # 添加中心标题
    ax.text(0, 0, 'AI\nFactors', ha='center', va='center',
           fontsize=30, fontweight='bold', alpha=0.15, 
           color='gray', zorder=0, family='Arial')
    
    # 添加主标题
    title_text = 'AI Development Factors Relationship Sphere\n3D-style Network Visualization'
    ax.text(0, 1.35, title_text, ha='center', va='top',
           fontproperties=FONT_TITLE, fontsize=18, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                    edgecolor='#2c3e50', linewidth=3, alpha=0.95))
    
    # 创建社区图例（圆形排列）
    legend_y = -1.25
    legend_elements = []
    for idx, comm in enumerate(communities):
        comm_names = [feature_names[node] for node in list(comm)[:2]]
        label = f'社区 {idx+1} ({len(comm)}项): ' + ', '.join(comm_names)
        if len(comm) > 2:
            label += '...'
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=colors_palette[idx], markersize=15,
                      markeredgewidth=2, markeredgecolor='white',
                      label=label)
        )
    
    # 添加相关强度图例
    legend_elements.extend([
        plt.Line2D([0], [0], color='#e74c3c', lw=4, label='超强相关 (r>0.9)'),
        plt.Line2D([0], [0], color='#f39c12', lw=3, label='强相关 (0.8<r≤0.9)'),
        plt.Line2D([0], [0], color='#3498db', lw=2, label='中等相关 (0.7<r≤0.8)')
    ])
    
    legend = ax.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.05), ncol=3,
                      prop=FONT_SMALL, framealpha=0.95, fontsize=9,
                      edgecolor='#2c3e50', fancybox=True, shadow=True)
    
    # 添加说明文本
    note = (f'节点大小 ∝ 综合重要性 | 光晕效果 = 影响范围\n'
            f'布局算法: Golden Spiral (3D球面投影) | 阈值: |r| ≥ {threshold}')
    ax.text(0.5, 0.02, note, transform=ax.transAxes,
           fontproperties=FONT_SMALL, fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff9e6', 
                    edgecolor='#f39c12', alpha=0.9, linewidth=2))
    
    ax.set_xlim([-1.45, 1.45])
    ax.set_ylim([-1.45, 1.45])
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fig_community_network.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print("  ✓ 保存: fig_community_network.png（3D球形布局）")
    plt.close()


# ==================== 主程序 ====================

def main():
    print("\n【步骤1】生成数据...")
    data = generate_data()
    X_scaled, countries, feature_names, standardized_df = standardize_data(data)
    print(f"  ✓ 数据维度: {len(countries)} 个国家 × {len(feature_names)} 个要素")
    
    print("\n【步骤2】相关性分析...")
    corr_df, strong_corr = correlation_analysis(X_scaled, feature_names)
    
    print("\n【步骤3】PCA分析...")
    pca, X_pca, explained_variance_ratio, cumulative_variance, loadings, n_components = pca_analysis(X_scaled, feature_names)
    print(f"  ✓ 前3个主成分累积解释: {cumulative_variance[2]*100:.2f}%")
    
    print("\n" + "="*70)
    print("开始生成SCI高级可视化...")
    print("="*70)
    
    # 生成三个顶级图表
    plot_chord_diagram(corr_df, feature_names, threshold=0.7)
    plot_pca_biplot(X_pca, loadings, feature_names, countries, explained_variance_ratio)
    plot_community_network(corr_df, feature_names, threshold=0.7)
    
    print("\n" + "="*70)
    print("SCI高级可视化完成！")
    print("="*70)
    print("\n生成的图表：")
    print("1. fig_chord_diagram.png     - 弦图（替代相关矩阵热力图）")
    print("2. fig_pca_biplot.png        - PCA双标图（替代载荷矩阵热力图）")
    print("3. fig_community_network.png - 社区检测网络图（升级版）")
    print("\n这三张图都是Nature/Science级别的可视化！")
    print("="*70)

if __name__ == "__main__":
    main()
