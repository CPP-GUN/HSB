import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import pandas as pd
import io
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==========================================
# 1. 样式设置 (MCM/ICM Style)
# ==========================================
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
except:
    plt.rcParams['font.family'] = 'serif'

plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 数据准备
# ==========================================

# 翻译映射
translation_map = {
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
    "国际AI投资": "International AI Investment"
}

# 原始数据
data_csv = """Indicator_1,Indicator_2,Correlation
AI研究人员数量,顶尖AI学者数量,0.8823427036457958
AI研究人员数量,AI企业数量,0.7210575394952075
AI研究人员数量,AI市场规模,0.9502487610427175
AI研究人员数量,AI政策数量,0.9358991034301386
AI研究人员数量,企业研发支出,0.9627204768475273
AI研究人员数量,政府AI投资,0.7910621291120972
AI研究人员数量,GPU集群规模,0.7126442344618326
AI研究人员数量,互联网带宽,0.8268635210950556
AI研究人员数量,电能生产,0.821346339104829
AI研究人员数量,AI算力平台,0.8005661085180261
AI研究人员数量,数据中心数量,0.9486372045911858
AI研究人员数量,AI_Book数量,0.9136957330501907
AI研究人员数量,国际AI投资,0.9496204791807259
顶尖AI学者数量,AI企业数量,0.7016703924083675
顶尖AI学者数量,AI市场规模,0.9739327279062311
顶尖AI学者数量,AI政策数量,0.9846718147450059
顶尖AI学者数量,企业研发支出,0.9025605487087762
顶尖AI学者数量,GPU集群规模,0.9187035707893282
顶尖AI学者数量,互联网带宽,0.8197164668333712
顶尖AI学者数量,数据中心数量,0.9614347983518761
顶尖AI学者数量,AI_Book数量,0.7811901977258704
顶尖AI学者数量,国际AI投资,0.9364527243120425
AI毕业生数量,大模型数量,0.7411620936382153
AI毕业生数量,AI社会信任度,0.7247455461789228
AI毕业生数量,互联网普及率,-0.8302840801594242
AI毕业生数量,电能生产,0.7977022569106973
AI企业数量,AI市场规模,0.7564622709975999
AI企业数量,AI政策数量,0.7508552603674824
AI企业数量,企业研发支出,0.7975065626552552
AI企业数量,数据中心数量,0.7273970098825222
AI企业数量,国际AI投资,0.7962022447791993
AI市场规模,AI政策数量,0.9975824315868032
AI市场规模,企业研发支出,0.9737109307561689
AI市场规模,GPU集群规模,0.8500781339941923
AI市场规模,互联网带宽,0.8862891153845759
AI市场规模,AI算力平台,0.7011993565888347
AI市场规模,数据中心数量,0.993943861890911
AI市场规模,AI_Book数量,0.8662932226986013
AI市场规模,国际AI投资,0.9858169198298976
AI应用渗透率,AI社会信任度,0.8495045518932257
大模型数量,政府AI投资,0.9548151433046399
大模型数量,电能生产,0.9717403130026542
大模型数量,AI算力平台,0.9729726644348103
大模型数量,TOP500上榜数,0.8570276699009526
大模型数量,AI_Book数量,0.7193256879646076
AI政策数量,企业研发支出,0.9594120642406887
AI政策数量,GPU集群规模,0.867551727992665
AI政策数量,互联网带宽,0.8720113652565704
AI政策数量,数据中心数量,0.9902373840950639
AI政策数量,AI_Book数量,0.84890576362847
AI政策数量,国际AI投资,0.9808800314546847
企业研发支出,政府AI投资,0.8050679676995817
企业研发支出,GPU集群规模,0.7416359866996695
企业研发支出,互联网带宽,0.9126903673769253
企业研发支出,电能生产,0.7962776828630238
企业研发支出,AI算力平台,0.8255131102486152
企业研发支出,数据中心数量,0.9721769765109427
企业研发支出,AI_Book数量,0.9035107448495477
企业研发支出,国际AI投资,0.9759875061366251
政府AI投资,电能生产,0.9492256860186767
政府AI投资,AI算力平台,0.9869248074094276
政府AI投资,TOP500上榜数,0.7084760161131655
政府AI投资,AI_Book数量,0.7544546512718733
政府AI投资,国际AI投资,0.7409602888517896
5G覆盖率,互联网普及率,0.8162577334058716
5G覆盖率,GitHub项目数,-0.8199179467460079
GPU集群规模,数据中心数量,0.806426980619134
GPU集群规模,国际AI投资,0.7911348278308001
互联网带宽,AI算力平台,0.7172981592923175
互联网带宽,数据中心数量,0.9144650396515425
互联网带宽,AI_Book数量,0.8676763810967804
互联网带宽,AI_Dataset数量,0.8120183989760184
互联网带宽,国际AI投资,0.8942574653066888
互联网普及率,GitHub项目数,-0.8096738877910991
电能生产,AI算力平台,0.9688275176760034
电能生产,TOP500上榜数,0.7601028876931324
电能生产,AI_Book数量,0.8067742520460962
电能生产,国际AI投资,0.7176572833507286
AI算力平台,数据中心数量,0.7165652081076971
AI算力平台,TOP500上榜数,0.7728288270064941
AI算力平台,AI_Book数量,0.8066338896141718
AI算力平台,国际AI投资,0.7524956262311687
数据中心数量,AI_Book数量,0.8973093649478813
数据中心数量,AI_Dataset数量,0.7203399408606377
数据中心数量,国际AI投资,0.9829460694690182
AI_Book数量,国际AI投资,0.8642423628307754
"""

# 加载数据
df = pd.read_csv(io.StringIO(data_csv))

# 翻译指标名称
df['Indicator_1'] = df['Indicator_1'].map(
    translation_map).fillna(df['Indicator_1'])
df['Indicator_2'] = df['Indicator_2'].map(
    translation_map).fillna(df['Indicator_2'])

# 获取所有唯一的指标节点
nodes = list(set(df['Indicator_1']).union(set(df['Indicator_2'])))
nodes.sort()  # 排序以保持一致性
num_nodes = len(nodes)

# 计算节点的“权重”（关联强度的总和），用于调整节点大小
node_weights = {node: 0.0 for node in nodes}
for idx, row in df.iterrows():
    w = abs(row['Correlation'])
    node_weights[row['Indicator_1']] += w
    node_weights[row['Indicator_2']] += w

# 归一化权重以控制节点大小范围
max_weight = max(node_weights.values()) if node_weights else 1
min_weight = min(node_weights.values()) if node_weights else 0


def get_node_size(weight):
    # 线性映射到 50 - 200 之间
    if max_weight == min_weight:
        return 100
    return 50 + (weight - min_weight) / (max_weight - min_weight) * 150


# ==========================================
# 3. 绘图逻辑 (Circular Layout with Bezier Curves)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
ax.axis('off')

# 定义圆的参数
radius = 1.0
label_radius = 1.15  # 稍微增加标签距离

# 计算每个节点的角度
angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
node_angles = dict(zip(nodes, angles))

# 颜色映射 (RdBu_r: 红色为正，蓝色为负，白色为0)
cmap = plt.get_cmap('RdBu_r')
norm = mcolors.Normalize(vmin=-1, vmax=1)

# 添加外围圆环 (装饰)
circle = plt.Circle((0, 0), radius, color='#d9d9d9',
                    fill=False, linewidth=1.5, zorder=0, alpha=0.5)
ax.add_artist(circle)

# --- 1. 绘制连线 (Chords) ---
# 先对 dataframe 按相关系数绝对值排序，让强相关的线画在上面
df['abs_corr'] = df['Correlation'].abs()
df = df.sort_values(by='abs_corr')

for idx, row in df.iterrows():
    u, v, corr = row['Indicator_1'], row['Indicator_2'], row['Correlation']

    # 获取起止角度
    angle1 = node_angles[u]
    angle2 = node_angles[v]

    # 极坐标转笛卡尔坐标
    x1, y1 = radius * np.cos(angle1), radius * np.sin(angle1)
    x2, y2 = radius * np.cos(angle2), radius * np.sin(angle2)

    # 贝塞尔曲线控制点 (圆心)
    verts = [
        (x1, y1),  # P0
        (0, 0),    # P1 (Control point at center)
        (x2, y2)   # P2
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)

    # 设置样式
    color = cmap(norm(corr))

    # 优化线宽和透明度
    abs_corr = abs(corr)
    linewidth = pow(abs_corr, 2) * 4.0  # 稍微加粗
    alpha = 0.2 + pow(abs_corr, 3) * 0.7  # 基础透明度0.2，最大0.9

    patch = patches.PathPatch(
        path, facecolor='none', edgecolor=color, lw=linewidth, alpha=alpha, zorder=1)
    ax.add_patch(patch)

# --- 2. 绘制节点和标签 ---
for node, angle in node_angles.items():
    x, y = radius * np.cos(angle), radius * np.sin(angle)

    # 绘制节点点 (大小基于权重)
    size = get_node_size(node_weights[node])

    # 节点颜色
    ax.scatter(x, y, color='#404040', s=size,
               edgecolors='white', linewidth=1.5, zorder=10)

    # 绘制标签
    degrees = np.degrees(angle)

    # 调整文字位置和旋转
    if 90 < degrees < 270:
        rotation = degrees + 180
        ha = 'right'
        lx, ly = label_radius * np.cos(angle), label_radius * np.sin(angle)
    else:
        rotation = degrees
        ha = 'left'
        lx, ly = label_radius * np.cos(angle), label_radius * np.sin(angle)

    ax.text(lx, ly, node, rotation=rotation, ha=ha, va='center',
            fontsize=11, family='serif', fontweight='medium')

# --- 3. 添加图例 (Colorbar) ---
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# 修改位置参数：增加 pad (距离)，减小 shrink (长度)，减小 fraction (宽度)
cbar = plt.colorbar(sm, ax=ax, orientation='vertical',
                    fraction=0.025, pad=0.15, shrink=0.5)

cbar.set_label('Correlation Coefficient', fontsize=12, labelpad=10)
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(labelsize=10)

# ==========================================
# 4. 保存
# ==========================================
output_filename = 'Correlation_Chord_Diagram.pdf'
# 增加 pad_inches 确保边缘不被裁剪
plt.savefig(output_filename, format='pdf', bbox_inches='tight', pad_inches=0.8)

print(f"成功生成文件: {output_filename}")
