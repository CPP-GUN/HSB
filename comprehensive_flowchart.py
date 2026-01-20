"""
综合流程思维导图：问题一到问题四
参考用户提供的流程图风格，创建AI发展能力建模的完整流程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# 定义颜色方案（参考用户图片）
color_problem1 = '#FFB6A3'  # 浅红橙色
color_problem2 = '#FFD599'  # 浅橙黄色
color_problem3 = '#D4E7C5'  # 浅绿色
color_problem4 = '#9DD4CF'  # 浅青色
color_arrow = '#FFD699'     # 箭头颜色

# 定义框的样式参数
box_style = "round,pad=0.1"
main_box_props = dict(boxstyle=box_style, facecolor='white', edgecolor='gray', linewidth=2)

def draw_main_box(ax, x, y, width, height, text, color, fontsize=16):
    """绘制主要流程框"""
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle=box_style, 
                         facecolor=color, 
                         edgecolor='none',
                         alpha=0.9,
                         zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold',
           ha='center', va='center', zorder=3)
    return box

def draw_sub_box(ax, x, y, width, height, text, color, fontsize=11):
    """绘制子流程框"""
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle=box_style,
                         facecolor=color,
                         edgecolor='gray',
                         linewidth=1,
                         alpha=0.7,
                         zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=fontsize,
           ha='center', va='center', zorder=3, linespacing=1.5)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=color_arrow, width=2, style='->'):
    """绘制连接箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color=color,
                           linewidth=width,
                           mutation_scale=30,
                           zorder=1)
    ax.add_patch(arrow)
    return arrow

def draw_double_chevron(ax, x, y, color=color_arrow):
    """绘制双箭头连接符（类似用户图片中的样式）"""
    # 第一个箭头
    chevron1 = mpatches.FancyBboxPatch((x-0.3, y-0.25), 0.5, 0.5,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color,
                                      edgecolor='none',
                                      alpha=0.6,
                                      transform=ax.transData)
    # 第二个箭头
    chevron2 = mpatches.FancyBboxPatch((x+0.1, y-0.25), 0.5, 0.5,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color,
                                      edgecolor='none',
                                      alpha=0.6,
                                      transform=ax.transData)
    ax.add_patch(chevron1)
    ax.add_patch(chevron2)

# ==================== 顶层：4个主要问题 ====================
y_top = 12.5
x_positions = [2.5, 7.5, 12.5, 17.5]
main_titles = ['问题一\n因素识别', '问题二\n综合评估', '问题三\n趋势预测', '问题四\n投资优化']
main_colors = [color_problem1, color_problem2, color_problem3, color_problem4]

for i, (x, title, color) in enumerate(zip(x_positions, main_titles, main_colors)):
    draw_main_box(ax, x, y_top, 2.8, 1.2, title, color, fontsize=18)
    
    # 绘制双箭头连接符
    if i < 3:
        draw_double_chevron(ax, x_positions[i] + 1.4, y_top)

# ==================== 问题一：因素识别与结构分析 ====================
y_p1_start = 10.5
x_p1 = 2.5

# 主标题
ax.text(x_p1, y_p1_start + 0.3, '因素识别与结构分析', fontsize=14, fontweight='bold',
       ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=color_problem1, alpha=0.3))

# 子任务
sub_tasks_p1 = [
    '数据收集\n24指标×10国家',
    '数据标准化\nMin-Max归一化',
    '相关性分析\nPearson相关矩阵',
    '聚类分析\n层次聚类',
    'PCA降维\n主成分分析',
    '因素重要性\n载荷×方差贡献'
]

y_start_p1 = 8.8
for i, task in enumerate(sub_tasks_p1):
    y_pos = y_start_p1 - i * 0.9
    draw_sub_box(ax, x_p1, y_pos, 2.5, 0.7, task, color_problem1, fontsize=10)
    if i > 0:
        draw_arrow(ax, x_p1, y_start_p1 - (i-1)*0.9 - 0.35, 
                  x_p1, y_pos + 0.35, color='gray', width=1.5)

# 输出结果框
draw_sub_box(ax, x_p1, 2.5, 2.5, 0.8, 
            '输出：指标体系\n强相关结构\n关键因素', 
            color_problem1, fontsize=10)
draw_arrow(ax, x_p1, y_start_p1 - 5*0.9 - 0.35, x_p1, 2.9, color='gray', width=1.5)

# ==================== 问题二：综合评估与排序 ====================
y_p2_start = 10.5
x_p2 = 7.5

# 主标题
ax.text(x_p2, y_p2_start + 0.3, '综合评估与2025排序', fontsize=14, fontweight='bold',
       ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=color_problem2, alpha=0.3))

# 子任务
sub_tasks_p2 = [
    '熵权法（EWM）\n客观赋权',
    'TOPSIS评分\n正负理想解',
    '灰色关联分析\n结构相似性验证',
    '排序融合\n等权合成得分',
    '稳健性检验\nSpearman秩相关',
    '敏感性分析\n权重扰动测试'
]

y_start_p2 = 8.8
for i, task in enumerate(sub_tasks_p2):
    y_pos = y_start_p2 - i * 0.9
    draw_sub_box(ax, x_p2, y_pos, 2.5, 0.7, task, color_problem2, fontsize=10)
    if i > 0:
        draw_arrow(ax, x_p2, y_start_p2 - (i-1)*0.9 - 0.35, 
                  x_p2, y_pos + 0.35, color='gray', width=1.5)

# 输出结果框
draw_sub_box(ax, x_p2, 2.5, 2.5, 0.8, 
            '输出：2025排名\n美国>中国>印度\n权重向量', 
            color_problem2, fontsize=10)
draw_arrow(ax, x_p2, y_start_p2 - 5*0.9 - 0.35, x_p2, 2.9, color='gray', width=1.5)

# ==================== 问题三：趋势预测 ====================
y_p3_start = 10.5
x_p3 = 12.5

# 主标题
ax.text(x_p3, y_p3_start + 0.3, '趋势预测（2026-2035）', fontsize=14, fontweight='bold',
       ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=color_problem3, alpha=0.3))

# 子任务
sub_tasks_p3 = [
    '历史数据整理\n2016-2025面板',
    'GM(1,1)预测\n灰色模型',
    '24指标逐年外推\n2026-2035',
    '继承Task2权重\nTOPSIS评分',
    '排名动态追踪\n10国×10年',
    '回测验证\nMAPE=10.35%'
]

y_start_p3 = 8.8
for i, task in enumerate(sub_tasks_p3):
    y_pos = y_start_p3 - i * 0.9
    draw_sub_box(ax, x_p3, y_pos, 2.5, 0.7, task, color_problem3, fontsize=10)
    if i > 0:
        draw_arrow(ax, x_p3, y_start_p3 - (i-1)*0.9 - 0.35, 
                  x_p3, y_pos + 0.35, color='gray', width=1.5)

# 输出结果框
draw_sub_box(ax, x_p3, 2.5, 2.5, 0.8, 
            '输出：2026-2035\n逐年排名\n竞争力演化', 
            color_problem3, fontsize=10)
draw_arrow(ax, x_p3, y_start_p3 - 5*0.9 - 0.35, x_p3, 2.9, color='gray', width=1.5)

# ==================== 问题四：投资优化 ====================
y_p4_start = 10.5
x_p4 = 17.5

# 主标题
ax.text(x_p4, y_p4_start + 0.3, '投资优化（中国专项）', fontsize=14, fontweight='bold',
       ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=color_problem4, alpha=0.3))

# 子任务
sub_tasks_p4 = [
    '预算设定\n1万亿元（10年）',
    '响应函数建模\n边际递减+时滞',
    '约束构造\n上下限+协同',
    'SLSQP优化\n最大化TOPSIS',
    '灵敏度分析\n±10%扰动',
    '策略建议\n基础设施优先'
]

y_start_p4 = 8.8
for i, task in enumerate(sub_tasks_p4):
    y_pos = y_start_p4 - i * 0.9
    draw_sub_box(ax, x_p4, y_pos, 2.5, 0.7, task, color_problem4, fontsize=10)
    if i > 0:
        draw_arrow(ax, x_p4, y_start_p4 - (i-1)*0.9 - 0.35, 
                  x_p4, y_pos + 0.35, color='gray', width=1.5)

# 输出结果框
draw_sub_box(ax, x_p4, 2.5, 2.5, 0.8, 
            '输出：最优配置\n基建32.33%\n政策建议', 
            color_problem4, fontsize=10)
draw_arrow(ax, x_p4, y_start_p4 - 5*0.9 - 0.35, x_p4, 2.9, color='gray', width=1.5)

# ==================== 底部：横向连接箭头 ====================
y_bottom = 1.5
for i in range(3):
    draw_arrow(ax, x_positions[i] + 1.4, 2.5, 
              x_positions[i+1] - 1.4, 2.5, 
              color=color_arrow, width=2.5, style='->')

# ==================== 添加整体标题 ====================
ax.text(10, 13.5, 'AI发展能力建模完整流程：从因素识别到投资优化',
       fontsize=22, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

# ==================== 添加数据流标注 ====================
# 问题1→问题2
ax.annotate('指标体系\n权重基础', xy=(5, 10), fontsize=9, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# 问题2→问题3
ax.annotate('权重向量\n评估口径', xy=(10, 10), fontsize=9, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# 问题3→问题4
ax.annotate('基准水平\n预测情景', xy=(15, 10), fontsize=9, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# ==================== 添加关键方法标注 ====================
methods_y = 0.5
ax.text(2.5, methods_y, '方法：PCA+聚类', fontsize=9, ha='center', style='italic',
       color='darkred')
ax.text(7.5, methods_y, '方法：EWM+TOPSIS+GRA', fontsize=9, ha='center', style='italic',
       color='darkorange')
ax.text(12.5, methods_y, '方法：GM(1,1)+回测', fontsize=9, ha='center', style='italic',
       color='darkgreen')
ax.text(17.5, methods_y, '方法：SLSQP+灵敏度', fontsize=9, ha='center', style='italic',
       color='darkblue')

plt.tight_layout()
plt.savefig('figure/comprehensive_flowchart.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.savefig('figure/comprehensive_flowchart.pdf', bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print("✅ 综合流程图已保存：")
print("   - figure/comprehensive_flowchart.png")
print("   - figure/comprehensive_flowchart.pdf")
plt.show()
