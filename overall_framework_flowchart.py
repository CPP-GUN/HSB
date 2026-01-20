"""
总体工作框架思维导图 - 美赛风格
展示从数据到决策的完整闭环
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# 定义配色方案
color_data = '#FFB6A3'        # 数据层-浅红橙
color_problem1 = '#FFE6B3'    # 问题1-浅黄
color_problem2 = '#FFD599'    # 问题2-橙黄
color_problem3 = '#D4E7C5'    # 问题3-浅绿
color_problem4 = '#9DD4CF'    # 问题4-浅青
color_output = '#B8B8D1'      # 输出-浅紫

def draw_box(ax, x, y, w, h, text, color, fontsize=11, fontweight='normal'):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                         boxstyle="round,pad=0.1",
                         facecolor=color,
                         edgecolor='gray',
                         linewidth=1.5,
                         alpha=0.85)
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight,
           ha='center', va='center', linespacing=1.6)

def draw_arrow(ax, x1, y1, x2, y2, color='gray', width=2, style='->', label=''):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color=color,
                           linewidth=width,
                           mutation_scale=25)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x, mid_y+0.3, label, fontsize=9, ha='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

def draw_circle_node(ax, x, y, r, text, color, fontsize=10):
    """绘制圆形节点"""
    circle = Circle((x, y), r, facecolor=color, edgecolor='gray', linewidth=1.5, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold', ha='center', va='center')

# ==================== 标题 ====================
ax.text(9, 11.2, 'Overall Framework: From Data to Decision', 
       fontsize=20, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

# ==================== 数据层（顶部） ====================
draw_box(ax, 3, 9.5, 4, 1.2, 
        'Data Collection & Processing\n24 Indicators × 10 Countries × 10 Years',
        color_data, fontsize=11, fontweight='bold')

# 数据详情
data_details = [
    'Talent (3)', 'Application (4)', 'Policy (3)',
    'Investment (3)', 'Infrastructure (6)', 'Output (5)'
]
x_start = 0.5
for i, detail in enumerate(data_details):
    draw_box(ax, x_start + i*1.05, 8.2, 0.95, 0.5, detail, color_data, fontsize=8)

draw_arrow(ax, 3, 8.9, 3, 8.5, color='gray', width=2)

# ==================== 四个问题（中间层） ====================
y_problems = 6

# Problem 1
draw_circle_node(ax, 2, y_problems, 0.5, 'P1', color_problem1, fontsize=12)
draw_box(ax, 2, 5, 2.5, 1.8,
        'Factor Identification\n\n• Pearson Correlation\n  (86 pairs, |r|>0.7)\n• PCA Decomposition\n  (4 PCs, 85% variance)',
        color_problem1, fontsize=9)

# Problem 2  
draw_circle_node(ax, 6, y_problems, 0.5, 'P2', color_problem2, fontsize=12)
draw_box(ax, 6, 5, 2.5, 1.8,
        'Comprehensive Evaluation\n\n• EWM Weighting\n• TOPSIS Scoring\n• GRA Validation',
        color_problem2, fontsize=9)

# Problem 3
draw_circle_node(ax, 10, y_problems, 0.5, 'P3', color_problem3, fontsize=12)
draw_box(ax, 10, 5, 2.5, 1.8,
        'Trend Forecasting\n\n• GM(1,1) Model\n• 2026-2035 Projection\n• Backtest (MAPE 10.35%)',
        color_problem3, fontsize=9)

# Problem 4
draw_circle_node(ax, 14, y_problems, 0.5, 'P4', color_problem4, fontsize=12)
draw_box(ax, 14, 5, 2.5, 1.8,
        'Investment Optimization\n\n• CNY 1T Budget\n• SLSQP Algorithm\n• Sensitivity Analysis',
        color_problem4, fontsize=9)

# 问题间连接
draw_arrow(ax, 3.3, 6, 5.2, 6, color='orange', width=2.5, label='Indicators')
draw_arrow(ax, 7.3, 6, 9.2, 6, color='orange', width=2.5, label='Weights')
draw_arrow(ax, 11.3, 6, 13.2, 6, color='orange', width=2.5, label='Baseline')

# 从数据层到四个问题
draw_arrow(ax, 2.5, 8.2, 2, 6.5, color='gray', width=1.5)
draw_arrow(ax, 3, 8.2, 6, 6.5, color='gray', width=1.5)
draw_arrow(ax, 3.5, 8.2, 10, 6.5, color='gray', width=1.5)
draw_arrow(ax, 4, 8.2, 14, 6.5, color='gray', width=1.5)

# ==================== 验证层（中间） ====================
y_validation = 3
draw_box(ax, 9, y_validation, 14, 0.8,
        'Multi-Layer Validation: Correlation Analysis | Cross-Method Comparison | Backtest Diagnostics | Sensitivity Test',
        'lightyellow', fontsize=9, fontweight='bold')

# 从四个问题到验证层
for x in [2, 6, 10, 14]:
    draw_arrow(ax, x, 4.1, 9, 3.4, color='gray', width=1, style='-')

# ==================== 输出层（底部） ====================
y_output = 1.5

outputs = [
    ('Key Drivers\nStructure', 2, color_problem1),
    ('2025 Ranking\nUS>CN>IN', 6, color_problem2),
    ('2035 Forecast\nEvolution', 10, color_problem3),
    ('Optimal Allocation\nInfra 32.33%', 14, color_problem4)
]

for text, x, color in outputs:
    draw_box(ax, x, y_output, 2.2, 1, text, color, fontsize=9, fontweight='bold')
    draw_arrow(ax, x, 2.6, x, 2.1, color='gray', width=2)

# ==================== 决策支持（最底部） ====================
draw_box(ax, 9, 0.3, 12, 0.5,
        'Policy Recommendations: Structural Priorities | Dynamic Windows | Coordination Constraints',
        color_output, fontsize=10, fontweight='bold')

for x in [2, 6, 10, 14]:
    draw_arrow(ax, x, 1, 9, 0.6, color='purple', width=1.5, style='->')

# ==================== 侧边标注 ====================
# 左侧阶段标注
stages = [
    ('Data\nLayer', 9.5),
    ('Modeling\nLayer', 5.5),
    ('Validation\nLayer', 3),
    ('Output\nLayer', 1.5),
    ('Decision\nLayer', 0.3)
]

for stage, y in stages:
    ax.text(0.3, y, stage, fontsize=9, ha='center', va='center',
           rotation=0, fontweight='bold', color='darkblue',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

# 右侧创新点标注
innovations = [
    'Unified 24-indicator system',
    'Multi-model cross-validation',
    'Closed-loop framework',
    'Reproducible & transparent'
]

for i, innov in enumerate(innovations):
    ax.text(17.2, 8.5-i*0.8, f'• {innov}', fontsize=8, ha='left', 
           style='italic', color='darkgreen')

ax.text(17.2, 9.3, 'Core Innovations:', fontsize=9, ha='left', fontweight='bold')

plt.tight_layout()
plt.savefig('figure/overall_framework.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figure/overall_framework.pdf', bbox_inches='tight', facecolor='white')
print("✅ 总体框架图已生成：")
print("   - figure/overall_framework.png")
print("   - figure/overall_framework.pdf")
plt.show()
