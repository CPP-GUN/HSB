"""
简化版总体框架图 - 清爽流程风格
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# 配色
c_data = '#FFB6A3'
c_p1 = '#FFD599'
c_p2 = '#FFE6B3'
c_p3 = '#D4E7C5'
c_p4 = '#9DD4CF'
c_valid = '#FFF4E6'
c_output = '#E6D8F5'

def box(ax, x, y, w, h, text, color, fs=10, bold=False):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                       facecolor=color, edgecolor='gray', linewidth=1.2, alpha=0.9)
    ax.add_patch(b)
    ax.text(x, y, text, fontsize=fs, ha='center', va='center', 
           fontweight='bold' if bold else 'normal', linespacing=1.5)

def arrow(ax, x1, y1, x2, y2, c='gray', w=1.8):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                       color=c, linewidth=w, mutation_scale=20)
    ax.add_patch(a)

# 标题
ax.text(8, 9.3, 'Overall Framework', fontsize=18, ha='center', fontweight='bold')

# 数据层
box(ax, 8, 8.3, 5, 0.8, 'Data: 24 Indicators × 10 Countries (2016-2025)', c_data, 11, True)

# 四个问题（横向排列）
y_prob = 6.5
box(ax, 2.5, y_prob, 2.8, 1.4, 
    'Problem 1\nFactor Identification\n\nPearson + PCA', c_p1, 9.5, True)
box(ax, 6, y_prob, 2.8, 1.4,
    'Problem 2\nEvaluation & Ranking\n\nEWM + TOPSIS + GRA', c_p2, 9.5, True)
box(ax, 9.5, y_prob, 2.8, 1.4,
    'Problem 3\nTrend Forecasting\n\nGM(1,1) + Backtest', c_p3, 9.5, True)
box(ax, 13, y_prob, 2.8, 1.4,
    'Problem 4\nInvestment Optimization\n\nSLSQP + Sensitivity', c_p4, 9.5, True)

# 箭头：数据到问题
for x in [2.5, 6, 9.5, 13]:
    arrow(ax, 8, 7.9, x, 7.2, 'darkgray', 1.5)

# 箭头：问题间传递
arrow(ax, 3.9, 6.5, 4.6, 6.5, 'orange', 2)
ax.text(4.25, 6.9, 'Indicators', fontsize=8, ha='center', style='italic')

arrow(ax, 7.4, 6.5, 8.1, 6.5, 'orange', 2)
ax.text(7.75, 6.9, 'Weights', fontsize=8, ha='center', style='italic')

arrow(ax, 10.9, 6.5, 11.6, 6.5, 'orange', 2)
ax.text(11.25, 6.9, 'Baseline', fontsize=8, ha='center', style='italic')

# 验证层
box(ax, 8, 4.8, 13, 0.7,
    'Multi-Layer Validation: Correlation | Cross-Method | Backtest | Sensitivity',
    c_valid, 9.5, True)

for x in [2.5, 6, 9.5, 13]:
    arrow(ax, x, 5.8, 8, 5.2, 'gray', 1)

# 输出层
y_out = 3.2
box(ax, 2.5, y_out, 2.6, 1, 'Key Drivers\n& Structure', c_p1, 9.5)
box(ax, 6, y_out, 2.6, 1, '2025 Ranking\nUS > CN > IN', c_p2, 9.5)
box(ax, 9.5, y_out, 2.6, 1, '2026-2035\nEvolution', c_p3, 9.5)
box(ax, 13, y_out, 2.6, 1, 'Optimal Allocation\nInfra: 32.33%', c_p4, 9.5)

for x in [2.5, 6, 9.5, 13]:
    arrow(ax, x, 4.4, x, 3.7, 'gray', 1.8)

# 决策层
box(ax, 8, 1.5, 12, 0.8,
    'Policy Recommendations: Structural + Dynamic + Coordination',
    c_output, 10, True)

for x in [2.5, 6, 9.5, 13]:
    arrow(ax, x, 2.7, 8, 1.9, 'purple', 1.5)

# 侧边标注
labels = [
    ('Data Layer', 8.3, 'blue'),
    ('Modeling Layer', 6.5, 'darkgreen'),
    ('Validation Layer', 4.8, 'darkred'),
    ('Output Layer', 3.2, 'darkorange'),
    ('Decision Layer', 1.5, 'purple')
]

for label, y, color in labels:
    ax.text(0.3, y, label, fontsize=9, ha='center', va='center',
           fontweight='bold', color=color, rotation=90,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

plt.tight_layout()
plt.savefig('figure/overall_framework.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figure/overall_framework.pdf', bbox_inches='tight', facecolor='white')
print("✅ Framework diagram saved:")
print("   - figure/overall_framework.png")
print("   - figure/overall_framework.pdf")
plt.show()
