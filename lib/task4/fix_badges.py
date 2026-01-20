# -*- coding: utf-8 -*-
"""临时脚本：将emoji徽章替换为纯数字"""

with open('plot_task4_figures.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换所有emoji为数字
replacements = [
    ('badge_text = "🥇"', 'badge_text = "1"'),
    ('badge_text = "🥈"', 'badge_text = "2"'),
    ('badge_text = "🥉"', 'badge_text = "3"'),
    ('badge_text = f"#{rank}"', 'badge_text = str(rank)'),
    ('badge_text = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"', 'badge_text = str(rank)'),
]

for old, new in replacements:
    content = content.replace(old, new)

with open('plot_task4_figures.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✓ 已将所有emoji替换为纯数字 1、2、3...')
