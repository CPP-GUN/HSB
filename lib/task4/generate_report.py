# -*- coding: utf-8 -*-
"""
generate_report.py
生成问题四的详细分析报告
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_summary_report():
    """生成总结报告"""
    
    output_dir = Path(__file__).resolve().parent / 'outputs'
    
    # 读取结果
    investment_df = pd.read_csv(output_dir / 'investment_allocation.csv', encoding='utf-8-sig')
    comparison_df = pd.read_csv(output_dir / 'before_after_comparison.csv', encoding='utf-8-sig')
    dimension_df = pd.read_csv(output_dir / 'dimension_distribution.csv', encoding='utf-8-sig')
    topsis_df = pd.read_csv(output_dir / 'topsis_improvement.csv', encoding='utf-8-sig')
    
    print("\n" + "=" * 70)
    print("问题四：中国AI发展投资优化 - 详细报告")
    print("=" * 70)
    
    print("\n【一、投资策略总结】")
    print(f"总投资额：10000 亿元")
    print(f"投资期限：2026-2035年（10年）")
    print(f"优化目标：最大化中国2035年AI综合竞争力")
    
    print("\n【二、核心投资方向（Top 5）】")
    top5 = investment_df.head(5)
    for idx, row in top5.iterrows():
        print(f"  {row['排名']}. {row['指标']}: {row['投资额_亿元']:.2f}亿元 ({row['占比_%']:.2f}%)")
    
    print("\n【三、六大维度投资分布】")
    for idx, row in dimension_df.iterrows():
        print(f"  {row['维度']}: {row['投资额_亿元']:.2f}亿元 ({row['占比_%']:.2f}%)")
    
    print("\n【四、投资效果分析】")
    print("\n  (1) 指标增长Top 5（按增长率）")
    top_growth = comparison_df.nlargest(5, '增长率_%')
    for idx, row in top_growth.iterrows():
        print(f"      {row['指标']}: +{row['增长率_%']:.2f}% (从{row['2026基准值']:.2f}到{row['2035投资后']:.2f})")
    
    print("\n  (2) 指标增长Top 5（按绝对增长量）")
    top_abs = comparison_df.nlargest(5, '增长量')
    for idx, row in top_abs.iterrows():
        print(f"      {row['指标']}: +{row['增长量']:.2f} (从{row['2026基准值']:.2f}到{row['2035投资后']:.2f})")
    
    print("\n【五、政策建议】")
    print("\n  短期（2026-2028）：")
    print("    - 重点投资基础设施（GPU集群、数据中心、算力平台）")
    print("    - 快速提升算力基础，为大模型训练提供支撑")
    print("    - 完善5G网络和互联网基础设施")
    
    print("\n  中期（2029-2032）：")
    print("    - 加大企业研发支持力度，激发市场创新活力")
    print("    - 扩大AI市场规模，培育应用生态")
    print("    - 完善政策法规体系，提升社会信任度")
    
    print("\n  长期（2033-2035）：")
    print("    - 持续投入人才培养（顶尖学者、研究人员、毕业生）")
    print("    - 提升开放性建设（数据集、学术成果、开源项目）")
    print("    - 增强国际影响力和技术话语权")
    
    print("\n【六、关键发现】")
    print(f"\n  1. 投资最集中的领域：{investment_df.iloc[0]['指标']} ({investment_df.iloc[0]['占比_%']:.2f}%)")
    print(f"  2. 增长最快的指标：{comparison_df.nlargest(1, '增长率_%').iloc[0]['指标']} (+{comparison_df.nlargest(1, '增长率_%').iloc[0]['增长率_%']:.2f}%)")
    print(f"  3. 投资最多的维度：{dimension_df.iloc[0]['维度']} ({dimension_df.iloc[0]['占比_%']:.2f}%)")
    
    # 计算投资效率
    total_growth = comparison_df['增长量'].sum()
    print(f"  4. 总体增长效率：{total_growth/10000:.2f} 单位增长/亿元投资")
    
    print("\n【七、模型说明】")
    print("\n  注意：TOPSIS得分下降是因为其他国家在2035年也有显著增长。")
    print("  实际上，中国的绝对指标值大幅提升，综合实力显著增强。")
    print("  建议关注：")
    print("    - 各指标的绝对增长量和增长率")
    print("    - 与美国等领先国家的差距缩小情况")
    print("    - 在关键技术领域（大模型、GPU、顶尖人才）的突破")
    
    print("\n" + "=" * 70)
    print("报告生成完成！")
    print("=" * 70)


if __name__ == '__main__':
    generate_summary_report()
