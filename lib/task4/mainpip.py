# -*- coding: utf-8 -*-
"""
mainpip.py
问题四：主入口程序
运行方式：python mainpip.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_all_data
from optimization_model import OptimizationModel, calculate_investment_effect, topsis_score
from parameters import INDICATORS, TOTAL_BUDGET, UNIT_COSTS, TIME_DISCOUNTS
import parameters


def ensure_outputs_dir():
    """确保输出目录存在"""
    output_dir = Path(__file__).resolve().parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def analyze_results(result, data):
    """
    分析优化结果
    """
    print("\n" + "=" * 70)
    print("结果分析")
    print("=" * 70)
    
    optimal_investment = result.x
    
    # 1. 投资分配方案
    investment_df = pd.DataFrame({
        '指标': INDICATORS,
        '投资额_亿元': optimal_investment,
        '占比_%': (optimal_investment / TOTAL_BUDGET) * 100
    })
    investment_df = investment_df.sort_values('投资额_亿元', ascending=False).reset_index(drop=True)
    investment_df.insert(0, '排名', range(1, len(investment_df) + 1))
    
    print("\n【投资分配方案 Top 10】")
    print(investment_df.head(10).to_string(index=False))
    
    # 2. 计算投资前后对比
    baseline = data['china_baseline_2026']
    upper_limits = data['upper_limits_array']
    unit_costs = np.array([UNIT_COSTS[ind] for ind in INDICATORS])
    time_discounts = np.array([TIME_DISCOUNTS[ind] for ind in INDICATORS])
    
    delta_x = calculate_investment_effect(
        optimal_investment, baseline, upper_limits, unit_costs, time_discounts
    )
    china_2035_after = np.minimum(baseline + delta_x, upper_limits)
    
    comparison_df = pd.DataFrame({
        '指标': INDICATORS,
        '2026基准值': baseline,
        '2035投资后': china_2035_after,
        '增长量': delta_x,
        '增长率_%': (delta_x / (baseline + 1e-6)) * 100
    })
    comparison_df = comparison_df.sort_values('增长率_%', ascending=False).reset_index(drop=True)
    
    print("\n【投资前后对比 Top 10（按增长率）】")
    print(comparison_df.head(10).to_string(index=False))
    
    # 3. TOPSIS得分提升
    # 投资前
    data_matrix_before = data['all_countries_2035'].values.copy()
    china_idx = list(data['all_countries_2035'].index).index('中国')
    scores_before = topsis_score(data_matrix_before, data['weights'])
    china_score_before = scores_before[china_idx]
    
    # 投资后
    data_matrix_after = data_matrix_before.copy()
    data_matrix_after[china_idx, :] = china_2035_after
    scores_after = topsis_score(data_matrix_after, data['weights'])
    china_score_after = scores_after[china_idx]
    
    improvement = china_score_after - china_score_before
    improvement_pct = (improvement / china_score_before) * 100
    
    print("\n【TOPSIS得分提升】")
    print(f"  投资前（2026基准）：{china_score_before:.6f}")
    print(f"  投资后（2035）：    {china_score_after:.6f}")
    print(f"  提升幅度：          +{improvement:.6f} (+{improvement_pct:.2f}%)")
    print(f"  投资效率：          {improvement/TOTAL_BUDGET:.2e} 得分/亿元")
    
    # 4. 六大维度投资分布
    dimensions = {
        'T_人才': ['AI研究人员数量', '顶尖AI学者数量', 'AI毕业生数量'],
        'A_应用': ['AI企业数量', 'AI市场规模', 'AI应用渗透率', '大模型数量'],
        'P_政策': ['AI社会信任度', 'AI政策数量', 'AI补贴金额'],
        'R_研发': ['企业研发支出', '政府AI投资', '国际AI投资'],
        'I_基础设施': ['5G覆盖率', 'GPU集群规模', '互联网带宽', '互联网普及率',
                    '电能生产', 'AI算力平台', '数据中心数量', 'TOP500上榜数'],
        'O_产出': ['AI_Book数量', 'AI_Dataset数量', 'GitHub项目数']
    }
    
    dimension_investment = {}
    for dim_name, indicators in dimensions.items():
        total = sum(optimal_investment[INDICATORS.index(ind)] for ind in indicators)
        dimension_investment[dim_name] = total
    
    dimension_df = pd.DataFrame({
        '维度': list(dimension_investment.keys()),
        '投资额_亿元': list(dimension_investment.values()),
        '占比_%': [v/TOTAL_BUDGET*100 for v in dimension_investment.values()]
    })
    dimension_df = dimension_df.sort_values('投资额_亿元', ascending=False).reset_index(drop=True)
    
    print("\n【六大维度投资分布】")
    print(dimension_df.to_string(index=False))
    
    return {
        'investment_df': investment_df,
        'comparison_df': comparison_df,
        'dimension_df': dimension_df,
        'topsis_improvement': {
            'before': china_score_before,
            'after': china_score_after,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }
    }


def save_results(result, analysis, output_dir):
    """
    保存结果到CSV文件
    """
    print("\n" + "=" * 70)
    print("保存结果...")
    print("=" * 70)
    
    # 1. 投资分配方案
    analysis['investment_df'].to_csv(
        output_dir / 'investment_allocation.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"  ✓ 投资分配方案: investment_allocation.csv")
    
    # 2. 投资前后对比
    analysis['comparison_df'].to_csv(
        output_dir / 'before_after_comparison.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"  ✓ 投资前后对比: before_after_comparison.csv")
    
    # 3. 维度投资分布
    analysis['dimension_df'].to_csv(
        output_dir / 'dimension_distribution.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"  ✓ 维度投资分布: dimension_distribution.csv")
    
    # 4. TOPSIS得分提升
    topsis_df = pd.DataFrame([{
        '指标': '中国TOPSIS得分',
        '投资前_2026': analysis['topsis_improvement']['before'],
        '投资后_2035': analysis['topsis_improvement']['after'],
        '提升幅度': analysis['topsis_improvement']['improvement'],
        '提升百分比_%': analysis['topsis_improvement']['improvement_pct']
    }])
    topsis_df.to_csv(
        output_dir / 'topsis_improvement.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"  ✓ TOPSIS得分提升: topsis_improvement.csv")
    
    print(f"\n所有结果已保存到 {output_dir.as_posix()}/ 目录")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("问题四：中国AI发展投资优化")
    print("=" * 70)
    print(f"投资总额：{TOTAL_BUDGET} 亿元")
    print(f"优化目标：最大化中国2035年AI综合竞争力（TOPSIS得分）")
    print("=" * 70)
    
    # 1. 加载数据
    data = load_all_data()
    
    # 2. 构建并求解优化模型
    model = OptimizationModel(data)
    result = model.solve()
    
    # 3. 分析结果
    analysis = analyze_results(result, data)
    
    # 4. 保存结果
    output_dir = ensure_outputs_dir()
    save_results(result, analysis, output_dir)
    
    print("\n" + "=" * 70)
    print("问题四运行完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
