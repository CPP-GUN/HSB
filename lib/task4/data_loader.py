# -*- coding: utf-8 -*-
"""
data_loader.py
问题四：数据加载模块
从问题一、二、三的结果中提取所需数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from parameters import INDICATORS, COUNTRIES, UPPER_LIMIT_MULTIPLIER


TASK4_DIR = Path(__file__).resolve().parent
# TASK4_DIR = <repo>/lib/task4
# parents[0] = <repo>/lib, parents[1] = <repo>
PROJECT_ROOT = TASK4_DIR.parents[1]


def _default_task2_outputs_dir() -> Path:
    return PROJECT_ROOT / "lib" / "task2" / "task2" / "outputs"


def _default_task3_outputs_dir() -> Path:
    return PROJECT_ROOT / "lib" / "task3" / "task3" / "outputs"


def _default_task1_dir() -> Path:
    return PROJECT_ROOT / "lib" / "task1" / "task1"


def load_weights(task2_dir=None):
    """
    从问题二加载24个指标的权重
    """
    task2_dir = _default_task2_outputs_dir() if task2_dir is None else Path(task2_dir)
    weights_path = task2_dir / 'weights_entropy.csv'
    df = pd.read_csv(weights_path, encoding='utf-8-sig')
    
    # 创建指标到权重的映射
    weights_dict = dict(zip(df['Indicator'], df['Weight']))
    
    # 按照INDICATORS顺序返回权重数组
    weights = np.array([weights_dict[ind] for ind in INDICATORS])
    
    print(f"[Data] 加载权重完成，共 {len(weights)} 个指标")
    return weights


def load_china_baseline_2026(task3_dir=None):
    """
    从问题三加载中国2026年的基准值（无额外投资）
    """
    task3_dir = _default_task3_outputs_dir() if task3_dir is None else Path(task3_dir)
    pred_path = task3_dir / 'predicted_indicators_2026_2035.csv'
    df = pd.read_csv(pred_path, encoding='utf-8-sig')
    
    # 筛选：year=2026, country=中国
    china_2026 = df[(df['year'] == 2026) & (df['country'] == '中国')].copy()
    
    # 创建指标到值的映射
    baseline_dict = dict(zip(china_2026['indicator'], china_2026['value_pred']))
    
    # 按照INDICATORS顺序返回基准值数组
    baseline = np.array([baseline_dict[ind] for ind in INDICATORS])
    
    print(f"[Data] 加载中国2026年基准值完成")
    print(f"       示例：AI研究人员={baseline[0]:.2f}, 大模型={baseline[6]:.2f}")
    return baseline


def load_all_countries_2035(task3_dir=None):
    """
    从问题三加载10国2035年的预测值
    返回：DataFrame，行=国家，列=指标
    """
    task3_dir = _default_task3_outputs_dir() if task3_dir is None else Path(task3_dir)
    pred_path = task3_dir / 'predicted_indicators_2026_2035.csv'
    df = pd.read_csv(pred_path, encoding='utf-8-sig')
    
    # 筛选：year=2035
    df_2035 = df[df['year'] == 2035].copy()
    
    # 透视表：行=国家，列=指标
    pivot = df_2035.pivot_table(
        index='country',
        columns='indicator',
        values='value_pred',
        aggfunc='first'
    )
    
    # 按照COUNTRIES和INDICATORS顺序重新排列
    pivot = pivot.reindex(index=COUNTRIES, columns=INDICATORS)
    
    print(f"[Data] 加载10国2035年预测值完成")
    print(f"       形状：{pivot.shape} (10国 × 24指标)")
    return pivot


def load_china_usa_2025(task1_dir=None):
    """
    从问题一加载中国和美国2025年的实际值
    用于判断领先/落后，设定指标上限
    """
    task1_dir = _default_task1_dir() if task1_dir is None else Path(task1_dir)
    data_path = task1_dir / 'data_raw_indicators.csv'
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 提取中国和美国的数据
    china_row = df[df['国家'] == '中国'].iloc[0]
    usa_row = df[df['国家'] == '美国'].iloc[0]
    
    # 转换为字典（去掉"国家"列）
    china_2025 = {col: china_row[col] for col in INDICATORS}
    usa_2025 = {col: usa_row[col] for col in INDICATORS}
    
    print(f"[Data] 加载中国和美国2025年实际值完成")
    return china_2025, usa_2025


def calculate_upper_limits(china_2025, usa_2025):
    """
    计算24个指标的增长上限
    规则：
    - 中国领先：L_j = 中国2025 × 1.5
    - 中国落后：L_j = 美国2025 × 2.0
    """
    upper_limits = {}
    
    for ind in INDICATORS:
        china_val = china_2025[ind]
        usa_val = usa_2025[ind]
        
        if china_val >= usa_val:
            # 中国领先
            upper_limits[ind] = china_val * UPPER_LIMIT_MULTIPLIER['leading']
            status = '领先'
        else:
            # 中国落后
            upper_limits[ind] = usa_val * UPPER_LIMIT_MULTIPLIER['lagging']
            status = '落后'
        
        # 特殊处理：比例类指标不超过100
        if ind in ['5G覆盖率', 'AI应用渗透率', '互联网普及率']:
            upper_limits[ind] = min(upper_limits[ind], 100.0)
    
    # 转换为数组
    upper_limits_array = np.array([upper_limits[ind] for ind in INDICATORS])
    
    print(f"[Data] 计算指标上限完成")
    print(f"       中国领先指标数：{sum(1 for ind in INDICATORS if china_2025[ind] >= usa_2025[ind])}")
    print(f"       中国落后指标数：{sum(1 for ind in INDICATORS if china_2025[ind] < usa_2025[ind])}")
    
    return upper_limits_array, upper_limits


def load_all_data():
    """
    一键加载所有数据
    """
    print("=" * 70)
    print("开始加载数据...")
    print("=" * 70)
    
    # 1. 加载权重
    weights = load_weights()
    
    # 2. 加载中国2026年基准值
    china_baseline_2026 = load_china_baseline_2026()
    
    # 3. 加载10国2035年预测值
    all_countries_2035 = load_all_countries_2035()
    
    # 4. 加载中国和美国2025年实际值
    china_2025, usa_2025 = load_china_usa_2025()
    
    # 5. 计算指标上限
    upper_limits_array, upper_limits_dict = calculate_upper_limits(china_2025, usa_2025)
    
    print("=" * 70)
    print("数据加载完成！")
    print("=" * 70)
    
    return {
        'weights': weights,
        'china_baseline_2026': china_baseline_2026,
        'all_countries_2035': all_countries_2035,
        'china_2025': china_2025,
        'usa_2025': usa_2025,
        'upper_limits_array': upper_limits_array,
        'upper_limits_dict': upper_limits_dict
    }


if __name__ == '__main__':
    # 测试数据加载
    data = load_all_data()
    
    print("\n数据摘要：")
    print(f"权重形状：{data['weights'].shape}")
    print(f"中国2026基准值形状：{data['china_baseline_2026'].shape}")
    print(f"10国2035预测值形状：{data['all_countries_2035'].shape}")
    print(f"指标上限形状：{data['upper_limits_array'].shape}")
