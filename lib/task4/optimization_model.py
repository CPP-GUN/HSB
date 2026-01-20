# -*- coding: utf-8 -*-
"""
optimization_model.py
问题四：优化模型核心模块
包含投资效应模型、TOPSIS评估、约束构建、优化求解
"""

import numpy as np
from scipy.optimize import minimize
from parameters import (
    INDICATORS, UNIT_COSTS, TIME_DISCOUNTS, SYNERGY_CONSTRAINTS,
    TOTAL_BUDGET, MAX_INVESTMENT_PER_INDICATOR, MIN_INVESTMENT_PER_INDICATOR
)


def calculate_investment_effect(investment, baseline, upper_limits, unit_costs, time_discounts):
    """
    计算投资效应（S型增长曲线）
    
    公式：Δx_j = (I_j / C_j) × (1 - x_j^baseline / L_j) × γ_j
    
    参数：
        investment: 投资额数组 (24,)
        baseline: 基准值数组 (24,)
        upper_limits: 上限数组 (24,)
        unit_costs: 单位成本数组 (24,)
        time_discounts: 时间折扣数组 (24,)
    
    返回：
        delta_x: 指标增长量数组 (24,)
    """
    # 避免除零
    safe_upper_limits = np.maximum(upper_limits, 1e-6)
    safe_unit_costs = np.maximum(unit_costs, 1e-6)
    
    # 边际递减效应
    marginal_effect = 1.0 - (baseline / safe_upper_limits)
    marginal_effect = np.maximum(marginal_effect, 0.0)  # 确保非负
    
    # 投资转化为增长量
    delta_x = (investment / safe_unit_costs) * marginal_effect * time_discounts
    
    return delta_x


def topsis_score(data_matrix, weights):
    """
    计算TOPSIS得分（与问题二一致）
    
    参数：
        data_matrix: 数据矩阵 (n_countries, n_indicators)
        weights: 权重数组 (n_indicators,)
    
    返回：
        scores: 得分数组 (n_countries,)
    """
    # 1. 向量归一化
    norms = np.sqrt(np.sum(data_matrix ** 2, axis=0))
    norms = np.where(norms == 0, 1, norms)  # 避免除零
    normalized = data_matrix / norms
    
    # 2. 加权归一化
    weighted = normalized * weights
    
    # 3. 正负理想解
    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)
    
    # 4. 距离计算
    dist_best = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))
    
    # 5. 相对贴近度
    scores = dist_worst / (dist_best + dist_worst + 1e-10)
    
    return scores


class OptimizationModel:
    """优化模型类"""
    
    def __init__(self, data):
        """
        初始化优化模型
        
        参数：
            data: 数据字典，包含weights, china_baseline_2026, all_countries_2035等
        """
        self.weights = data['weights']
        self.china_baseline_2026 = data['china_baseline_2026']
        self.all_countries_2035 = data['all_countries_2035'].copy()
        self.upper_limits = data['upper_limits_array']
        
        # 转换参数为数组
        self.unit_costs = np.array([UNIT_COSTS[ind] for ind in INDICATORS])
        self.time_discounts = np.array([TIME_DISCOUNTS[ind] for ind in INDICATORS])
        
        # 记录优化过程
        self.iteration = 0
        self.best_score = -np.inf
        
        print("[Model] 优化模型初始化完成")
    
    def objective_function(self, investment):
        """
        目标函数：最大化中国2035年TOPSIS得分
        
        参数：
            investment: 投资额数组 (24,)
        
        返回：
            -score: 负的TOPSIS得分（因为minimize求最小）
        """
        # 1. 计算投资效应
        delta_x = calculate_investment_effect(
            investment,
            self.china_baseline_2026,
            self.upper_limits,
            self.unit_costs,
            self.time_discounts
        )
        
        # 2. 计算中国2035年指标值
        china_2035 = self.china_baseline_2026 + delta_x
        
        # 3. 确保不超过上限
        china_2035 = np.minimum(china_2035, self.upper_limits)
        
        # 4. 更新10国数据矩阵
        data_matrix = self.all_countries_2035.values.copy()
        china_idx = list(self.all_countries_2035.index).index('中国')
        data_matrix[china_idx, :] = china_2035
        
        # 5. 计算TOPSIS得分
        scores = topsis_score(data_matrix, self.weights)
        china_score = scores[china_idx]
        
        # 6. 记录最佳结果
        self.iteration += 1
        if china_score > self.best_score:
            self.best_score = china_score
            if self.iteration % 10 == 0:
                print(f"  迭代 {self.iteration}: 当前最佳得分 = {china_score:.6f}")
        
        return -china_score  # 负号因为minimize
    
    def build_constraints(self):
        """
        构建约束条件
        """
        constraints = []
        
        # 1. 预算约束：∑I_j = 10000
        constraints.append({
            'type': 'eq',
            'fun': lambda I: np.sum(I) - TOTAL_BUDGET
        })
        
        # 2. 协同约束（5个）
        # 2.1 GPU ↔ 大模型
        idx_gpu = INDICATORS.index('GPU集群规模')
        idx_model = INDICATORS.index('大模型数量')
        coef = SYNERGY_CONSTRAINTS['gpu_model']['coefficient']
        constraints.append({
            'type': 'ineq',
            'fun': lambda I: (
                coef * (self.china_baseline_2026[idx_gpu] + 
                       calculate_investment_effect(I, self.china_baseline_2026, 
                                                  self.upper_limits, self.unit_costs, 
                                                  self.time_discounts)[idx_gpu])
                - (self.china_baseline_2026[idx_model] + 
                   calculate_investment_effect(I, self.china_baseline_2026, 
                                              self.upper_limits, self.unit_costs, 
                                              self.time_discounts)[idx_model])
            )
        })
        
        # 2.2 顶尖学者 ↔ 研究人员
        idx_researcher = INDICATORS.index('AI研究人员数量')
        idx_scholar = INDICATORS.index('顶尖AI学者数量')
        coef = SYNERGY_CONSTRAINTS['scholar_researcher']['coefficient']
        constraints.append({
            'type': 'ineq',
            'fun': lambda I: (
                coef * (self.china_baseline_2026[idx_researcher] + 
                       calculate_investment_effect(I, self.china_baseline_2026, 
                                                  self.upper_limits, self.unit_costs, 
                                                  self.time_discounts)[idx_researcher])
                - (self.china_baseline_2026[idx_scholar] + 
                   calculate_investment_effect(I, self.china_baseline_2026, 
                                              self.upper_limits, self.unit_costs, 
                                              self.time_discounts)[idx_scholar])
            )
        })
        
        # 2.3 AI_Book ↔ 研究人员
        idx_book = INDICATORS.index('AI_Book数量')
        coef = SYNERGY_CONSTRAINTS['book_researcher']['coefficient']
        constraints.append({
            'type': 'ineq',
            'fun': lambda I: (
                coef * (self.china_baseline_2026[idx_researcher] + 
                       calculate_investment_effect(I, self.china_baseline_2026, 
                                                  self.upper_limits, self.unit_costs, 
                                                  self.time_discounts)[idx_researcher])
                - (self.china_baseline_2026[idx_book] + 
                   calculate_investment_effect(I, self.china_baseline_2026, 
                                              self.upper_limits, self.unit_costs, 
                                              self.time_discounts)[idx_book])
            )
        })
        
        # 2.4 AI企业 ↔ 市场规模
        idx_market = INDICATORS.index('AI市场规模')
        idx_enterprise = INDICATORS.index('AI企业数量')
        coef = SYNERGY_CONSTRAINTS['enterprise_market']['coefficient']
        constraints.append({
            'type': 'ineq',
            'fun': lambda I: (
                coef * (self.china_baseline_2026[idx_market] + 
                       calculate_investment_effect(I, self.china_baseline_2026, 
                                                  self.upper_limits, self.unit_costs, 
                                                  self.time_discounts)[idx_market])
                - (self.china_baseline_2026[idx_enterprise] + 
                   calculate_investment_effect(I, self.china_baseline_2026, 
                                              self.upper_limits, self.unit_costs, 
                                              self.time_discounts)[idx_enterprise])
            )
        })
        
        # 2.5 Dataset ↔ 企业研发
        idx_rd = INDICATORS.index('企业研发支出')
        idx_dataset = INDICATORS.index('AI_Dataset数量')
        coef = SYNERGY_CONSTRAINTS['dataset_rd']['coefficient']
        constraints.append({
            'type': 'ineq',
            'fun': lambda I: (
                coef * (self.china_baseline_2026[idx_rd] + 
                       calculate_investment_effect(I, self.china_baseline_2026, 
                                                  self.upper_limits, self.unit_costs, 
                                                  self.time_discounts)[idx_rd])
                - (self.china_baseline_2026[idx_dataset] + 
                   calculate_investment_effect(I, self.china_baseline_2026, 
                                              self.upper_limits, self.unit_costs, 
                                              self.time_discounts)[idx_dataset])
            )
        })
        
        print(f"[Model] 构建约束条件完成：1个预算约束 + 5个协同约束")
        return constraints
    
    def solve(self):
        """
        求解优化问题
        """
        print("\n" + "=" * 70)
        print("开始优化求解...")
        print("=" * 70)
        
        # 初始猜测：均匀分配
        x0 = np.ones(24) * (TOTAL_BUDGET / 24)
        
        # 投资上下限（增加最小投资约束）
        bounds = [(MIN_INVESTMENT_PER_INDICATOR, MAX_INVESTMENT_PER_INDICATOR) for _ in range(24)]
        
        # 约束条件
        constraints = self.build_constraints()
        
        # 求解
        print("\n[Solver] 使用SLSQP算法求解...")
        result = minimize(
            self.objective_function,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 500,
                'ftol': 1e-6,
                'disp': False
            }
        )
        
        print("\n" + "=" * 70)
        print("优化求解完成！")
        print("=" * 70)
        print(f"状态：{result.message}")
        print(f"成功：{result.success}")
        print(f"迭代次数：{result.nit}")
        print(f"最终得分：{-result.fun:.6f}")
        
        return result


if __name__ == '__main__':
    # 测试优化模型
    from data_loader import load_all_data
    
    data = load_all_data()
    model = OptimizationModel(data)
    result = model.solve()
    
    print("\n最优投资方案（前5名）：")
    optimal_investment = result.x
    sorted_idx = np.argsort(optimal_investment)[::-1]
    for i in range(5):
        idx = sorted_idx[i]
        print(f"  {i+1}. {INDICATORS[idx]}: {optimal_investment[idx]:.2f} 亿元")
