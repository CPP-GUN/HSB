# Problem 2: AI Development Capability Evaluation Model and 2025 Competitiveness Ranking

> **Perspective**: Joint modeling-handbook & code-handbook
>
> This document is written to serve *two audiences simultaneously*: (i) modelers/readers who focus on mathematical logic and assumptions, and (ii) implementers/reviewers who focus on reproducibility, code structure, and data flow.

---

## 1. Problem Restatement

Based on the AI development indicators identified and validated in **Problem 1**, this task aims to:

1. Construct a **scientifically sound and objective evaluation model** for national AI development capability;
2. Apply the model to evaluate **10 representative countries** using **24 indicators**;
3. Produce the **2025 AI competitiveness ranking**;
4. Verify the reliability and robustness of the ranking results.

The evaluated countries are:

> United States, China, United Kingdom, Germany, South Korea, Japan, France, Canada, United Arab Emirates, India.

---

## 2. Modeling Philosophy and Design Logic

### 2.1 Core Challenges

The evaluation of national AI capability involves several intrinsic difficulties:

* **High-dimensional indicators** (24 indicators across 6 dimensions);
* **Heterogeneous units and scales**;
* **Avoidance of subjective weighting**;
* **Need for ranking reliability and robustness verification**.

### 2.2 Overall Strategy

To address these challenges, a **three-model coupled framework** is adopted:

1. **Entropy Weight Method (EWM)** – objective, data-driven weight determination;
2. **TOPSIS** – multi-dimensional comprehensive evaluation via ideal-solution distance;
3. **Grey Relational Analysis (GRA)** – structural similarity-based ranking validation.

Additional procedures are introduced to ensure reliability:

* **Borda count fusion** – to integrate multiple rankings;
* **Spearman rank correlation** – to quantitatively test consistency;
* **Sensitivity analysis** – to assess robustness under weight perturbations.

---

## 3. Basic Assumptions

To ensure tractability and interpretability, the following assumptions are made:

1. **Indicator Sufficiency Assumption**
   The selected 24 indicators comprehensively reflect national AI development capability.

2. **Benefit-Type Indicator Assumption**
   All indicators are treated as benefit-type variables, i.e., larger values indicate stronger AI capability.

3. **Cross-Sectional Consistency Assumption**
   All indicator values correspond to the same evaluation year (2025) and are cross-sectionally comparable.

4. **Independence in Weight Perturbation**
   In sensitivity analysis, each indicator weight is perturbed independently while others remain unchanged.

---

## 4. Data Description and Preprocessing

### 4.1 Data Source

* Data file: `data_raw_indicators.csv`
* Origin: Problem 1 (manually collected from authoritative public sources)
* Structure:

  * Rows: 10 countries
  * Columns: 24 AI indicators + country name

### 4.2 Indicator Dimensions

The 24 indicators are grouped into six dimensions:

* **T (Talent)**: AI researchers, top AI scholars, AI graduates
* **A (Application)**: AI firms, market size, application penetration, large models
* **P (Policy)**: social trust, AI policies, subsidies
* **R (R&D)**: enterprise R&D, government AI investment, international AI investment
* **I (Infrastructure)**: 5G, GPU clusters, bandwidth, penetration, power, platforms, data centers, TOP500
* **O (Output)**: AI books, datasets, GitHub projects

### 4.3 Normalization

To eliminate dimensional effects, **Min–Max normalization** is applied:

[
x'*{ij} = \frac{x*{ij} - \min_i x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}, \quad x'_{ij} \in [0,1]
]

> **Engineering note**: If an indicator has zero variance, its normalized values are set to 1 to avoid numerical instability.

---

## 5. Model I: Entropy Weight Method (EWM)

### 5.1 Rationale

Entropy reflects the amount of information provided by an indicator. Greater dispersion implies higher information content and therefore higher importance.

### 5.2 Mathematical Formulation

1. **Proportion matrix**:
   [
   p_{ij} = \frac{x'*{ij}}{\sum*{i=1}^{n} x'_{ij}}
   ]

2. **Entropy**:
   [
   E_j = -\frac{1}{\ln n} \sum_{i=1}^{n} p_{ij} \ln p_{ij}
   ]

3. **Information utility**:
   [
   d_j = 1 - E_j
   ]

4. **Weight**:
   [
   w_j = \frac{d_j}{\sum_{j=1}^{m} d_j}
   ]

### 5.3 Output

* Indicator weights
* Entropy values
* Redundancy (information utility)

All results are stored in `weights_entropy.csv`.

---

## 6. Model II: TOPSIS Comprehensive Evaluation

### 6.1 Principle

The optimal alternative should be closest to the **positive ideal solution** and farthest from the **negative ideal solution**.

### 6.2 Steps

1. **Vector normalization**:
   [
   r_{ij} = \frac{x'*{ij}}{\sqrt{\sum*{i=1}^{n} (x'_{ij})^2}}
   ]

2. **Weighted normalized matrix**:
   [
   Z_{ij} = w_j r_{ij}
   ]

3. **Ideal solutions**:
   [
   Z^+*j = \max_i Z*{ij}, \quad Z^-*j = \min_i Z*{ij}
   ]

4. **Distances**:
   [
   D_i^+ = \sqrt{\sum_j (Z_{ij} - Z_j^+)^2}, \quad D_i^- = \sqrt{\sum_j (Z_{ij} - Z_j^-)^2}
   ]

5. **Relative closeness**:
   [
   C_i = \frac{D_i^-}{D_i^+ + D_i^-}
   ]

Results are stored in `result_topsis.csv`.

---

## 7. Model III: Grey Relational Analysis (GRA)

### 7.1 Purpose

To validate TOPSIS ranking from the perspective of **structural similarity**.

### 7.2 Formulation

1. **Reference sequence**:
   [
   x_{0j} = \max_i x'_{ij}
   ]

2. **Grey relational coefficient**:
   [
   \xi_{ij} = \frac{\min \Delta + \rho \max \Delta}{|x_{0j} - x'_{ij}| + \rho \max \Delta}
   ]

3. **Weighted relational grade**:
   [
   \gamma_i = \sum_{j=1}^{m} w_j \xi_{ij}
   ]

where ( \rho = 0.5 ).

Results are stored in `result_grey_relation.csv`.

---

## 8. Consistency Verification

### 8.1 Spearman Rank Correlation

The consistency between TOPSIS and GRA rankings is evaluated using Spearman’s rank correlation coefficient:

[
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}
]

A permutation-based test is applied if SciPy is unavailable.

Results are stored in `result_spearman.csv`.

---

## 9. Ranking Fusion and Final Evaluation

### 9.1 Borda Count

[
B_i = R_i^{\text{TOPSIS}} + R_i^{\text{GRA}}
]

### 9.2 Comprehensive Score

[
S_i = 0.5 C_i + 0.5 \gamma_i
]

### 9.3 Grade Classification

| Grade | Condition       |
| ----- | --------------- |
| A+    | S ≥ 0.60        |
| A     | 0.30 ≤ S < 0.60 |
| B     | 0.25 ≤ S < 0.30 |
| C     | S < 0.25        |

Final results are stored in `result_final_ranking.csv`.

---

## 10. Sensitivity Analysis

To test robustness, each indicator weight is perturbed independently:

[
w_j^{(\delta)} = w_j (1 + \delta), \quad \delta \in {-30%, -15%, 0, +15%, +30%}
]

After renormalization, TOPSIS rankings are recomputed.

The ranking range for each country is defined as:

[
\text{Range}_i = \max R_i - \min R_i
]

Results are stored in:

* `result_sensitivity_rank_matrix.csv`
* `result_sensitivity_range.csv`

---

## 11. Dimension-Level Analysis

For each dimension (k), the score is computed as:

[
\text{Score}*{ik} = \frac{1}{|I_k|} \sum*{j \in I_k} x'_{ij}
]

Results are stored in `result_dimension_scores.csv`.

---

## 12. Reproducibility and Engineering Design

* Code is modularized into three files:

  * `model_utils.py`
  * `evaluation_pipeline.py`
  * `run_task2.py`
* All outputs are written to `outputs/`.
* Fixed random seeds ensure reproducibility.
* No visualization is included at this stage.

---

## 13. Summary

This chapter establishes a **fully objective, verifiable, and robust** evaluation framework for national AI development capability. By combining entropy-based weighting, distance-based evaluation, and structural validation, the model produces a reliable 2025 AI competitiveness ranking and lays a solid foundation for dynamic forecasting in Problem 3.
