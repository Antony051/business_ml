# 💳 Credit Default Risk — Business-Impact Tabular Classification

> **XGBoost + Optuna + SHAP + Cost-Sensitive Threshold Optimisation. Shifting the decision threshold from 0.50 to 0.23 catches 72.8% more defaults and delivers a net saving of £5.96M on a 6,000-customer test portfolio.**

---

## Business Result

By replacing the naive 0.5 probability threshold with a mathematically derived optimal threshold of **0.23**, the model transformed abstract ML accuracy into quantifiable financial outcome:

| | Default (0.50 threshold) | Optimised (0.23 threshold) | Change |
|---|---|---|---|
| Defaults caught | 741 | 1,281 | **+540** |
| Detection rate | — | — | **+72.8%** |
| Net financial saving | — | **£5,960,000** | vs. baseline |

The £5.96M figure accounts for every false positive incurred at the lower threshold — the saving is real net of intervention cost.

📄 **[Download Executive Summary PDF](outputs/executive_summary.pdf)** — 1-page briefing written for a non-technical risk committee.

---

## Cost Matrix

Traditional models optimise for accuracy or AUC and apply a 0.5 decision threshold. This treats every error as equally costly. In credit risk, that assumption is wrong by an order of magnitude:

| Error type | Business cost | Reasoning |
|---|---|---|
| False Negative (approve a defaulter) | **£25,000** | Principal loss + collections + legal fees |
| False Positive (reject a good customer) | **£2,500** | Lost interest margin + lifetime value |
| **Cost ratio** | **10:1** | Missing a default is 10× more expensive |

---

## Threshold Optimisation

The model outputs a raw probability. The optimal decision threshold was found by evaluating total business cost (FN cost + FP cost) across every threshold from 0.01 to 0.99:

![Cost Curve](outputs/figures/cost_curve.png)

The minimum of this curve occurs at **0.23** — meaning a customer with even a 23% estimated default probability has a higher expected cost if approved than if rejected.

---

## SHAP Interpretability

The model is not a black box. SHAP (SHapley Additive Explanations) was used to provide exact feature-level attribution for every prediction — a requirement for regulatory compliance in real banking deployments.

![SHAP Beeswarm](outputs/figures/shap_beeswarm.png)

**Top risk drivers identified:**

1. **`PAY_1` (most recent payment delay)** — Two or more months of delinquency causes a massive spike in default probability. Early-stage payment behaviour is the strongest leading indicator in the dataset.
2. **`LIMIT_BAL` (credit limit)** — Lower assigned credit limits correlate strongly with default, indicating the bank's baseline underwriting captures risk but the ML model extracts deeper non-linear patterns from the same signal.
3. **`utilization_rate`** — Customers approaching their credit ceiling (bill amount near limit balance) push the model sharply toward a default prediction.

---

## Confusion Matrices: Default vs Optimised Threshold

![Confusion Matrices](outputs/figures/confusion_matrices.png)

---

## Model Pipeline

```
UCI Credit Default Dataset (30,000 clients, Taiwan 2005)
        │
        ▼
┌──────────────────────────┐
│  Feature Engineering     │  ← utilization_rate, avg_utilization,
│                          │    payment_trend (6-month trajectory)
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Cross-Validated XGBoost │  ← SMOTE inside CV folds (no leakage)
│  + Optuna (30 trials)    │    Optuna Bayesian hyperparameter search
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Cost Curve Optimisation │  ← Find threshold minimising £ total cost
│  → threshold = 0.23      │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  SHAP Analysis           │  ← Global beeswarm + local waterfall
│  (global + per-customer) │    per individual prediction
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Executive Summary PDF   │  ← Auto-generated 1-page business brief
└──────────────────────────┘
```

---

## Tech Stack

| Tool | Purpose | Why |
|---|---|---|
| XGBoost | Classification model | Industry standard for tabular data; interpretable; no deep learning needed |
| Optuna | Hyperparameter tuning | Bayesian optimisation over 30 cross-validated trials; beats random/grid search |
| SMOTE (imblearn) | Class imbalance | Synthetic oversampling inside CV folds — prevents leakage into validation sets |
| SHAP | Explainability | Exact attribution via Shapley values; regulatory-grade interpretability |
| Pandas / NumPy | Data processing | — |
| Matplotlib / Seaborn | Visualisation | — |

---

## Dataset

**Default of Credit Card Clients** — UCI ML Repository (ID: 350)

- 30,000 credit card clients in Taiwan, April–September 2005
- 23 features: demographics, credit limit, 6-month payment history, bill amounts, actual payments
- Target: binary default (next month)
- Class distribution: 22.1% defaults (imbalanced)

**Engineered features added:**
- `utilization_rate` = bill amount / credit limit
- `avg_utilization` = mean utilisation across 6 months
- `payment_trend` = trajectory of repayment status (improving / worsening)

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis notebook
jupyter notebook notebooks/credit_risk_analysis.ipynb

# Regenerate the executive summary PDF
python generate_pdf.py
```

---

## Strategic Deployment Recommendations

1. **Decouple training from decisioning.** The XGBoost model outputs raw probabilities. A separate business-logic layer applies the 0.23 threshold — allowing the threshold to be adjusted without retraining the model.

2. **Dynamic cost matrices.** As interest rates or customer lifetime values change, the optimal threshold shifts. The threshold should be recalculated quarterly from updated cost parameters without touching the underlying model.

3. **Soft interventions in the 0.23–0.50 band.** Customers in this range should not be outright rejected. Lowering their credit limit, offering a balance transfer, or requiring a co-signer minimises the £2,500 false-positive penalty while protecting against the £25,000 default risk.

---

## Skills Demonstrated

- End-to-end tabular ML pipeline (EDA → feature engineering → model training → business output)
- Cost-sensitive threshold optimisation with asymmetric loss function
- Bayesian hyperparameter search (Optuna, 30 cross-validated trials)
- Imbalanced classification with SMOTE strictly inside CV folds (no leakage)
- SHAP global (beeswarm) and local (waterfall) interpretability
- Business framing: translating model output into £ financial impact
- Executive summary PDF auto-generation for non-technical stakeholders
