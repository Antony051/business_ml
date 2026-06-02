# Business-Impact Tabular Classification: Credit Risk Optimization

This repository contains an end-to-end machine learning pipeline that predicts credit card defaults. Unlike traditional data science projects that optimize for abstract metrics like Accuracy or AUC, this project is built entirely around **Business ROI and Cost-Sensitive Threshold Optimization**.

By replacing the default `0.5` probability threshold with a mathematically derived optimal threshold (`0.23`), this model increases default detection by **72.8%** and demonstrates a net savings of **£5.96 Million** on a 6,000-customer test portfolio.

---

## 1. Project Design & Architecture

The project was designed with strict commercial constraints to mirror real-world banking ML deployments:
- **No Deep Learning:** The pipeline relies strictly on gradient boosting (XGBoost), which is the industry standard for tabular data due to its performance and explainability.
- **Optimization Standard:** Hyperparameter tuning was conducted using `Optuna` to efficiently navigate the search space over 30 cross-validated trials.
- **Imbalance Handling:** The target class (defaults) is inherently imbalanced (22%). To prevent data leakage, `imblearn`'s SMOTE algorithm was strictly encapsulated *inside* the cross-validation folds.
- **Explainability First:** The model is not treated as a black box. `SHAP` (SHapley Additive exPlanations) is utilized to ensure regulatory compliance and provide exact feature-level reasoning for every prediction.

## 2. Project Structure

```text
project_4_business_ml/
├── data/
│   └── raw/                          # Raw CSV files (credit_features.csv, credit_targets.csv)
├── notebooks/
│   └── credit_risk_analysis.ipynb    # Main pipeline containing EDA, Training, and SHAP
├── outputs/
│   ├── figures/                      # Generated visualization plots
│   └── executive_summary.pdf         # Auto-generated 1-page PDF for business stakeholders
├── requirements.txt                  # Python dependencies
└── generate_pdf.py                   # Script to convert results into the executive summary
```

## 3. Dataset Description

The project uses the **Default of Credit Card Clients Dataset** (UCI ML Repository ID: 350).

- **Samples:** 30,000 credit card clients in Taiwan (from April to September 2005).
- **Features:** 23 independent variables including:
  - **Demographics:** Age, Sex, Education, Marital Status.
  - **Financial History:** Credit Limit (`LIMIT_BAL`).
  - **Payment Behavior:** Repayment status for the last 6 months (`PAY_1` to `PAY_6`).
  - **Billing & Payments:** Bill statement amounts and actual amount paid over the last 6 months.
- **Target Variable:** `default` (1 = default next month, 0 = no default).
- **Class Distribution:** ~22.1% defaults (highly imbalanced).

Several business-relevant features were engineered from the raw data, such as `utilization_rate` (Bill Amount / Credit Limit), `avg_utilization`, and `payment_trend` (Trajectory of late payments).

## 4. Tests and Analyses Carried Out

### 4.1 Cost Matrix Definition
The traditional `0.5` threshold assumes all errors are equally bad. We defined a custom asymmetric cost matrix:
- **Cost of a False Negative (FN):** £25,000 (Approving a loan that defaults costs the principal, collections, and legal fees).
- **Cost of a False Positive (FP):** £2,500 (Wrongly rejecting a good customer costs the bank the lost interest margin and lifetime value).
- **Cost Ratio:** 10:1 (Missing a default is 10 times more expensive).

### 4.2 Threshold Optimization
The model outputs a raw default probability. We computationally evaluated the total business cost (FN cost + FP cost) across every threshold from `0.01` to `0.99` to find the exact point that minimized financial loss.

![Cost Curve](outputs/figures/cost_curve.png)

### 4.3 SHAP Interpretability
We performed global and local SHAP analyses to understand the model's decision-making process. The beeswarm plot identifies the highest overall risk drivers across the portfolio.

![SHAP Beeswarm Plot](outputs/figures/shap_beeswarm.png)

---

## 5. Results & Inference

### The Optimal Threshold
By evaluating the cost curve, we computationally proved that the **Optimal Threshold is 0.23**. 
If a customer has even a 23% probability of defaulting, the mathematical expected value dictates that the bank must intervene, due to the heavy £25,000 penalty of missing a default.

### Financial ROI
Based on the 6,000-customer test set, shifting the decision threshold from the naive 0.50 to the optimal 0.23 yielded the following results:
- **Defaults Caught:** 540 additional defaults were identified that the baseline model would have approved.
- **Detection Increase:** A **72.8%** increase in default detection.
- **Net Savings:** Even after accounting for the £2,500 penalty incurred for every additional False Positive, the model achieved a net savings of **£5,960,000** on the test set alone.

![Confusion Matrices](outputs/figures/confusion_matrices.png)

### SHAP Inferences
Based on the SHAP analysis, the strongest predictors of default are:
1. **`PAY_1` (Recent Payment Delay):** Customers who delayed their most recent payment by 2 or more months saw a massive spike in default probability. Early-stage delinquency is the most reliable leading indicator.
2. **`LIMIT_BAL` (Credit Limit):** Lower assigned credit limits are strongly correlated with default, indicating that the bank's baseline underwriting rules capture risk, but the ML model extracts deeper non-linear patterns.
3. **`utilization_rate`:** Customers who max out their available credit (Bill Amount nearing Limit Balance) push the model heavily toward a default prediction.

## 6. Strategic Recommendations for Deployment

1. **Decouple Training from Decisioning:** The XGBoost model should output raw probabilities, while a separate business-logic layer applies the `0.23` threshold.
2. **Dynamic Cost Matrices:** As interest rates or lifetime values change, the threshold should be dynamically recalculated without needing to retrain the underlying XGBoost model.
3. **Soft Interventions:** Rather than an outright rejection for customers landing between the `0.23` and `0.50` probabilities, banks should implement "soft interventions" (e.g., lowering credit limits, offering balance transfers) to minimize the £2,500 False Positive penalty while mitigating the £25,000 default risk.
