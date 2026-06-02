# **AGY Skill Playbook: Business-Impact Tabular Classification**

## **1\. System Goal**

Develop an end-to-end machine learning pipeline for tabular classification (e.g., Credit Risk, Healthcare Readmission, or Telecom Churn). The ultimate objective is not just model accuracy, but framing the model's predictions as a financial/business decision using threshold optimization and interpretability.

## **2\. Technical Constraints & Coding Standards**

When generating the Python codebase and Jupyter Notebooks for this project, the agent must strictly adhere to the following paradigms:

* **No Deep Learning:** Do not use PyTorch, TensorFlow, or any neural network architectures. Stick strictly to industry-standard gradient boosting (XGBoost or LightGBM).  
* **Optimization Standard:** Use optuna for all hyperparameter tuning. Do not use standard GridSearch or RandomSearch.  
* **Imbalance Handling:** If the chosen dataset is imbalanced (e.g., default/churn rates), implement imblearn (SMOTE or class weights) within a strict cross-validation pipeline to prevent data leakage.  
* **Functional Preprocessing:** Write all data cleaning, feature engineering, and EDA pipelines as pure, isolated Python functions. No monolithic procedural blocks.

## **3\. Pipeline Architecture & Components**

Instruct subagents to build the pipeline with the following distinct functional phases:

* **Phase 1: EDA & Feature Engineering:** Generate robust baseline statistics, handle missing values, and engineer business-relevant features.  
* **Phase 2: Model Training & Tuning:** Train an XGBoost/LightGBM model. Optimize hyperparameters using an Optuna study object.  
* **Phase 3: Threshold Optimization (Critical):** Do NOT use the default 0.5 decision threshold. Define a custom Cost Matrix (Cost of False Positive vs. Cost of False Negative) and computationally find the optimal probability threshold that minimizes total business cost.  
* **Phase 4: SHAP Interpretability:** \* Generate a SHAP **beeswarm plot** to explain global feature importance.  
  * Generate SHAP **waterfall plots** for specific, individual predictions to answer: "Why did this specific customer get flagged?"  
* **Phase 5: Business Impact & What-If Analysis:** Calculate the concrete financial impact. Formulate the output explicitly as: *"At this threshold, catching X% more defaults/churns saves £Y in losses at £Z intervention cost per case."* Include a "What-if" simulation altering 1-2 features to show product-level thinking.

## **4\. Deliverables & UI/UX Polish Requirements**

The agent must ultimately output two finalized deliverables:

1. **Polished Jupyter Notebook:** The code must be cleanly separated into cells with well-written Markdown headers explaining the *business* rationale behind technical choices.  
2. **Executive Summary PDF:** A 1-page automated markdown-to-PDF export summarizing the ROI, cost matrix logic, and SHAP insights, written explicitly for a non-technical manager.

## **5\. Execution Strategy**

1. Initialize in **Planning mode** to select the target dataset (Credit Risk, Healthcare, or Telecom) and define the hypothetical Cost Matrix.  
2. Spawn an async subagent (agy /spawn) to handle the heavy Optuna hyperparameter sweep in the background.  
3. Spawn a second subagent to compute the SHAP values and generate the waterfall/beeswarm visualizations using matplotlib/shap.  
4. Consolidate the outputs into the final Jupyter Notebook and generate the 1-page executive summary.