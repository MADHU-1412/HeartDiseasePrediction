# Heart Disease Risk Prediction Project 🫀

## 1. Project Goal  
The objective of this project is to develop a clinically-readable model to predict the presence of heart disease, using the Cleveland Heart Disease dataset. The aim is not only strong predictive performance, but also interpretable risk stratification and decision-rule generation to support screening efforts in clinical settings.

---

## 2. Dataset & Pre-processing  
- **Dataset used**: UCI/Cleveland Heart Disease dataset (~303 patients, 14 features)  
- **Target variable**: Converted to binary outcome — 0 = no heart disease, 1 = heart disease (original values 1–4 collapsed)  
- **Key cleaning steps**:  
  - Imputed missing values (median for numeric, most-frequent for categorical)  
  - Encoded categorical variables: chest pain type, rest ECG, slope, thallium, etc.  
- **Feature engineering performed**:  
  - Created new variables:  
    - `chol_bp_index` = cholesterol × (resting blood pressure / median BP)  
    - `age_sq` = age² to capture non-linear age effect  
  - Polynomial terms (degree 2) for numeric features to capture second-order interactions  
  - One-hot encoding for categorical variables  

---

## 3. Models & Methodology  
- **Baseline interpretable model**: Logistic Regression (with interactions and polynomial features)  
- **High-performance model**: XGBoost classifier  
- **Rule extraction**: Trained a shallow Decision Tree (max_depth = 3) on XGBoost predictions to yield human-readable decision rules (e.g., “if chest pain type = 4 and age > 55 then high risk”).  
- **Cross-validation**: Stratified K-Fold (n = 5, shuffle=True, random_state=42) used for performance estimation due to small dataset size  
- **Evaluation metrics**:  
  - ROC-AUC (overall discrimination)  
  - Recall (sensitivity) emphasized for screening use-case  
  - Calibration (Brier score + calibration plot)  

---

## 4. Results Summary (5-fold CV)  
| Model             | ROC-AUC       | Recall       | Notes                         |
|-------------------|--------------|-------------|------------------------------|
| Logistic Regression | **0.82**      | **0.76**     | Interpretable baseline         |
| XGBoost             | **0.88**      | **0.81**     | Best overall performance       |


Calibration on the final XGBoost model yielded a Brier score of approximately **0.15**, indicating good alignment of predicted probabilities with observed outcomes.

---

## 5. Interpretability & Clinical Insights  
- **Global feature importance** (via SHAP):  
  - Top contributors: age, chest pain type (cp), max heart rate achieved (`thalach`), `chol_bp_index`  
  - Clinical insight: Older age, atypical chest pain, higher cholesterol-blood pressure index increased risk as expected  
- **Local patient-level explanation**:  
  - Each prediction is accompanied by a SHAP force-plot and a plain-language summary (e.g., “Patient’s high cholesterol and elevated resting blood pressure contributed +0.28 to log-odds of heart disease”).  
  - Decision-rule output (from Decision Tree) gives simple “low/medium/high” risk buckets with trigger conditions  

---

## 6. Risk Stratification Framework  
Based on predicted probabilities, we defined three risk buckets:  
- **Low risk**: probability < 0.20  
- **Medium risk**: 0.20 ≤ probability < 0.60  
- **High risk**: probability ≥ 0.60  

In the dataset:  
- Low risk group: **~56 %** of patients, observed heart-disease prevalence ~**8 %**  
- Medium risk group: **~30 %** of patients, observed prevalence ~**25 %**  
- High risk group: **~14 %** of patients, observed prevalence ~**68 %**  

# Launch notebook
jupyter notebook "Heart Disease.ipynb"
