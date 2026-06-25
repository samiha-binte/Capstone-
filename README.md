# Heart Disease Risk Prediction — ML Capstone Project

A machine learning project that predicts the likelihood of heart disease using clinical 
and demographic patient data. Three supervised classification models were trained, 
evaluated, and compared on a dataset of 918 patient records from Kaggle.

**Kent State University | BA-64099 | July 2025**

---

## Project Overview

Cardiovascular disease is one of the leading causes of death globally. This project 
explores whether machine learning can support early detection of high-risk patients 
using commonly available clinical measurements — enabling proactive medical 
decision-making without invasive testing.

---

## Dataset

- **Source:** [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size:** 918 patient records → 643 after outlier removal
- **Target variable:** HeartDisease (1 = present, 0 = absent)

**Features used:**

| Type | Features |
|---|---|
| Demographic | Age, Sex |
| Clinical | RestingBP, Cholesterol, MaxHR, Oldpeak, FastingBS |
| Diagnostic | ChestPainType, RestingECG, ExerciseAngina, ST_Slope |

---

## Methodology

**Data Preprocessing**
- Replaced zero values in RestingBP and Cholesterol with column medians
- Removed outliers using the IQR method (918 → 643 records)
- One-hot encoded all categorical variables using `pd.get_dummies(drop_first=True)`

**Feature Selection**
- Mutual Information scoring (SelectKBest)
- Random Forest feature importances
- Recursive Feature Elimination (RFE) with Logistic Regression

**Top predictive features:**

| Feature | Mutual Info Score | RF Importance |
|---|---|---|
| ST_Slope_Up | 0.20 | 0.17 |
| ExerciseAngina_Y | 0.17 | 0.10 |
| Oldpeak | 0.11 | 0.10 |
| MaxHR | 0.10 | 0.13 |

---

## Models & Results

All models trained on a stratified 80/20 train-test split.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 88% | 0.87 | 0.88 | 0.88 | 0.95 |
| Decision Tree | 84% | 0.83 | 0.83 | 0.83 | 0.88 |
| Random Forest | 87% | 0.83 | 0.90 | 0.86 | 0.94 |

**Best overall:** Logistic Regression achieved the highest ROC-AUC (0.95).  
**Best recall:** Random Forest achieved 90% recall — critical for minimizing missed diagnoses.

---

## Key Findings

- Patients with lower MaxHR, higher Oldpeak, exercise-induced angina, and flat 
  ST slopes showed significantly higher rates of heart disease
- Transfer of clinical knowledge aligned with model outputs — ST_Slope_Up and 
  ExerciseAngina were the strongest predictors
- Increasing dataset quality via outlier removal improved model generalization 
  more than raw sample size

---

## Ethical Considerations

- **Bias & Fairness:** Gender (Sex_M) is a feature — fairness auditing recommended 
  before any real-world deployment
- **Privacy:** Dataset is anonymized and publicly available; real-world use requires 
  HIPAA compliance and patient consent
- **Accountability:** ML outputs should support, not replace, clinical judgment
- **Transparency:** SHAP or LIME explainability frameworks recommended for 
  production deployment

---

## Tools & Technologies

- Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Google Colab / Jupyter Notebook
- Kaggle dataset API (kagglehub)

---

## Files

| File | Description |
|---|---|
| `Capstone_project_Samiha.ipynb` | Full notebook — EDA, preprocessing, modeling, evaluation |
| `Capstone_Project_Final_Report_Samiha.pdf` | Complete written report with methodology and findings |
| `Heart_Disease_Prediction_Capstone_Presentation.pptx` | Slide deck presentation |
| `Project_Proposal.pdf` | Original project proposal |
