# 🏦 Loan Approval Prediction

> **Dataset:** [Loan Approval Prediction — Kaggle](https://www.kaggle.com/datasets/bhanupratapbiswas/loan-approval-prediction-case-study)
> **Goal:** Build a supervised model to predict loan approval using borrower features, with focus on preprocessing, handling class imbalance, and evaluation.

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── 📓 loan_approval_prediction.ipynb   # Full analysis notebook ⭐
├── 🐍 loan_prediction.py               # Standalone Python script (run in VS Code)
├── 📄 Loan_Approval_Report.docx        # Full report with findings & recommendations
├── 📋 README.md                        # You are here
│
├── visualizations/
│   ├── viz1_target_distribution.png
│   ├── viz2_categorical_vs_loan_status.png
│   ├── viz3_numeric_distributions.png
│   ├── viz4_credit_history_analysis.png
│   ├── viz5_correlation_heatmap.png
│   ├── viz6_smote_balance.png
│   ├── viz7_model_comparison_roc.png
│   ├── viz8_feature_importance.png
│   └── viz9_confusion_matrix.png
│
└── loan_prediction.csv                 # ⚠️ NOT included — download from Kaggle
```

---

## 📊 Dataset

| Property        | Value                                  |
|-----------------|-----------------------------------------|
| Source          | Kaggle (bhanupratapbiswas)             |
| Rows            | 614 loan applications                  |
| Columns         | 13                                      |
| Target          | `Loan_Status` (Y = Approved, N = Rejected) |
| Class Balance   | 68.7% Approved / 31.3% Rejected        |

### Column Reference

| Column | Description |
|--------|-------------|
| `Loan_ID` | Unique application identifier |
| `Gender` | Male / Female |
| `Married` | Marital status |
| `Dependents` | Number of dependents (0, 1, 2, 3+) |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Yes / No |
| `ApplicantIncome` | Primary applicant's income |
| `CoapplicantIncome` | Co-applicant's income |
| `LoanAmount` | Requested loan amount (₹ thousands) |
| `Loan_Amount_Term` | Loan repayment term (months) |
| `Credit_History` | 1 = good history, 0 = none/poor |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | Target variable |

---

## 🧹 Pipeline & Methodology

This project follows a strict, **leakage-safe** ML pipeline:

```
Load → Raw Overview → Clean → EDA → Feature Engineering →
Encode → Split (Train/Test) → SMOTE (train only) → Scale → Train → Evaluate
```

### Two Critical Design Decisions

**1. Clean BEFORE EDA**
EDA performed on dirty data produces misleading statistics. All visualizations in this project use the fully cleaned dataset.

**2. Split BEFORE SMOTE (avoiding data leakage)**
SMOTE is applied **only to the training set**, after the train/test split. Applying SMOTE before splitting would let synthetic data points "see" information from the test set, artificially inflating accuracy and ROC-AUC. The test set in this project remains untouched and reflects the real-world class imbalance.

### Missing Value Imputation

| Column | Missing | Strategy |
|--------|---------|----------|
| Gender | 13 | Mode |
| Married | 3 | Mode |
| Dependents | 15 | Mode |
| Self_Employed | 32 | Mode |
| LoanAmount | 22 | Median |
| Loan_Amount_Term | 14 | Mode |
| Credit_History | 50 | Mode |

All 614 rows retained — no rows dropped.

---

## ⚙️ Feature Engineering

| Feature | Formula | Purpose |
|---------|---------|---------|
| `Total_Income` | ApplicantIncome + CoapplicantIncome | Household income signal |
| `LoanAmount_log` | log(1 + LoanAmount) | Reduce skewness |
| `Total_Income_log` | log(1 + Total_Income) | Reduce skewness |
| `EMI` | LoanAmount / Loan_Amount_Term | Monthly repayment burden proxy |
| `Balance_Income` | Total_Income − (EMI × 1000) | Disposable income after loan |

---

## 🤖 Model Results (Evaluated on Untouched Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** ⭐ | 80.5% | 82.4% | 90.6% | **0.890** | **0.846** |
| Decision Tree | 70.7% | 78.1% | 78.8% | 0.785 | 0.708 |
| Random Forest | 77.2% | 80.8% | 87.1% | 0.838 | 0.817 |
| Gradient Boosting | 75.6% | 80.0% | 84.7% | 0.823 | 0.799 |
| SVM | 78.0% | 80.6% | 89.4% | 0.847 | 0.836 |

**Winner: Logistic Regression** — best ROC-AUC on honest, leakage-free evaluation. Simple linear models often outperform complex ensembles on small datasets (614 rows) where overfitting risk is high.

---

## 🔑 Key Findings

- **Credit_History is the dominant predictor** — ~80% approval rate with good history vs ~8% without
- Income level alone is a weak predictor — creditworthiness history matters far more
- Married applicants and Graduates show marginally higher approval rates
- Class imbalance (68.7%/31.3%) successfully corrected using leakage-safe SMOTE

---

## 💼 Business Recommendations

| # | Recommendation |
|---|----------------|
| 1 | Deploy Logistic Regression as the production model |
| 2 | Use 0.5 as the default classification threshold (best F1 balance) |
| 3 | Always split data before SMOTE in future pipelines |
| 4 | Investigate alternative credit signals for "thin-file" applicants |
| 5 | Re-evaluate threshold quarterly against actual default rates |

### Threshold Trade-off

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.3–0.4 | Lower | Higher | Growth-focused, approve more |
| **0.5** | **Balanced** | **Balanced** | **Recommended default** |
| 0.6–0.7 | Higher | Lower | Risk-averse, conservative |

---

## ▶️ How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### Option 1 — Jupyter Notebook (Recommended)
```bash
# Download loan_prediction.csv from Kaggle and place it in the same folder
jupyter notebook loan_approval_prediction.ipynb
```

### Option 2 — Python Script
```bash
python loan_prediction.py
```

> ⚠️ **Note:** `loan_prediction.csv` is not included in this repo due to dataset licensing/size conventions.
> Download it from [Kaggle](https://www.kaggle.com/datasets/bhanupratapbiswas/loan-approval-prediction-case-study) and place it in the root folder.

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualizations |
| scikit-learn | ML models & metrics |
| imbalanced-learn | SMOTE class balancing |
| Jupyter Notebook | Development environment |

---

## 👤 Author

**Internship Project** — Alfido Tech
*Yaksh M. Dhamat*
