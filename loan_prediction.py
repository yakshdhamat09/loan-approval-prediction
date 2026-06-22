# Loan Approval Prediction
#  Dataset: kaggle.com/datasets/bhanupratapbiswas/loan-approval-prediction-case-study
#___________________________________________________________________________________________

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

#ML
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE

#styling
sns.set_theme(style= 'darkgrid', palette = 'muted')
plt.rcParams.update({'figure.dpi' : 120, 'font.size' : 11})

def save_show(name):
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    print(f'Saved : {name}.png')
    plt.show()


# S1 - Loading The Dataset :

print('S1 - Loading the Dataset : ')
df = pd.read_csv("D:\int_task\Loan Approval\loan_prediction.csv")
print("Shape : ", df.shape)
print(f'Columns : {df.columns.tolist()}')
print()
print(df.head())


#s2 - Raw Data overview (before cleaning) :
print('S2 - RAw Data Overview : ')

print('\nData Types :')
print(df.dtypes)

print('\nMissing Values :')
print(df.isnull().sum())

print('\nSummary Statistics :')
print(df.describe().round(2))

print('\nTarget Distribution : ')
print(df['Loan_Status'].value_counts())
approved_pct = (df['Loan_Status'] == 'Y').sum() / len(df) * 100
print(f'Approved : {approved_pct:.1f}')

##s3 Data Cleaning :
print("S3 - Data Cleaning :")

df_clean = df.copy()

# Impute missing values________
#categorical columns -> mode (most frequent value)
#Numeric columns -> median (robust to outliers)

df_clean['Gender'] = df_clean['Gender'].fillna(df_clean['Gender'].mode()[0])
df_clean['Married'] = df_clean['Married'].fillna(df_clean['Married'].mode()[0])
df_clean['Dependents'] = df_clean['Dependents'].fillna(df_clean['Dependents'].mode()[0])
df_clean['Self_Employed'] = df_clean['Self_Employed'].fillna(df_clean['Self_Employed'].mode()[0])
df_clean['LoanAmount'] = df_clean['LoanAmount'].fillna(df_clean['LoanAmount'].median())
df_clean['Loan_Amount_Term'] = df_clean['Loan_Amount_Term'].fillna(df_clean['Loan_Amount_Term'].mode()[0])
df_clean['Credit_History'] = df_clean['Credit_History'].fillna(df_clean['Credit_History'].mode()[0])

print('Missing values after cleaning : ')
missing_after = df_clean.isnull().sum()
print(missing_after[missing_after > 0] if missing_after.any() else 'No missing values, dataset is clean')

print('Rows Before : ', len(df))
print(f'Rows after :  {len(df_clean)} (no rows dropped, all imputed)')



#S4 - EDA (on Clean dataset) :
print('-'*60)
print('S4 - EDA (on clean Dataset) :')
print()

# viz1 - Target Distribution :
print('Viz1 - Target Distribution :')
fig, axes = plt.subplots(1, 2, figsize = (10, 4))

counts = df_clean['Loan_Status'].value_counts()
axes[0].bar(['Approved (Y)', 'Rejected (N)'], counts.values,
             color = ['#2DC653', '#E85D04'], edgecolor = 'white', width = 0.5)
axes[0].set_title('Loan Approval Distribution', weight = 'bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v+4, str(v), ha = 'center', fontsize = 12, fontweight = 'bold')

axes[1].pie(counts.values, labels = ['Approved (Y)', 'Rejected (N)'],
            colors = ['#2DC653', '#E85D04'], autopct = '%1.1f%%',
            startangle = 90, wedgeprops = {'edgecolor': 'White', 'linewidth' : 2})
axes[1].set_title('Approval Rate', weight = 'bold')
plt.suptitle('Target Variable Distribution (Clean Data)', fontsize = 13, weight = 'bold')
save_show('viz1_target_distribution')


#viz2 - Categorical Features vs Loan Status :
print()
print('viz2 - Categorical Features vs Loan Status :')
cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed',
            'Property_Area', 'Dependents']

fig, axes = plt.subplots(2, 3, figsize = (15, 9))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df_clean[col], df_clean['Loan_Status'], 
                     normalize = 'index')*100
    ct.plot(kind = 'bar', ax = axes[i], color = ['#E85D04', '#2DC653'],
            edgecolor = 'white', rot = 30, width = 0.6)
    axes[i].set_title(f'{col} vs Loan Status', weight = 'bold')
    axes[i].set_ylabel('Percentage (%)')
    axes[i].set_xlabel('')
    axes[i].legend(['Rejected (N)', 'Approved (Y)'], fontsize = 9)
    

plt.suptitle('categorical Features vs Loan Approval', fontsize = 13, weight = 'bold')
save_show('viz2_categorical_vs_loan_status')
print()


#viz3 - Numeric Distributions :
print('viz3 - Numeric Distributions :')
num_cols = ['ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term']

fig, axes = plt.subplots(2, 2, figsize = (12, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    for status, color, label in [('Y', '#2DC653', 'Approved'),
                                 ('N', '#E85D04', 'Rejected')]:
        data = df_clean[df_clean['Loan_Status'] == status][col]
        axes[i].hist(data, bins = 30, alpha = 0.6, color = color,
                     label = label, edgecolor = 'white')

    axes[i].set_title(f'Distribution of {col}', weight = 'bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

plt.suptitle('Numeric Feature Distributions by Loan Status', fontsize = 13, weight = 'bold')
save_show('viz3_numeric_distributions')
print()


#viz4 - Credit History vs Loan Status :
print('viz4 - Credit History Analysis :')
fig, axes = plt.subplots(1, 2, figsize = (11, 4))

# credit history approval rate
ch_approval = df_clean.groupby('Credit_History')['Loan_Status'].apply(
    lambda x : (x == 'Y').sum() / len(x)*100).reset_index()

ch_approval.columns = ['Credit_History', 'Approval_Rate']
ch_approval['Credit_History'] = ch_approval['Credit_History'].map(
    {0.0: 'No History (0)', 1.0 : 'Has History (1)'})

axes[0].bar(ch_approval['Credit_History'], ch_approval['Approval_Rate'],
            color = ['#E85D04', '#2DC653'], edgecolor = 'white', width = 0.4)
axes[0].set_title('Approval Rate by Credit History', weight = 'bold')
axes[0].set_ylabel('Approval Rate (%)')
axes[0].set_ylim(0, 110)
for i, v in enumerate(ch_approval['Approval_Rate']):
    axes[0].text(i, v+3, f'{v:.1f}%', ha = 'center', fontsize = 12, weight = 'bold')

#Loan amount by credit history
df_clean['Credit_History_Label'] = df_clean['Credit_History'].map(
    {0.0 : 'No History (0)', 1.0 : 'Has History (1)'})
sns.boxplot(data = df_clean, x = 'Credit_History_Label', y = 'LoanAmount',
            hue = 'Loan_Status', palette = ['#E85D04', '#2DC653'],
            ax = axes[1], showfliers = False)
axes[1].set_title('Loan Amount by Credit History & Status', weight = 'bold')
axes[1].set_xlabel('Credit History')

plt.suptitle('Credit History Impact on Loan Approval', fontsize = 13, weight = 'bold')
save_show('viz4_credit_history_analysis')
print()


#v5 - Summary Statictics Table :
print('viz5 - Summary statistics :')
print(df_clean[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                'Loan_Amount_Term', 'Credit_History']].describe().round(2))
print()



#S5 - Feature Engineering & Encoding :
print('-'*60)
print('S5 - Feature Engineering & Label Encoding :')

df_ml = df_clean.copy()

# New Featyres :
df_ml['Total_Income'] = df_ml['ApplicantIncome'] + df_ml['CoapplicantIncome']
df_ml['LoanAmount_log'] = np.log1p(df_ml['LoanAmount'])
df_ml['Total_Income_log'] = np.log1p(df_ml['Total_Income'])
df_ml['EMI'] = df_ml['LoanAmount'] / df_ml['Loan_Amount_Term']
df_ml['Balance_Income'] = df_ml['Total_Income'] - (df_ml['EMI'] * 1000)

print(' New features created : ')
print(' Total_Income      =   ApplicantIncome + CoapplicantIncome')
print(' LoanAmount_log    = log(LoanAmount) -> reduces skewness ')
print(' Total_Income_log  = log(Total_Income) -> reduces skewness')
print(' EMI               = LoanAmount / Loan_Amount_Term')
print(' Balance_Income    = Total_Income - (EMI * 1000)')

# Label Encoding :
le = LabelEncoder()
encode_cols = ['Gender', 'Married', 'Dependents', 'Education',
               'Self_Employed', 'Property_Area']

for col in encode_cols :
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# target : Y -> 1, N -> 0 
df_ml['Loan_Status'] = df_ml['Loan_Status'].map({'Y' : 1, 'N' : 0})

print('\nLabel encoding done.')
print()

#viz6 - Correlation heatmap (encoded features):
print('viz6-Correlation heatmap : ')
heatmap_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'LoanAmount_log', 'Loan_Amount_Term',
                'Credit_History', 'Property_Area',
                'Total_Income_log', 'EMI', 'Loan_Status']

fig, ax = plt.subplots(figsize = (12, 9))
corr = df_ml[heatmap_cols].corr()
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(corr, mask = mask, annot = True, fmt = '.2f', cmap = 'RdYlGn',
            linewidths = 0.5, ax =ax, vmin = -1, vmax = 1)
ax.set_title('Feature Correlation Heatmap', fontsize = 14, weight = 'bold')
save_show('viz6_correlation_heatmap') 
print()


#S6- Train/test Split & Handle Imbalance with SMOTE :
print('-'*60)
print('S6 - Splitting data & Handling Class Imbalance')

feature_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'LoanAmount_log', 'Loan_Amount_Term',
                'Credit_History', 'Property_Area',
                'Total_Income_log', 'EMI', 'Balance_Income']

X = df_ml[feature_cols]
y = df_ml['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 42, stratify = y)

print(f'Train set - Approved : {sum(y_train ==1)} | Rejected : {sum(y_train == 0)}')
print(f'Test set  - Approved : {sum(y_test == 1)} | Rejected : {sum(y_test == 0)}')
print(f'Test set is kept Imbalanced -> represents real world distribution')


#2. SMOTE only on training data
smote = SMOTE(random_state = 42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f'\nTraining set after SMOTE -> Approved : {sum(y_train_res == 1)} | '
      f'Rejected : {sum(y_train_res == 0)}')


# viz7 : Before and After SMOTE :
print('viz7 - SMOTE balance chart')
fig, axes = plt.subplots(1, 2, figsize = (10, 4))

for ax, (data, title) in zip(axes, [
    (y_train,     'Train set before SMOTE'),
    (y_train_res, 'Train set After SMOTE'),

]):
    counts = pd.Series(data).value_counts().sort_index()
    ax.bar(['Rejected (0)', 'Approved (1)'], counts.values,
           color = ['#E85D04', '#2DC653'], edgecolor = 'white', width = 0.5)
    ax.set_title(title, weight = 'bold')
    ax.set_ylabel('Count')
    for i, v in enumerate(counts.values):
        ax.text(i, v+3, str(v), ha = 'center', fontsize = 11, weight = 'bold')

plt.suptitle('SMOTE applied to Training set only', fontsize = 13, weight = 'bold')
save_show('viz7_smote_balance')
print()


# 3. SCALE - fit on resampled train, transform test
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc = scaler.transform(X_test)

print(f'Train (after SMOTE) : {X_train_res.shape[0]} | Test : {X_test.shape[0]}')
print('Split, SMOTE, and scaling is done')
print()



#S7 - Model Training & Evaluation :
print('-'*60)
print('S7 - Model Training & Evaluation : ')

models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state = 42)),
    ('Decision Tree', DecisionTreeClassifier(max_depth = 10, random_state = 42)),
    ('Random Forest', RandomForestClassifier(n_estimators= 100, random_state = 42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state = 42)),
    ('SVM', SVC(probability = True, random_state = 42)),
]

results = []
roc_data = {}

for name, model in models:
    model.fit(X_train_sc, y_train_res)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    results.append({
        'Model'     : name,
        'Accuracy'  : round(accuracy_score(y_test, y_pred), 4),
        'Precision' : round(precision_score(y_test, y_pred), 4),
        'Recall'    : round(recall_score(y_test, y_pred), 4),
        'F1 Score'  : round(f1_score(y_test, y_pred), 4),
        'ROC-AUC'   : round(roc_auc_score(y_test, y_prob), 4),

    })

    roc_data[name] = (y_prob, model)
    print(f'{name:25s} Acc = {results[-1]['Accuracy']:.3f} | '
          f'F1 = {results[-1]['F1 Score']:.3f} | AUC = {results[-1]['ROC-AUC']:.3f}' )
    
results_df = pd.DataFrame(results)
print('Full Results')
print(results_df.to_string(index = False))
print()


#Viz8 - Model Comparison + ROC curves :
print('viz8 - MOdel Comparison + ROC + Feature Importance + Confusion Matrix')
fig, axes = plt.subplots(1, 2, figsize = (15, 6))

#Grouped Matrics bar :
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
x = np.arange(len(results_df))
w = 0.15
colors = ['#E85D04', '#F48C06', '#2DC653', '#1A6B3C', '#1A1A2E']
for i, (metric, color) in enumerate(zip(metrics, colors)):
    axes[0].bar(x + i*w, results_df[metric], width = w,
                label = metric, color = color, edgecolor = 'white')
    
axes[0].set_xticks(x + 2*w)
axes[0].set_xticklabels(results_df['Model'], rotation = 15, ha = 'right')
axes[0].set_title('All Metrics - Model Comparison', weight = 'bold')
axes[0].set_ylim(0.5, 1.05)
axes[0].legend(fontsize = 9)

#ROC curves:
roc_colors = ['#E85D04','#F48C06','#2DC653','#1A6B3C','#1A1A2E']
for (name, (y_prob, _)), color in zip(roc_data.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = results_df[results_df['Model'] == name]['ROC-AUC'].values[0]
    axes[1].plot(fpr, tpr, color = color, linewidth = 2,
                 label = f'{name} (AUC= {auc_val:.3f})')
axes[1].plot([0,1], [0,1], 'k--', linewidth = 1, label = 'Random')
axes[1].set_title('ROC Curves - All models', weight = 'bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize = 9)
axes[1].grid(True)
save_show('viz8_model_comparison_roc') 
print()

# Feature Importance:
best_rf = [m for n, m in models if n == 'Random Forest'][0]
importances = pd.Series(best_rf.feature_importances_,
                        index = feature_cols).sort_values(ascending = True)
fig, ax = plt.subplots(figsize = (9, 6))
importances.plot(kind = 'barh', ax = ax, color = '#E85D04', edgecolor = 'white')
ax.set_title('Feature Importance - Random Forest', fontsize = 14, weight = 'bold')
ax.set_xlabel('Importance Score')
save_show('viz9_feature_importance')
print()

#Confusion Maarix - best model by F1:
best_name = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
best_model = [m for n, m in models if n == best_name][0]
y_pred_best = best_model.predict(X_test_sc)

fig, ax = plt.subplots(figsize = (6, 5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'YlOrRd', ax = ax,
            xticklabels = ['Rejected (0)', 'Approved (1)'],
            yticklabels = ['Rejected (0)', 'Approved (1)'])
ax.set_title(f'Confusion Matrix - {best_name}', weight = 'bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
save_show('viz10_confusion_matrix')
print()

#S8 - Business Interpretation & Threshold Analysis
print('-'*60)
print('S8 - Business Interpretation & Threshold Analysis')

best_auc_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_auc_model = [m for n, m in models if n == best_auc_name][0]
y_probs = best_auc_model.predict_proba(X_test_sc)[:, 1]

print(f'\nBest Model fy ROC-AUC : {best_auc_name}')
print(f'\nClassification Report (threshold = 0.5) :')
print(classification_report(y_test, best_auc_model.predict(X_test_sc),
                            target_names = ['Rejected (N)', 'Approved (Y)']))

print('Threshold Analysis')
print(f'{"Threshold":>10} {"Precision":>10} {"Recall":>10} '
      f'{"F1":>8} {"Approved%":>10}')
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7] :
    y_t = (y_probs >= thresh).astype(int)
    p = precision_score(y_test, y_t, zero_division = 0)
    r = recall_score(y_test, y_t, zero_division = 0)
    f = f1_score(y_test, y_t, zero_division = 0)
    pct = y_t.mean() * 100
    print(f'{thresh:>10.1f} {p:>10.3f} {r:>10.3f} {f:>8.3f} {pct:>9.1f}%')

print()
print('Final Model Summary :')
print(results_df.to_string(index = False))

best_row = results_df.loc[results_df['ROC-AUC'].idxmax()]
print(f'\nBest Model (ROC-AUC) : {best_row['Model']}')
print(f'  Accuracy  : {best_row['Accuracy'] :.2%}')
print(f'  Precision : {best_row['Precision'] :.2%}')
print(f'  Recall    : {best_row['Recall'] :2%}')
print(f'  F1 Score  : {best_row['F1 Score'] :.4f}')
print(f'  ROC-AUC   : {best_row['ROC-AUC'] :.4f}')

print('-'*60)
print('Auf Wiedersehen')
 
