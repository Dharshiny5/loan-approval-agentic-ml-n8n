import pandas as pd
import numpy as np
import json
import sys
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

### Load Data###
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else 'loan_approval_dataset.csv'
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '.'

df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()  # Strip whitespace from column names
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()  # Strip whitespace from string columns

print(f"[INFO] Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")

### Feature Engineering ###

df['total_assets'] = (
    df['residential_assets_value'] +
    df['commercial_assets_value'] +
    df['luxury_assets_value'] +
    df['bank_asset_value']
)

df['loan_to_income_ratio'] = df['loan_amount'] / (df['income_annum'] + 1)
df['loan_to_assets_ratio'] = df['loan_amount'] / (df['total_assets'] + 1)
df['asset_to_income_ratio'] = df['total_assets'] / (df['income_annum'] + 1)

print("[INFO] Feature engineering complete")

### Encode categorical variables ###

le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_target = LabelEncoder()

df['education_encoded'] = le_education.fit_transform(df['education'])
df['self_employed_encoded'] = le_self_employed.fit_transform(df['self_employed'])
df['loan_status_encoded'] = le_target.fit_transform(df['loan_status'])

class_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
approved_label = class_mapping.get('Approved', 1)

print(f"[INFO] Class mapping: {class_mapping}")

### DEFINE FEATURES & SPLIT ###

FEATURES = [
    'no_of_dependents',
    'education_encoded',
    'self_employed_encoded',
    'income_annum',
    'loan_amount',
    'loan_term',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value',
    'total_assets',
    'loan_to_income_ratio',
    'loan_to_assets_ratio',
    'asset_to_income_ratio'
]

FEATURE_LABELS = [
    'No. of Dependents',
    'Education',
    'Self Employed',
    'Annual Income',
    'Loan Amount',
    'Loan Term',
    'Residential Assets',
    'Commercial Assets',
    'Luxury Assets',
    'Bank Assets',
    'Total Assets',
    'Loan-to-Income Ratio',
    'Loan-to-Assets Ratio',
    'Asset-to-Income Ratio'
]

X = df[FEATURES]
y = df['loan_status_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")


### Train Models ###

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42,
                                   eval_metric='logloss', verbosity=0)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model_object': model,
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob), 4)
    }

    print(f"[INFO] {name} -> AUC: {results[name]['roc_auc']}, F1: {results[name]['f1']}")

### Pick Best Model ###

best_model_name = max(results, key=lambda m: results[m]['roc_auc'])
best_model = results[best_model_name]['model_object']

print(f"[INFO] Best model: {best_model_name} (AUC: {results[best_model_name]['roc_auc']})")

### Generate Predictions ###

all_probs = best_model.predict_proba(X)[:, approved_label]
all_preds = best_model.predict(X)

df['prediction'] = le_target.inverse_transform(all_preds)
df['approval_probability'] = (all_probs * 100).round(2)

df['estimated_eligible_amount'] = df.apply(
    lambda row: round(min(row['loan_amount'],
                          row['income_annum'] * 5 * (row['approval_probability'] / 100)), 2)
    if row['prediction'] == 'Approved' else 0,
    axis=1
)

def risk_tier(prob):
    if prob >= 75:
        return 'Low Risk'
    elif prob >= 50:
        return 'Medium Risk'
    else:
        return 'High Risk'
    
df['risk_tier'] = df['approval_probability'].apply(risk_tier)

### Feature Importance ###

importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': FEATURE_LABELS,
    'importance': importances
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=importance_df, x='importance', y='feature', palette='Blues_r', ax=ax)
ax.set_title(f'Feature Importance — {best_model_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_ylabel('')
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[INFO] Feature importance chart saved to {chart_path}")

model_names = list(results.keys())
auc_scores = [results[m]['roc_auc'] for m in model_names]
f1_scores  = [results[m]['f1'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, auc_scores, width, label='ROC-AUC', color='#2196F3')
bars2 = ax.bar(x + width/2, f1_scores,  width, label='F1 Score', color='#4CAF50')
ax.set_title('Model Comparison — ROC-AUC vs F1 Score', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15)
ax.set_ylim(0, 1.1)
ax.legend()
ax.bar_label(bars1, fmt='%.3f', padding=3)
ax.bar_label(bars2, fmt='%.3f', padding=3)
plt.tight_layout()
model_chart_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
plt.savefig(model_chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[INFO] Model comparison chart saved to {model_chart_path}")

### Highlight Cases ###

approved_df   = df[df['prediction'] == 'Approved'].nlargest(5, 'approval_probability')
rejected_df   = df[df['prediction'] == 'Rejected'].nsmallest(5, 'approval_probability')
borderline_df = df[
    (df['approval_probability'] >= 45) & (df['approval_probability'] <= 55)
].head(5)

def applicant_summary(row):
    return {
        'loan_id': int(row['loan_id']),
        'prediction': row['prediction'],
        'approval_probability_pct': float(row['approval_probability']),
        'risk_tier': row['risk_tier'],
        'cibil_score': int(row['cibil_score']),
        'income_annum': int(row['income_annum']),
        'loan_amount': int(row['loan_amount']),
        'loan_to_income_ratio': round(float(row['loan_to_income_ratio']), 4),
        'total_assets': int(row['total_assets']),
        'estimated_eligible_amount': float(row['estimated_eligible_amount'])
    }

highlight_cases = {
    'top_approved':  [applicant_summary(r) for _, r in approved_df.iterrows()],
    'top_rejected':  [applicant_summary(r) for _, r in rejected_df.iterrows()],
    'borderline':    [applicant_summary(r) for _, r in borderline_df.iterrows()]
}

### Output ###

class_counts = df['loan_status'].value_counts().to_dict()

portfolio_summary = {
    'total_applicants': int(len(df)),
    'total_approved': int((df['prediction'] == 'Approved').sum()),
    'total_rejected': int((df['prediction'] == 'Rejected').sum()),
    'approval_rate_pct': round((df['prediction'] == 'Approved').mean() * 100, 2),
    'avg_approval_probability_approved': round(
        df[df['prediction'] == 'Approved']['approval_probability'].mean(), 2),
    'avg_cibil_approved': round(
        df[df['prediction'] == 'Approved']['cibil_score'].mean(), 2),
    'avg_cibil_rejected': round(
        df[df['prediction'] == 'Rejected']['cibil_score'].mean(), 2),
    'risk_distribution': df['risk_tier'].value_counts().to_dict()
}

model_comparison = {
    name: {k: v for k, v in metrics.items() if k != 'model_object'}
    for name, metrics in results.items()
}

output = {
    'dataset_info': {
        'total_rows': int(df.shape[0]),
        'total_features': len(FEATURES),
        'class_distribution_original': class_counts,
        'engineered_features': [
            'total_assets',
            'loan_to_income_ratio',
            'loan_to_assets_ratio',
            'asset_to_income_ratio'
        ]
    },
    'model_comparison': model_comparison,
    'best_model': {
        'name': best_model_name,
        'roc_auc': results[best_model_name]['roc_auc'],
        'f1': results[best_model_name]['f1'],
        'accuracy': results[best_model_name]['accuracy']
    },
    'feature_importance': importance_df.to_dict(orient='records'),
    'portfolio_summary': portfolio_summary,
    'highlight_cases': highlight_cases,
    'chart_paths': {
        'feature_importance': chart_path,
        'model_comparison': model_chart_path
    }
}

output_json_path = os.path.join(OUTPUT_DIR, 'loan_results.json')
with open(output_json_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"[INFO] Results saved to {output_json_path}")
print("[DONE] ML pipeline complete.")

print("JSON_OUTPUT_START")
print(json.dumps(output))
print("JSON_OUTPUT_END")