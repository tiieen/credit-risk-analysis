"""
Phan tich rui ro tin dung - German Credit Dataset
Du an DA: Credit Risk Analysis & Default Prediction
"""
import pandas as pd
import numpy as np
import json
import os

os.makedirs('charts', exist_ok=True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

pd.set_option('display.max_columns', None)

df = pd.read_csv('GermanCredit.csv')

# ---------- 1. Data cleaning / feature engineering ----------
# credit_risk: 1 = tra no dung han (good), 0 = vo no (default/bad)
df['default'] = 1 - df['credit_risk']  # 1 = default, de de doc hon khi phan tich rui ro

# Nhom tuoi
bins = [18, 25, 35, 45, 55, 100]
labels = ['18-25', '26-35', '36-45', '46-55', '56+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

# Nhom so tien vay
df['amount_group'] = pd.qcut(df['amount'], 4, labels=['Thap', 'Trung binh', 'Kha cao', 'Cao'])

# Gioi tinh tu personal_status_sex
def get_sex(x):
    return 'female' if 'female' in x else 'male'
df['sex'] = df['personal_status_sex'].apply(get_sex)

results = {}

# ---------- 2. EDA: cac chi so nghiep vu ----------
results['overall_default_rate'] = round(df['default'].mean() * 100, 2)
results['total_records'] = len(df)

def default_rate_by(col):
    g = df.groupby(col, observed=True)['default'].agg(['mean', 'count']).reset_index()
    g['mean'] = (g['mean'] * 100).round(2)
    g.columns = [col, 'default_rate_pct', 'count']
    return g.sort_values('default_rate_pct', ascending=False).to_dict('records')

results['by_age_group'] = default_rate_by('age_group')
results['by_purpose'] = default_rate_by('purpose')
results['by_housing'] = default_rate_by('housing')
results['by_job'] = default_rate_by('job')
results['by_amount_group'] = default_rate_by('amount_group')
results['by_sex'] = default_rate_by('sex')
results['by_savings'] = default_rate_by('savings')
results['by_checking_status'] = default_rate_by('status')

# Amount distribution stats
results['amount_stats'] = {
    'mean': round(df['amount'].mean(), 0),
    'median': round(df['amount'].median(), 0),
    'default_avg_amount': round(df[df['default']==1]['amount'].mean(), 0),
    'good_avg_amount': round(df[df['default']==0]['amount'].mean(), 0),
}

results['duration_stats'] = {
    'default_avg_duration': round(df[df['default']==1]['duration'].mean(), 1),
    'good_avg_duration': round(df[df['default']==0]['duration'].mean(), 1),
}

# ---------- 3. Model: Logistic Regression ----------
model_df = df.drop(columns=['credit_risk', 'age_group', 'amount_group', 'personal_status_sex'])
cat_cols = model_df.select_dtypes(include='object').columns.tolist()

le_dict = {}
model_encoded = model_df.copy()
for c in cat_cols:
    le = LabelEncoder()
    model_encoded[c] = le.fit_transform(model_encoded[c])
    le_dict[c] = le

X = model_encoded.drop(columns=['default'])
y = model_encoded['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
y_prob = clf.predict_proba(X_test_s)[:, 1]

auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred).tolist()

results['model'] = {
    'auc': round(auc, 3),
    'accuracy': round(report['accuracy'], 3),
    'precision_default': round(report['1']['precision'], 3),
    'recall_default': round(report['1']['recall'], 3),
    'confusion_matrix': cm,
    'test_size': len(y_test),
}

# Feature importance (coef magnitude, standardized)
coefs = pd.Series(clf.coef_[0], index=X.columns)
top_features = coefs.abs().sort_values(ascending=False).head(10)
feat_importance = []
for feat in top_features.index:
    feat_importance.append({
        'feature': feat,
        'coefficient': round(coefs[feat], 3),
        'direction': 'Tang rui ro' if coefs[feat] > 0 else 'Giam rui ro'
    })
results['feature_importance'] = feat_importance

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print("DONE. Overall default rate:", results['overall_default_rate'], "%")
print("Model AUC:", results['model']['auc'])
print(json.dumps(feat_importance, ensure_ascii=False, indent=2))
# ---------- 4. Save charts as PNG ----------

# Chart 1: Default rate by checking account status
status_chart = (
    df.groupby('status', observed=True)['default']
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    x=status_chart.values,
    y=status_chart.index,
    color='#c0392b',
    ax=ax
)

ax.set_title('Ty le vo no theo tinh trang tai khoan thanh toan')
ax.set_xlabel('Ty le vo no (%)')
ax.set_ylabel('Tinh trang tai khoan')

plt.tight_layout()
plt.savefig(
    'charts/01_default_rate_by_status.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()


# Chart 2: Default rate by age group
age_chart = (
    df.groupby('age_group', observed=True)['default']
    .mean()
    .mul(100)
)

age_order = ['18-25', '26-35', '36-45', '46-55', '56+']
age_chart = age_chart.reindex(age_order)

fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    x=age_chart.index,
    y=age_chart.values,
    color='#8e44ad',
    ax=ax
)

ax.set_title('Ty le vo no theo nhom tuoi')
ax.set_xlabel('Nhom tuoi')
ax.set_ylabel('Ty le vo no (%)')

plt.tight_layout()
plt.savefig(
    'charts/02_default_rate_by_age.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()


# Chart 3: Default rate by loan purpose
purpose_chart = (
    df.groupby('purpose', observed=True)['default']
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 7))

sns.barplot(
    x=purpose_chart.values,
    y=purpose_chart.index,
    color='#2980b9',
    ax=ax
)

ax.set_title('Ty le vo no theo muc dich vay')
ax.set_xlabel('Ty le vo no (%)')
ax.set_ylabel('Muc dich vay')

plt.tight_layout()
plt.savefig(
    'charts/03_default_rate_by_purpose.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()


# Chart 4: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    fpr,
    tpr,
    label=f'Logistic Regression (AUC = {auc:.3f})'
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    color='gray',
    label='Random classifier'
)

ax.set_title('ROC Curve - Du doan vo no')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(
    'charts/04_roc_curve.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()


# Chart 5: Top Logistic Regression coefficients
top10_coefs = (
    coefs.reindex(top_features.index)
    .sort_values()
)

bar_colors = [
    '#c0392b' if value > 0 else '#27ae60'
    for value in top10_coefs.values
]

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(
    top10_coefs.index,
    top10_coefs.values,
    color=bar_colors
)

ax.axvline(
    0,
    color='black',
    linewidth=0.8
)

ax.set_title('Top 10 he so anh huong - Logistic Regression')
ax.set_xlabel('He so mo hinh')
ax.set_ylabel('Bien')

plt.tight_layout()
plt.savefig(
    'charts/05_logistic_regression_coefficients.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()


# Chart 6: Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['Du doan tot', 'Du doan vo no'],
    yticklabels=['Thuc te tot', 'Thuc te vo no'],
    ax=ax
)

ax.set_title('Confusion Matrix')
ax.set_xlabel('Gia tri du doan')
ax.set_ylabel('Gia tri thuc te')

plt.tight_layout()
plt.savefig(
    'charts/06_confusion_matrix.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()


print("Da luu cac bieu do PNG vao thu muc charts.")
