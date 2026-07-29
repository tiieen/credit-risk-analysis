"""
Phan tich rui ro tin dung - German Credit Dataset
Du an DA: Credit Risk Analysis & Default Prediction
"""
import pandas as pd
import numpy as np
import json
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
