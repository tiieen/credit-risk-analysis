import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('results.json', encoding='utf-8') as f:
    r = json.load(f)

plt.rcParams['font.size'] = 11

# Chart 1: default rate by checking account status
d = r['by_checking_status']
labels = [x['status'] for x in d]
vals = [x['default_rate_pct'] for x in d]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(labels, vals, color='#c0392b')
ax.set_xlabel('Ty le vo no (%)')
ax.set_title('Ty le vo no theo tinh trang tai khoan thanh toan (checking account)')
for b, v in zip(bars, vals):
    ax.text(v + 0.5, b.get_y() + b.get_height()/2, f'{v}%', va='center')
plt.tight_layout()
plt.savefig('chart_checking_status.png', dpi=150)
plt.close()

# Chart 2: default rate by purpose
d = r['by_purpose']
labels = [x['purpose'] for x in d]
vals = [x['default_rate_pct'] for x in d]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(labels, vals, color='#2980b9')
ax.invert_yaxis()
ax.set_xlabel('Ty le vo no (%)')
ax.set_title('Ty le vo no theo muc dich vay')
for b, v in zip(bars, vals):
    ax.text(v + 0.5, b.get_y() + b.get_height()/2, f'{v}%', va='center')
plt.tight_layout()
plt.savefig('chart_purpose.png', dpi=150)
plt.close()

# Chart 3: default rate by age group
d = r['by_age_group']
order = ['18-25', '26-35', '36-45', '46-55', '56+']
d_sorted = sorted(d, key=lambda x: order.index(x['age_group']))
labels = [x['age_group'] for x in d_sorted]
vals = [x['default_rate_pct'] for x in d_sorted]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, vals, color='#8e44ad')
ax.set_ylabel('Ty le vo no (%)')
ax.set_title('Ty le vo no theo nhom tuoi')
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v}%', ha='center')
plt.tight_layout()
plt.savefig('chart_age.png', dpi=150)
plt.close()

# Chart 4: feature importance
d = r['feature_importance']
labels = [x['feature'] for x in d]
vals = [x['coefficient'] for x in d]
colors = ['#c0392b' if v > 0 else '#27ae60' for v in vals]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(labels, vals, color=colors)
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('He so hoi quy (chuan hoa)')
ax.set_title('Cac yeu to anh huong den rui ro vo no (Logistic Regression)')
plt.tight_layout()
plt.savefig('chart_feature_importance.png', dpi=150)
plt.close()

print("Charts saved.")
