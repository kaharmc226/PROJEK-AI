import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.simplefilter(action='ignore')
# ---
data = pd.read_csv("medical_insurance.csv")
# ---
data.head()
# ---
data.info()
# ---
data.describe()
# ---
data['charges'].median()
# ---
data['charges'].mean()
# ---
data=data[np.abs(stats.zscore(data['charges'])) < 3]
# ---
data.describe()
# ---
plt.figure(figsize=(30,28))
for i, col in enumerate( ['age','bmi','children','charges']):
    plt.subplot(3, 3, i+1)
    sns.histplot(data = data,
            x = col,
            kde = True,
            bins = 30,
            color = 'blue')

plt.show()
# ---
plt.figure(figsize=(12,9))
for i,col in enumerate(['sex','smoker','region']):
    plt.subplot(3,2,i+1)
    x=data[col].value_counts().reset_index()
    plt.title(col)
    plt.pie(x=x['count'],labels=x[col],autopct="%0.1f%%",colors=sns.color_palette('muted'))
# ---
data_corr= data.copy()
label_encoder = LabelEncoder()
for col in ['sex','smoker','region']:
    data_corr[col] = label_encoder.fit_transform(data_corr[col])
data_corr.head()
# ---
data_corr = data_corr.corr()
# ---
sns.heatmap(data=data_corr,annot=True)
# ---
# sns.pairplot(data=data)
# plt.show()
# ---
# sns.scatterplot(data=data,x=data.charges,y=data.smoker,hue=data.age)
# plt.show()
# ---
# data.groupby('sex')['charges'].median()
# ---
# sns.barplot(data=data,x=data.sex,y=data.charges,estimator=np.mean)
# plt.show()
# ---
# sns.barplot(data=data,x=data.region,y=data.charges,estimator=np.mean)
# plt.show()
# ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
# ---
dum = pd.get_dummies(data[['sex','region','smoker']],dtype=int)
# ---
dum.head()
# ---
data_model = pd.concat([data[['age','bmi','children','charges']],dum],axis=1)
# ---
X=data_model.drop(columns=['charges'])
y=data_model["charges"]
# ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RFR, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings, itertools
warnings.simplefilter(action='ignore')
# ---
splits = {
    '70/30': train_test_split(X, y, test_size=0.30, random_state=7),
    '80/20': train_test_split(X, y, test_size=0.20, random_state=7),
    '90/10': train_test_split(X, y, test_size=0.10, random_state=7),
}
print('Data splits prepared:', list(splits.keys()))
# ---
# Linear Regression has no meaningful hyperparameters to loop —
# we iterate over fit_intercept to keep the pattern consistent.
lr_params = [
    {'fit_intercept': True},
    {'fit_intercept': False},
]

rfr_params = [
    {'n_estimators': 100, 'max_depth': None, 'random_state': 7},
    {'n_estimators': 200, 'max_depth': 5,    'random_state': 7},
    {'n_estimators': 300, 'max_depth': 10,   'random_state': 7},
]

gbr_params = [
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'random_state': 7},
    {'n_estimators': 200, 'learning_rate': 0.10, 'max_depth': 4, 'random_state': 7},
    {'n_estimators': 300, 'learning_rate': 0.15, 'max_depth': 5, 'random_state': 7},
]

print('Parameter grids ready.')
# ---
best_results = {}   # key: split_name -> {model_name: {train_r2, test_r2, params}}

for split_name, (X_tr, X_te, y_tr, y_te) in splits.items():
    best_results[split_name] = {}
    print(f'\n=== Split: {split_name} ===')

    # ── Linear Regression ──
    best_lr_score, best_lr = -1, None
    for p in lr_params:
        m = LinearRegression(**p).fit(X_tr, y_tr)
        sc = r2_score(y_te, m.predict(X_te))
        if sc > best_lr_score:
            best_lr_score, best_lr = sc, (m, p)
    m, p = best_lr
    best_results[split_name]['Linear Regression'] = {
        'train_r2': r2_score(y_tr, m.predict(X_tr)),
        'test_r2':  best_lr_score,
        'params':   p,
        'model':    m
    }
    print(f'  LR  best test R2: {best_lr_score:.4f}  params={p}')

    # ── Random Forest Regressor ──
    best_rfr_score, best_rfr = -1, None
    for p in rfr_params:
        m = RFR(**p).fit(X_tr, y_tr)
        sc = r2_score(y_te, m.predict(X_te))
        if sc > best_rfr_score:
            best_rfr_score, best_rfr = sc, (m, p)
    m, p = best_rfr
    best_results[split_name]['Random Forest'] = {
        'train_r2': r2_score(y_tr, m.predict(X_tr)),
        'test_r2':  best_rfr_score,
        'params':   p,
        'model':    m
    }
    print(f'  RFR best test R2: {best_rfr_score:.4f}  params={p}')

    # ── Gradient Boosting Regressor ──
    best_gbr_score, best_gbr = -1, None
    for p in gbr_params:
        m = GradientBoostingRegressor(**p).fit(X_tr, y_tr)
        sc = r2_score(y_te, m.predict(X_te))
        if sc > best_gbr_score:
            best_gbr_score, best_gbr = sc, (m, p)
    m, p = best_gbr
    best_results[split_name]['Gradient Boosting'] = {
        'train_r2': r2_score(y_tr, m.predict(X_tr)),
        'test_r2':  best_gbr_score,
        'params':   p,
        'model':    m
    }
    print(f'  GBR best test R2: {best_gbr_score:.4f}  params={p}')

print('\nDone.')
# ---
model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
split_names  = ['70/30', '80/20', '90/10']

matrix_test = pd.DataFrame(
    index=model_names, columns=split_names, dtype=float
)
for sp in split_names:
    for mn in model_names:
        matrix_test.loc[mn, sp] = round(best_results[sp][mn]['test_r2'], 4)

print('=== Test R² Matrix (rows=Models, cols=Splits) ===')
matrix_test
# ---
matrix_train = pd.DataFrame(
    index=model_names, columns=split_names, dtype=float
)
for sp in split_names:
    for mn in model_names:
        matrix_train.loc[mn, sp] = round(best_results[sp][mn]['train_r2'], 4)

print('=== Train R² Matrix ===')
matrix_train
# ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.heatmap(matrix_test.astype(float), annot=True, fmt='.4f',
            cmap='YlGnBu', vmin=0, vmax=1, ax=axes[0],
            linewidths=0.5, linecolor='white')
axes[0].set_title('Test R² — Best Model per Split', fontsize=13)
axes[0].set_xlabel('Data Split')
axes[0].set_ylabel('Model')

sns.heatmap(matrix_train.astype(float), annot=True, fmt='.4f',
            cmap='YlOrRd', vmin=0, vmax=1, ax=axes[1],
            linewidths=0.5, linecolor='white')
axes[1].set_title('Train R² — Best Model per Split', fontsize=13)
axes[1].set_xlabel('Data Split')
axes[1].set_ylabel('Model')

plt.tight_layout()
plt.show()
# ---
rows = []
for sp in split_names:
    for mn in model_names:
        r = best_results[sp][mn]
        rows.append({'Split': sp, 'Model': mn,
                     'Best Params': str(r['params']),
                     'Train R2': round(r['train_r2'], 4),
                     'Test R2':  round(r['test_r2'],  4)})

summary_df = pd.DataFrame(rows)
summary_df
# ---
