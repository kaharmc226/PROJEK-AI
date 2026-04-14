"""
Comprehensive optimization of insdata100k_tuned_gpu_optimized.ipynb.

Improvements:
 1. Consolidate all imports into one cell; remove duplicates
 2. Add clear section markdown headers throughout
 3. Handle alcohol_freq missing values
 4. Add data.shape printout after outlier removal
 5. Add before/after outlier removal distribution plot
 6. Replace unreadable 53-feature correlation heatmap with top-15
 7. Add more categorical feature pie charts
 8. Add predicted vs actual scatter plots
 9. Add residual distribution plots
10. Add feature importance bar chart
11. Add MAE/RMSE metrics alongside R-squared
12. Save best model with joblib
"""

import json, os, copy

INPUT  = os.path.join(os.path.dirname(__file__), "insdata100k_tuned_gpu_optimized.ipynb")
OUTPUT = os.path.join(os.path.dirname(__file__), "insdata100k_tuned_gpu_v2.ipynb")


# ── Helpers ────────────────────────────────────────────────────────────────────

def md(source: str, cell_id: str = "") -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if "\n" not in source else [l + "\n" for l in source.split("\n")],
        "id": cell_id or f"md_{id(source)}"
    }

def code(source: str, cell_id: str = "") -> dict:
    """Create a code cell."""
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]  # last line no trailing \n
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "source": src,
        "outputs": [],
        "execution_count": None,
        "id": cell_id or f"code_{id(source)}"
    }

def get_src(cell: dict) -> str:
    """Get source text from cell."""
    return "".join(cell.get("source", []))


# ── Build new notebook ─────────────────────────────────────────────────────────

def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        nb = json.load(f)

    old_cells = nb["cells"]
    new_cells = []

    # ── SECTION 0: GPU Setup (keep cell 0 as-is) ──────────────────────────────
    new_cells.append(old_cells[0])  # GPU instructions markdown

    # ── SECTION 1: Imports ─────────────────────────────────────────────────────
    new_cells.append(md("# 1. Imports", "sec1_hdr"))
    new_cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RFR_sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings, itertools
warnings.simplefilter(action='ignore')""", "imports_cell"))

    # ── SECTION 2: Data Loading ────────────────────────────────────────────────
    new_cells.append(md("# 2. Data Loading", "sec2_hdr"))

    # Keep original data-loading cell (cell 3)
    new_cells.append(old_cells[3])  # pip install, kaggle download, pd.read_csv

    new_cells.append(code("data.head()", "data_head"))
    new_cells.append(code("data.info()", "data_info"))
    new_cells.append(code("data.shape", "data_shape_raw"))

    # ── NEW: Missing data handling ─────────────────────────────────────────────
    new_cells.append(md("## 2.1 Handling Missing Values", "sec2_1_hdr"))
    new_cells.append(code("""\
# Check missing values
print("Missing values per column:")
print(data.isnull().sum()[data.isnull().sum() > 0])
print(f"\\nTotal rows with missing values: {data.isnull().any(axis=1).sum()}")

# Fill missing alcohol_freq with 'Unknown'
data['alcohol_freq'] = data['alcohol_freq'].fillna('Unknown')
print(f"\\nalcohol_freq missing after fill: {data['alcohol_freq'].isnull().sum()}")""",
        "handle_missing"))

    # ── SECTION 3: EDA ─────────────────────────────────────────────────────────
    new_cells.append(md("# 3. Exploratory Data Analysis (EDA)", "sec3_hdr"))

    new_cells.append(md("## 3.1 Descriptive Statistics", "sec3_1_hdr"))
    new_cells.append(code("data.describe()", "describe_raw"))

    new_cells.append(md("## 3.2 Target Variable Distribution (Before Cleaning)", "sec3_2_hdr"))
    new_cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(data['annual_medical_cost'], kde=True, bins=50, color='steelblue', ax=axes[0])
axes[0].set_title('Distribution of Annual Medical Cost', fontsize=13)
axes[0].axvline(data['annual_medical_cost'].mean(), color='red', linestyle='--', label=f"Mean: {data['annual_medical_cost'].mean():.2f}")
axes[0].axvline(data['annual_medical_cost'].median(), color='green', linestyle='--', label=f"Median: {data['annual_medical_cost'].median():.2f}")
axes[0].legend()

sns.boxplot(y=data['annual_medical_cost'], color='steelblue', ax=axes[1])
axes[1].set_title('Box Plot of Annual Medical Cost', fontsize=13)

plt.tight_layout()
plt.show()""", "target_dist_before"))

    new_cells.append(md("## 3.3 Annual Medical Cost by Key Categories", "sec3_3_hdr"))
    # Keep original boxplots cell (cell 7)
    new_cells.append(old_cells[7])

    new_cells.append(md("## 3.4 Statistical Test: Smoker vs Non-Smoker", "sec3_4_hdr"))
    # Keep original t-test cell (cell 9)
    new_cells.append(old_cells[9])

    # ── SECTION 4: Outlier Removal ─────────────────────────────────────────────
    new_cells.append(md("# 4. Outlier Detection & Removal", "sec4_hdr"))

    new_cells.append(code("""\
print(f"Median annual_medical_cost: {data['annual_medical_cost'].median():.2f}")
print(f"Mean annual_medical_cost:   {data['annual_medical_cost'].mean():.2f}")""",
        "median_mean"))

    new_cells.append(code("""\
# Remove outliers using z-score method (threshold = 3)
data_cleaned = data[np.abs(stats.zscore(data['annual_medical_cost'])) < 3]

rows_removed = len(data) - len(data_cleaned)
print(f"Rows before:   {len(data):,}")
print(f"Rows after:    {len(data_cleaned):,}")
print(f"Rows removed:  {rows_removed:,} ({rows_removed/len(data)*100:.1f}%)")""",
        "outlier_removal"))

    new_cells.append(md("## 4.1 Before vs After Outlier Removal", "sec4_1_hdr"))
    new_cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(data['annual_medical_cost'], kde=True, bins=50, color='salmon', ax=axes[0])
axes[0].set_title('BEFORE Outlier Removal', fontsize=13)
axes[0].set_xlabel('Annual Medical Cost')

sns.histplot(data_cleaned['annual_medical_cost'], kde=True, bins=50, color='seagreen', ax=axes[1])
axes[1].set_title('AFTER Outlier Removal', fontsize=13)
axes[1].set_xlabel('Annual Medical Cost')

plt.suptitle('Effect of Outlier Removal on Target Distribution', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()""", "before_after_outlier"))

    new_cells.append(code("data_cleaned.describe()", "describe_cleaned"))

    # ── SECTION 5: Post-Cleaning Visualizations ────────────────────────────────
    new_cells.append(md("# 5. Post-Cleaning Visualizations", "sec5_hdr"))

    new_cells.append(md("## 5.1 Numeric Feature Distributions", "sec5_1_hdr"))
    new_cells.append(code("""\
numeric_cols = ['age', 'bmi', 'income', 'dependents', 'annual_medical_cost',
                'visits_last_year', 'medication_count']
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

plt.figure(figsize=(18, 5 * n_rows))
for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(data=data_cleaned, x=col, kde=True, bins=30, color='steelblue')
    plt.title(f'Distribution of {col}', fontsize=12)
plt.tight_layout()
plt.show()""", "numeric_dists"))

    new_cells.append(md("## 5.2 Categorical Feature Distributions", "sec5_2_hdr"))
    new_cells.append(code("""\
cat_cols = ['sex', 'smoker', 'region', 'urban_rural', 'education',
            'marital_status', 'employment_status', 'plan_type', 'network_tier']
n_cols_plot = 3
n_rows_plot = (len(cat_cols) + n_cols_plot - 1) // n_cols_plot

plt.figure(figsize=(18, 5 * n_rows_plot))
for i, col in enumerate(cat_cols):
    plt.subplot(n_rows_plot, n_cols_plot, i + 1)
    x = data_cleaned[col].value_counts().reset_index()
    plt.pie(x=x['count'], labels=x[col], autopct="%0.1f%%",
            colors=sns.color_palette('muted'), textprops={'fontsize': 9})
    plt.title(col, fontsize=12)
plt.suptitle('Categorical Feature Distributions', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()""", "cat_dists"))

    new_cells.append(md("## 5.3 Correlation Analysis (Top Features)", "sec5_3_hdr"))
    new_cells.append(code("""\
# Encode categorical columns for correlation
data_corr = data_cleaned.copy()
label_encoder = LabelEncoder()
for col in data_corr.select_dtypes(include='object').columns:
    data_corr[col] = label_encoder.fit_transform(data_corr[col])

corr_matrix = data_corr.corr()

# Show top 15 features most correlated with the target
target_corr = corr_matrix['annual_medical_cost'].drop('annual_medical_cost').abs().sort_values(ascending=False)
top_features = target_corr.head(15).index.tolist() + ['annual_medical_cost']

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix.loc[top_features, top_features], annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5, linecolor='white')
plt.title('Top 15 Features Correlated with Annual Medical Cost', fontsize=14)
plt.tight_layout()
plt.show()

print("\\nTop 15 correlations with annual_medical_cost:")
print(target_corr.head(15).to_string())""", "correlation_top15"))

    # ── SECTION 6: Feature Engineering ─────────────────────────────────────────
    new_cells.append(md("# 6. Feature Engineering & Encoding", "sec6_hdr"))
    new_cells.append(code("""\
# One-hot encode categorical features
dum = pd.get_dummies(data_cleaned.select_dtypes(include='object'), drop_first=True, dtype=int)
print(f"Encoded features shape: {dum.shape}")
dum.head()""", "one_hot"))

    new_cells.append(code("""\
# Combine numeric and encoded features
data_model = pd.concat([data_cleaned.select_dtypes(exclude='object'), dum], axis=1)
print(f"Final feature matrix shape: {data_model.shape}")
data_model.head()""", "combine_features"))

    new_cells.append(code("""\
X = data_model.drop(columns=['annual_medical_cost'])
y = data_model['annual_medical_cost']
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")""", "xy_split"))

    # ── SECTION 7: Model Training ──────────────────────────────────────────────
    new_cells.append(md("# 7. Model Training", "sec7_hdr"))

    new_cells.append(md("## 7.1 GPU-Accelerated Imports", "sec7_1_hdr"))
    new_cells.append(code("""\
# GPU-accelerated models (for Colab/Kaggle with GPU runtime)
try:
    from cuml.ensemble import RandomForestRegressor as RFR
    print("Using cuML GPU-accelerated RandomForestRegressor")
except ImportError:
    from sklearn.ensemble import RandomForestRegressor as RFR
    print("cuML not available, using sklearn RandomForestRegressor (CPU)")

from xgboost import XGBRegressor
print("XGBRegressor ready (GPU via tree_method='hist', device='cuda')")""", "gpu_imports"))

    new_cells.append(md("## 7.2 Data Splits: 70/30, 80/20, 90/10", "sec7_2_hdr"))
    new_cells.append(code("""\
splits = {
    '70/30': train_test_split(X, y, test_size=0.30, random_state=7),
    '80/20': train_test_split(X, y, test_size=0.20, random_state=7),
    '90/10': train_test_split(X, y, test_size=0.10, random_state=7),
}
for name, (X_tr, X_te, _, _) in splits.items():
    print(f"  {name}: train={X_tr.shape[0]:,}, test={X_te.shape[0]:,}")""", "data_splits"))

    new_cells.append(md("## 7.3 Hyperparameter Grids", "sec7_3_hdr"))
    # Keep original param grids (cell 34)
    new_cells.append(old_cells[34])

    new_cells.append(md("## 7.4 Training Loop", "sec7_4_hdr"))
    new_cells.append(code("""\
best_results = {}

for split_name, (X_tr, X_te, y_tr, y_te) in splits.items():
    best_results[split_name] = {}
    print(f'\\n{"="*50}')
    print(f'Split: {split_name}')
    print(f'{"="*50}')

    # -- Linear Regression --
    best_lr_score, best_lr = -1, None
    for p in lr_params:
        m = LinearRegression(**p).fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        sc = r2_score(y_te, y_pred)
        if sc > best_lr_score:
            best_lr_score, best_lr = sc, (m, p)
    m, p = best_lr
    y_pred_te = m.predict(X_te)
    y_pred_tr = m.predict(X_tr)
    best_results[split_name]['Linear Regression'] = {
        'train_r2': r2_score(y_tr, y_pred_tr),
        'test_r2':  best_lr_score,
        'test_mae': mean_absolute_error(y_te, y_pred_te),
        'test_rmse': mean_squared_error(y_te, y_pred_te, squared=False),
        'params': p, 'model': m
    }
    print(f'  LR   R2={best_lr_score:.4f}  MAE={best_results[split_name]["Linear Regression"]["test_mae"]:.2f}  RMSE={best_results[split_name]["Linear Regression"]["test_rmse"]:.2f}')

    # -- Random Forest Regressor --
    best_rfr_score, best_rfr = -1, None
    for p in rfr_params:
        m = RFR(**p).fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        sc = r2_score(y_te, y_pred)
        if sc > best_rfr_score:
            best_rfr_score, best_rfr = sc, (m, p)
    m, p = best_rfr
    y_pred_te = m.predict(X_te)
    y_pred_tr = m.predict(X_tr)
    best_results[split_name]['Random Forest'] = {
        'train_r2': r2_score(y_tr, y_pred_tr),
        'test_r2':  best_rfr_score,
        'test_mae': mean_absolute_error(y_te, y_pred_te),
        'test_rmse': mean_squared_error(y_te, y_pred_te, squared=False),
        'params': p, 'model': m
    }
    print(f'  RFR  R2={best_rfr_score:.4f}  MAE={best_results[split_name]["Random Forest"]["test_mae"]:.2f}  RMSE={best_results[split_name]["Random Forest"]["test_rmse"]:.2f}')

    # -- XGBoost Regressor --
    best_gbr_score, best_gbr = -1, None
    for p in gbr_params:
        m = XGBRegressor(tree_method="hist", device="cuda", **p).fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        sc = r2_score(y_te, y_pred)
        if sc > best_gbr_score:
            best_gbr_score, best_gbr = sc, (m, p)
    m, p = best_gbr
    y_pred_te = m.predict(X_te)
    y_pred_tr = m.predict(X_tr)
    best_results[split_name]['Gradient Boosting'] = {
        'train_r2': r2_score(y_tr, y_pred_tr),
        'test_r2':  best_gbr_score,
        'test_mae': mean_absolute_error(y_te, y_pred_te),
        'test_rmse': mean_squared_error(y_te, y_pred_te, squared=False),
        'params': p, 'model': m
    }
    print(f'  GBR  R2={best_gbr_score:.4f}  MAE={best_results[split_name]["Gradient Boosting"]["test_mae"]:.2f}  RMSE={best_results[split_name]["Gradient Boosting"]["test_rmse"]:.2f}')

print('\\nTraining complete.')""", "training_loop"))

    # ── SECTION 8: Results & Evaluation ────────────────────────────────────────
    new_cells.append(md("# 8. Results & Evaluation", "sec8_hdr"))

    new_cells.append(md("## 8.1 Comparison Matrices", "sec8_1_hdr"))
    new_cells.append(code("""\
model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
split_names = ['70/30', '80/20', '90/10']

# R-squared matrix
matrix_test = pd.DataFrame(index=model_names, columns=split_names, dtype=float)
matrix_train = pd.DataFrame(index=model_names, columns=split_names, dtype=float)
matrix_mae = pd.DataFrame(index=model_names, columns=split_names, dtype=float)
matrix_rmse = pd.DataFrame(index=model_names, columns=split_names, dtype=float)

for sp in split_names:
    for mn in model_names:
        r = best_results[sp][mn]
        matrix_test.loc[mn, sp] = round(r['test_r2'], 4)
        matrix_train.loc[mn, sp] = round(r['train_r2'], 4)
        matrix_mae.loc[mn, sp] = round(r['test_mae'], 2)
        matrix_rmse.loc[mn, sp] = round(r['test_rmse'], 2)

print('=== Test R-squared Matrix ===')
display(matrix_test)
print('\\n=== Test MAE Matrix ===')
display(matrix_mae)
print('\\n=== Test RMSE Matrix ===')
display(matrix_rmse)""", "comparison_matrices"))

    new_cells.append(md("## 8.2 Heatmap Visualizations", "sec8_2_hdr"))
    new_cells.append(code("""\
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.heatmap(matrix_test.astype(float), annot=True, fmt='.4f',
            cmap='YlGnBu', vmin=0, vmax=1, ax=axes[0, 0],
            linewidths=0.5, linecolor='white')
axes[0, 0].set_title('Test R-squared', fontsize=13)

sns.heatmap(matrix_train.astype(float), annot=True, fmt='.4f',
            cmap='YlOrRd', vmin=0, vmax=1, ax=axes[0, 1],
            linewidths=0.5, linecolor='white')
axes[0, 1].set_title('Train R-squared', fontsize=13)

sns.heatmap(matrix_mae.astype(float), annot=True, fmt='.2f',
            cmap='Blues', ax=axes[1, 0],
            linewidths=0.5, linecolor='white')
axes[1, 0].set_title('Test MAE (lower is better)', fontsize=13)

sns.heatmap(matrix_rmse.astype(float), annot=True, fmt='.2f',
            cmap='Reds', ax=axes[1, 1],
            linewidths=0.5, linecolor='white')
axes[1, 1].set_title('Test RMSE (lower is better)', fontsize=13)

for ax in axes.flat:
    ax.set_xlabel('Data Split')
    ax.set_ylabel('Model')

plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""", "heatmap_viz"))

    new_cells.append(md("## 8.3 Full Results Summary", "sec8_3_hdr"))
    new_cells.append(code("""\
rows = []
for sp in split_names:
    for mn in model_names:
        r = best_results[sp][mn]
        rows.append({
            'Split': sp, 'Model': mn,
            'Best Params': str(r['params']),
            'Train R2': round(r['train_r2'], 4),
            'Test R2':  round(r['test_r2'],  4),
            'Test MAE': round(r['test_mae'], 2),
            'Test RMSE': round(r['test_rmse'], 2),
        })
summary_df = pd.DataFrame(rows)
summary_df""", "results_summary"))

    # ── SECTION 9: Model Diagnostics ───────────────────────────────────────────
    new_cells.append(md("# 9. Model Diagnostics", "sec9_hdr"))

    new_cells.append(md("## 9.1 Predicted vs Actual (Best Model per Split)", "sec9_1_hdr"))
    new_cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, split_name in enumerate(split_names):
    models = best_results[split_name]
    best_model_name = max(models, key=lambda k: models[k]['test_r2'])
    best = models[best_model_name]
    X_tr, X_te, y_tr, y_te = splits[split_name]
    y_pred = best['model'].predict(X_te)

    axes[idx].scatter(y_te, y_pred, alpha=0.2, s=5, color='steelblue')
    min_val = min(y_te.min(), y_pred.min())
    max_val = max(y_te.max(), y_pred.max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    axes[idx].set_title(f'{split_name}: {best_model_name}\\nR2={best["test_r2"]:.4f}', fontsize=12)
    axes[idx].set_xlabel('Actual')
    axes[idx].set_ylabel('Predicted')
    axes[idx].legend()

plt.suptitle('Predicted vs Actual Values', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()""", "pred_vs_actual"))

    new_cells.append(md("## 9.2 Residual Analysis", "sec9_2_hdr"))
    new_cells.append(code("""\
# Use the best overall model for residual analysis
best_split = max(best_results, key=lambda s: max(m['test_r2'] for m in best_results[s].values()))
best_mn = max(best_results[best_split], key=lambda m: best_results[best_split][m]['test_r2'])
best_model_info = best_results[best_split][best_mn]

X_tr, X_te, y_tr, y_te = splits[best_split]
y_pred = best_model_info['model'].predict(X_te)
residuals = y_te - y_pred

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(residuals, kde=True, bins=50, color='steelblue', ax=axes[0])
axes[0].axvline(x=0, color='red', linestyle='--')
axes[0].set_title('Residual Distribution', fontsize=13)
axes[0].set_xlabel('Residual (Actual - Predicted)')

axes[1].scatter(y_pred, residuals, alpha=0.2, s=5, color='steelblue')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Residuals vs Predicted Values', fontsize=13)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Residual')

plt.suptitle(f'Residual Analysis: {best_mn} ({best_split})', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

print(f"Mean residual:   {residuals.mean():.4f}")
print(f"Std residual:    {residuals.std():.4f}")""", "residuals"))

    new_cells.append(md("## 9.3 Feature Importance", "sec9_3_hdr"))
    new_cells.append(code("""\
# Feature importance from the best model (XGBoost or Random Forest)
if hasattr(best_model_info['model'], 'feature_importances_'):
    importances = best_model_info['model'].feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

    plt.figure(figsize=(12, max(8, len(feat_imp) * 0.25)))
    feat_imp.tail(20).plot(kind='barh', color='steelblue')
    plt.title(f'Top 20 Feature Importances ({best_mn})', fontsize=14)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    print("\\nTop 10 most important features:")
    print(feat_imp.sort_values(ascending=False).head(10).to_string())
else:
    print(f"{best_mn} does not provide feature importances.")""", "feature_importance"))

    # ── SECTION 10: Save Best Model ────────────────────────────────────────────
    new_cells.append(md("# 10. Save Best Model", "sec10_hdr"))
    new_cells.append(code("""\
# Identify and save the overall best model
best_split = max(best_results, key=lambda s: max(m['test_r2'] for m in best_results[s].values()))
best_mn = max(best_results[best_split], key=lambda m: best_results[best_split][m]['test_r2'])
best_model = best_results[best_split][best_mn]['model']
best_r2 = best_results[best_split][best_mn]['test_r2']

model_path = 'best_model.pkl'
joblib.dump(best_model, model_path)
print(f"Best model saved: {best_mn}")
print(f"  Split:    {best_split}")
print(f"  Test R2:  {best_r2:.4f}")
print(f"  Test MAE: {best_results[best_split][best_mn]['test_mae']:.2f}")
print(f"  Test RMSE:{best_results[best_split][best_mn]['test_rmse']:.2f}")
print(f"  Params:   {best_results[best_split][best_mn]['params']}")
print(f"  Saved to: {model_path}")""", "save_model"))

    # ── Write output ───────────────────────────────────────────────────────────
    nb["cells"] = new_cells
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"Optimized notebook written to: {OUTPUT}")
    print(f"Total cells: {len(new_cells)} (was {len(old_cells)})")


if __name__ == "__main__":
    main()
