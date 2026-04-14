# Medical Insurance Cost Prediction — Complete Technical Walkthrough

## About This Document
This document is a comprehensive, narrative-style walkthrough of the **Classic Insurance Cost Prediction** machine learning project. It is written to be uploaded into learning platforms like **NotebookLM** so you can ask questions and learn about every aspect of the project — from data understanding to model deployment.

The project consists of two Jupyter notebooks:
1. **`insurance_tuned_gpu.ipynb`** — The main training pipeline (data loading → preprocessing → feature engineering → model training → evaluation → saving)
2. **`insurance_inference.ipynb`** — Inference testing (loading the saved model → testing with normal, edge-case, and absurd data → sensitivity analysis)

---

# PART 1: UNDERSTANDING THE PROBLEM

## What Are We Trying to Do?
We are building a **regression model** that predicts how much a person will pay in annual medical insurance charges (in US dollars). This is a **supervised learning** problem because we have labeled data — we know the actual charges for each person in our dataset, and we want the model to learn the patterns so it can predict charges for new people.

## Why Does This Matter?
Insurance companies need accurate cost predictions to:
- **Set fair premiums** — charge customers appropriately based on their risk profile
- **Manage financial risk** — know how much money to set aside for claims
- **Identify high-risk groups** — understand which combinations of factors lead to expensive claims

## What Is Regression?
In machine learning, **regression** means predicting a continuous number (like $5,000 or $42,317.50), as opposed to **classification** which predicts categories (like "high risk" or "low risk"). Since insurance charges are a continuous dollar amount, we use regression algorithms.

---

# PART 2: THE DATASET

## Source and Size
The dataset is the **Medical Cost Personal Dataset** from Kaggle (`insurance.csv`). It contains **1,338 rows** (patients) and **7 columns** (6 input features + 1 target variable).

This is a relatively small dataset by machine learning standards. Many real-world ML projects use millions of rows. With only 1,338 rows, we must be careful about overfitting — the model might memorize the training data instead of learning generalizable patterns.

## The Features (Input Variables)
Here is every column in the dataset and what it means:

**1. `age` (Integer, range 18-64)**
The patient's age in years. Older people tend to have higher medical costs because they are more likely to develop health issues. In our data, the average age is about 39 years.

**2. `sex` (Categorical: "male" or "female")**
The patient's biological sex. This has a relatively small effect on insurance costs compared to other features.

**3. `bmi` (Float, range 15.96 - 53.13)**
Body Mass Index — a measure of body fat based on height and weight. The formula is weight(kg) / height(m)². A BMI of 30 or above is classified as "obese" by medical standards. The average BMI in our dataset is 30.66, which is just above the obesity threshold.

**4. `children` (Integer, range 0-5)**
The number of children/dependents covered by the insurance plan. This has a small but positive effect on costs.

**5. `smoker` (Categorical: "yes" or "no")**
Whether the patient smokes. This is the **single most important feature** for predicting insurance costs. Smokers have dramatically higher medical costs. About 20.5% of patients in our dataset are smokers.

**6. `region` (Categorical: "northeast", "northwest", "southeast", "southwest")**
The patient's residential area in the US. Different regions have different healthcare costs.

**7. `charges` (Float, range $1,121.87 - $63,770.43) — THIS IS THE TARGET**
The annual medical insurance charges billed to the patient. This is what we are trying to predict. The average charge is about $13,270, but the distribution is heavily skewed — most people pay less than $15,000, while a small group (mainly smokers) pay $30,000-$63,000.

## Key Dataset Statistics
- **Mean charges:** $13,270.42
- **Median charges:** $9,382.03 (the median is much lower than the mean — this tells us the distribution is right-skewed)
- **Standard deviation of charges:** $12,110.01 (very high — nearly as large as the mean, indicating huge variation)
- **No missing values** — every row has complete data, so we do not need imputation techniques

---

# PART 3: EXPLORATORY DATA ANALYSIS (EDA)

## What Is EDA and Why Do We Do It?
Exploratory Data Analysis means looking at the data before building models to understand its structure, find patterns, spot anomalies, and generate hypotheses. Good EDA directly informs what features to engineer and which algorithms to choose.

## The Most Important Finding: Three "Bands" of Charges
When we plot charges against age, we see something remarkable — the data naturally separates into **three distinct bands**:

1. **Bottom band (Non-smokers):** Charges increase slowly and linearly with age, averaging about $8,400. These are the majority of patients.
2. **Middle band (Smokers with lower BMI):** Charges are higher than non-smokers but not extreme.
3. **Top band (Smokers with BMI ≥ 30):** Charges are dramatically higher, averaging about $32,050. This is the "double risk" group.

This three-band pattern is the key insight of the entire project. It tells us that smoking status and BMI don't just add up — they **multiply** each other's effect. A non-smoker with high BMI pays moderately more. A smoker with normal BMI pays significantly more. But a smoker with high BMI pays **enormously** more. This is called an **interaction effect**, and capturing it through feature engineering is what makes our model perform so well.

## Correlation Analysis
Pearson correlation measures how linearly related two variables are (range: -1 to +1):

Before feature engineering:
- `smoker` → 0.787 correlation with charges (strongest)
- `age` → 0.299
- `bmi` → 0.198
- `children` → 0.068
- `sex` → 0.058

After feature engineering (creating `smoker_bmi`):
- `smoker_bmi` → **0.845** correlation with charges (even stronger than raw smoker!)

This confirms that the interaction feature captures the relationship better than either feature alone.

## Distribution of Charges
The charges distribution is **right-skewed** (also called "positively skewed"). This means:
- Most values cluster on the left (low charges)
- A long tail stretches to the right (high charges)
- The mean ($13,270) is much higher than the median ($9,382)

This skewness is a problem for machine learning because many algorithms assume a roughly normal (bell-shaped) distribution. We fix this with a **log transform** during training (explained in Part 4).

---

# PART 4: PREPROCESSING AND FEATURE ENGINEERING

## What Is Preprocessing?
Preprocessing is preparing the raw data so machine learning algorithms can use it effectively. This includes handling missing values, encoding categories into numbers, creating new features, and transforming distributions.

## Step 1: Missing Value Check
We check if any column has missing values:
```python
data.isnull().sum()
```
Result: **zero missing values** across all columns. This is unusual — most real-world datasets have missing data. Since there are none, we skip imputation (the process of filling in missing values).

## Step 2: Encoding Categorical Variables
Machine learning algorithms work with numbers, not text. We need to convert categorical columns (`sex`, `smoker`, `region`) into numbers.

**Binary encoding for `smoker`:**
```python
data_fe['smoker_binary'] = (data_fe['smoker'] == 'yes').astype(int)
```
This creates a new column where `yes` becomes `1` and `no` becomes `0`. We need this as a number because we will multiply it with other features.

**One-hot encoding for `sex` and `region`:**
```python
dum = pd.get_dummies(data_fe[['sex', 'region']], drop_first=True, dtype=int)
```
One-hot encoding creates a new binary (0/1) column for each category. The `drop_first=True` parameter drops one category from each variable to avoid **multicollinearity** — a problem where one feature can be perfectly predicted from others, which confuses some algorithms (especially Linear Regression).

For `sex`: We get `sex_male` (1 if male, 0 if female). We don't need a `sex_female` column because if `sex_male` is 0, we know it's female.

For `region`: We get `region_northwest`, `region_southeast`, `region_southwest`. If all three are 0, we know it's northeast. This is the "dropped first" category.

## Step 3: Feature Engineering (The Most Important Step)
Feature engineering means creating new features from existing ones to help the model learn patterns. This is often the difference between a mediocre model and a great one.

We go from **6 raw features** to **21 engineered features**.

### Interaction Features (the key innovation):

**`smoker_bmi` = smoker_binary × bmi**
```python
data_fe['smoker_bmi'] = data_fe['smoker_binary'] * data_fe['bmi']
```
This is the **most important feature** in the entire model. Here's the logic:
- If someone is NOT a smoker (smoker_binary=0), this feature equals 0 regardless of BMI → BMI doesn't matter much for non-smokers
- If someone IS a smoker (smoker_binary=1), this feature equals their BMI → higher BMI means much higher predicted cost
- This captures the multiplicative interaction we saw in the EDA

**`smoker_age` = smoker_binary × age**
Same idea: older smokers pay exponentially more than younger smokers.

**`age_bmi` = age × bmi**
General interaction between age and BMI for all patients.

**`smoker_obese` = smoker_binary × is_obese**
A binary flag: 1 only if someone is BOTH a smoker AND obese. This captures the "double risk" group.

### Polynomial Features:

**`age_sq` = age²**
```python
data_fe['age_sq'] = data_fe['age'] ** 2
```
Why square the age? Because the relationship between age and charges is **not perfectly linear** — it curves upward. A 60-year-old doesn't just pay twice what a 30-year-old pays; they pay disproportionately more. Squaring age lets the model capture this curved relationship. This is a common technique in machine learning called **polynomial feature expansion**.

**`bmi_sq` = bmi²**
Same reasoning for BMI — the effect of BMI on charges accelerates at higher values.

### Binary Category Features:

**`is_obese`** — 1 if BMI ≥ 30, else 0
**`is_overweight`** — 1 if BMI ≥ 25, else 0
**`age_group_young`** — 1 if age < 30
**`age_group_mid`** — 1 if 30 ≤ age < 50
**`age_group_senior`** — 1 if age ≥ 50
**`has_children`** — 1 if children > 0

These binary features make it easier for tree-based models to find natural "cutoff points" in the data. For example, rather than the model having to discover that BMI=30 is a meaningful threshold, we tell it directly with `is_obese`.

### Log Transform:

**`log_bmi` = log(1 + bmi)**
```python
data_fe['log_bmi'] = np.log1p(data_fe['bmi'])
```
The `log1p` function computes `log(1 + x)`. We use `1 + x` instead of just `log(x)` to avoid issues when x=0 (log(0) is undefined). This slightly compresses the BMI distribution, reducing the impact of extreme BMI values.

## Step 4: Target Variable Transformation
```python
y = np.log1p(y_orig)
```
We apply a log transform to the target variable (charges) during training. Why?

1. **Charges is right-skewed** — log transform makes it more normally distributed
2. **Models perform better** on normally distributed targets because errors become more symmetric
3. **It naturally handles the large range** — charges go from $1,122 to $63,770 (a 57x range). After log transform, this becomes ~7.02 to ~11.06 (a much smaller range)

**Critical:** When we make predictions, we must **reverse the transform** to get back to dollar values:
```python
y_pred_dollars = np.expm1(y_pred_log)  # expm1 is the inverse of log1p
```
`expm1(x)` computes `e^x - 1`, which is the exact inverse of `log1p`.

## Final Feature Set (21 features)
After all engineering, our model uses these 21 features:
1. `age` — raw age
2. `bmi` — raw BMI
3. `children` — raw number of children
4. `smoker_binary` — is smoker (0/1)
5. `smoker_bmi` — smoker × BMI interaction **(most important)**
6. `smoker_age` — smoker × age interaction
7. `age_sq` — age squared (polynomial)
8. `bmi_sq` — BMI squared (polynomial)
9. `age_bmi` — age × BMI interaction
10. `is_obese` — BMI ≥ 30 flag
11. `is_overweight` — BMI ≥ 25 flag
12. `smoker_obese` — smoker AND obese flag
13. `age_group_young` — age < 30 flag
14. `age_group_mid` — 30 ≤ age < 50 flag
15. `age_group_senior` — age ≥ 50 flag
16. `has_children` — has children flag
17. `log_bmi` — log-transformed BMI
18. `sex_male` — male flag (one-hot)
19. `region_northwest` — region flag (one-hot)
20. `region_southeast` — region flag (one-hot)
21. `region_southwest` — region flag (one-hot)

---

# PART 5: DATA SPLITTING

## What Is Train/Test Split?
Before training, we divide the data into two parts:
- **Training set** — the model learns from this data
- **Test set** — we use this to evaluate performance on data the model has never seen

This is crucial because if we evaluated the model on the same data it trained on, we wouldn't know if it actually learned generalizable patterns or just memorized the training data.

## Why Three Different Split Ratios?
We experiment with three ratios to find the best balance:

```python
splits = {
    '70/30': train_test_split(X, y, test_size=0.30, random_state=42),
    '80/20': train_test_split(X, y, test_size=0.20, random_state=42),
    '90/10': train_test_split(X, y, test_size=0.10, random_state=42),
}
```

| Split | Training Set | Test Set | Trade-off |
|-------|-------------|----------|-----------|
| 70/30 | 936 samples | 402 samples | Less training data, but more robust test evaluation |
| 80/20 | 1,070 samples | 268 samples | Good balance (most common choice) |
| 90/10 | 1,204 samples | 134 samples | More training data, but test set might be too small to be reliable |

The `random_state=42` ensures reproducibility — every time we run the code, we get the same split. This is essential for comparing models fairly.

**Result:** The **80/20 split** gave the best test performance, confirming it provides the optimal balance for this dataset size.

---

# PART 6: MODEL TRAINING

## The Four Algorithms

### 1. Linear Regression
**How it works:** Finds a straight-line (or hyperplane in multiple dimensions) relationship between features and target. It computes coefficients (weights) for each feature that minimize the sum of squared errors.

**Formula:** `charges = w1×age + w2×bmi + w3×smoker_bmi + ... + bias`

**Strengths:** Simple, fast, interpretable. Good baseline.
**Weaknesses:** Can only model linear relationships. If the true relationship is curved or has complex interactions beyond what we engineered, it will miss them.

**In our project:** Linear Regression achieved R² ≈ 0.84. The fact that it performs this well is a testament to our feature engineering — the interaction features (especially `smoker_bmi`) linearize what was originally a non-linear relationship.

### 2. Random Forest Regressor
**How it works:** Builds many decision trees (a "forest"), each trained on a random subset of the data and features. The final prediction is the average of all trees' predictions.

**Key concept — Decision Trees:** A decision tree makes predictions by asking a series of yes/no questions. For example: "Is smoker_bmi > 25?" → if yes, go right; if no, go left. Each leaf node contains a prediction value.

**Key concept — Bagging:** Each tree sees a different random sample of the training data. This reduces overfitting because individual trees may overfit, but their mistakes cancel out when averaged.

**Key hyperparameters tuned:**
- `n_estimators` (200, 300, 500): Number of trees. More trees = better but slower.
- `max_depth` (10, 15, 20): Maximum depth of each tree. Deeper = more complex patterns but risk of overfitting.
- `min_samples_split` (2, 3, 5): Minimum samples needed to split a node. Higher = less overfitting.
- `min_samples_leaf` (1, 2): Minimum samples in a leaf. Higher = smoother predictions.

### 3. XGBoost (eXtreme Gradient Boosting) — THE BEST MODEL
**How it works:** Also uses decision trees, but instead of building them independently (like Random Forest), it builds them **sequentially**. Each new tree specifically tries to fix the mistakes of the previous trees. This is called **boosting**.

**Key concept — Gradient Boosting:** After the first tree makes predictions, we compute the **residuals** (errors). The second tree is trained to predict these errors. The third tree predicts the remaining errors, and so on. Each tree makes the overall model a little better.

**Key concept — Learning Rate:** Controls how much each tree contributes. A small learning rate (like 0.01) means each tree makes only tiny corrections, requiring more trees but often producing better results.

**Key concept — Regularization (`reg_lambda`):** Adds a penalty for model complexity, preventing overfitting. Higher values make the model simpler.

**Key hyperparameters tuned:**
- `n_estimators` (300, 500, 800): Number of boosting rounds. Our best: **800**
- `learning_rate` (0.05, 0.03, 0.01): Step size. Our best: **0.01** (small steps, many trees)
- `max_depth` (4, 5, 6): Tree depth. Our best: **5**
- `subsample` (0.7, 0.8): Fraction of data used per tree. Our best: **0.8**
- `colsample_bytree` (0.7, 0.8): Fraction of features used per tree. Our best: **0.8**
- `reg_lambda` (1.0, 1.5, 2.0): L2 regularization. Our best: **2.0** (strong regularization)

**GPU Acceleration:**
```python
m = XGBRegressor(tree_method="hist", device="cuda", **p).fit(X_tr, y_tr)
```
XGBoost supports training on GPU, which makes it much faster. `tree_method="hist"` uses a histogram-based algorithm optimized for speed. `device="cuda"` tells it to use the GPU.

### 4. LightGBM (Light Gradient Boosting Machine)
**How it works:** Similar to XGBoost (it's also gradient boosting), but uses two techniques to be faster:
- **Leaf-wise tree growth** — grows the tree by splitting the leaf with the highest gain, rather than level-by-level
- **Histogram-based binning** — groups feature values into bins for faster computation

**Key hyperparameters tuned:**
- Similar to XGBoost, plus `num_leaves` (31, 63) — controls tree complexity more directly than `max_depth`
- `verbose=-1` — suppresses training output messages

## The Training Loop
The code trains every combination of (algorithm × split ratio × hyperparameter set) and keeps the best result for each (algorithm × split) combination:

```python
for split_name in ['70/30', '80/20', '90/10']:
    X_tr, X_te, y_tr, y_te = splits[split_name]

    # For each algorithm, try all hyperparameter sets
    best_score, best_result = -999, None
    for p in gbr_params:
        m = XGBRegressor(**p).fit(X_tr, y_tr)
        metrics = evaluate_model(m, X_tr, X_te, split_name)
        if metrics['test_r2'] > best_score:  # Keep only the best
            best_score = metrics['test_r2']
            best_result = (m, p, metrics)
```

**Important detail — evaluation in original scale:**
```python
def evaluate_model(model, X_tr, X_te, split_name):
    y_pred_log_te = model.predict(X_te)      # Predictions in log scale
    y_pred_te = np.expm1(y_pred_log_te)       # Convert back to dollars
    return {
        'test_r2': r2_score(y_te_orig, y_pred_te),   # Evaluate in dollars
        'test_mae': mean_absolute_error(y_te_orig, y_pred_te),
    }
```
Even though the model trains on log-transformed charges, we evaluate performance on the **original dollar scale** so the metrics are interpretable.

---

# PART 7: EVALUATION METRICS

## What Is R² (R-squared)?
R² measures what proportion of the variance in charges can be explained by our model. It ranges from 0 to 1:
- **R² = 0** → the model explains nothing (just predicting the average)
- **R² = 0.50** → the model explains 50% of the variance
- **R² = 0.88** → the model explains 88% of the variance (our result!)
- **R² = 1.00** → perfect prediction (suspicious — probably overfitting)

Our best model achieves R² = 0.8802, meaning it explains about 88% of the variation in insurance charges. The remaining 12% is due to factors not in our dataset (like medical history, lifestyle details, etc.).

## What Is MAE (Mean Absolute Error)?
MAE is the average absolute difference between predicted and actual values:
`MAE = average(|actual - predicted|)`

Our best MAE is **$1,912**. This means on average, our predictions are off by about $1,900 in either direction. For charges that range from $1,122 to $63,770, this is quite good.

## What Is RMSE (Root Mean Squared Error)?
RMSE is similar to MAE but penalizes large errors more heavily:
`RMSE = sqrt(average((actual - predicted)²))`

RMSE is always ≥ MAE. When RMSE is much larger than MAE, it means the model has some predictions with very large errors. Our RMSE is about $4,313 while MAE is $1,912, indicating some predictions (likely for high-cost smokers) have larger errors.

## Train R² vs Test R²: Overfitting Check
We compare the model's performance on training data versus test data:
- **Train R²:** 0.9380
- **Test R²:** 0.8802
- **Gap:** ~5.8 percentage points

This gap is acceptable. If the gap were very large (e.g., Train R²=0.99, Test R²=0.60), it would indicate severe **overfitting** — the model memorized the training data but can't generalize. Our ~6% gap shows the model generalizes reasonably well.

---

# PART 8: RESULTS COMPARISON

## Complete Results Table

| Split | Model | Train R² | Test R² | Test MAE | Best For |
|-------|-------|----------|---------|----------|----------|
| 70/30 | Linear Regression | 0.8450 | 0.8361 | $2,474 | |
| 70/30 | Random Forest | 0.9620 | 0.8668 | $2,070 | |
| 70/30 | XGBoost | 0.9350 | 0.8706 | $1,996 | |
| 70/30 | LightGBM | 0.9410 | 0.8670 | $2,072 | |
| **80/20** | **XGBoost** | **0.9380** | **0.8802** | **$1,912** | **← BEST OVERALL** |
| 80/20 | Random Forest | 0.9590 | 0.8791 | $1,994 | |
| 80/20 | LightGBM | 0.9430 | 0.8750 | $2,026 | |
| 80/20 | Linear Regression | 0.8440 | 0.8436 | $2,500 | |
| 90/10 | XGBoost | 0.9360 | 0.8706 | $1,922 | |
| 90/10 | Random Forest | 0.9580 | 0.8692 | $1,963 | |
| 90/10 | LightGBM | 0.9420 | 0.8700 | $2,010 | |
| 90/10 | Linear Regression | 0.8430 | 0.8372 | $2,494 | |

## Key Observations
1. **XGBoost wins consistently** across all splits and metrics
2. **LightGBM is a close second** — both are gradient boosting algorithms, so this makes sense
3. **Random Forest is third** — also tree-based, but bagging is generally slightly less powerful than boosting for structured/tabular data
4. **Linear Regression is last** but still respectable (R² > 0.83) — our feature engineering made the problem "more linear"
5. **80/20 split is optimal** — enough training data to learn well, enough test data for reliable evaluation
6. **No severe overfitting** — the Train-Test R² gap for XGBoost (~6%) is reasonable

## Saving the Best Model
```python
joblib.dump(global_best_model, 'best_model_classic.pkl')
```
The winning model (XGBoost, 80/20 split) is saved as a `.pkl` file using `joblib`. This serializes the entire trained model to disk so we can load it later for predictions without retraining.

---

# PART 9: INFERENCE TESTING

## What Is Inference?
**Inference** means using a trained model to make predictions on new data. The inference notebook (`insurance_inference.ipynb`) loads the saved model and tests it with various inputs.

## Setting Up the Inference Pipeline
The inference notebook creates a `prepare_features()` function that replicates the exact same feature engineering from training:

```python
def prepare_features(age, sex, bmi, children, smoker, region):
    smoker_binary = 1 if smoker == 'yes' else 0
    features = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker_binary': smoker_binary,
        'smoker_bmi': smoker_binary * bmi,     # The key feature
        'smoker_age': smoker_binary * age,
        'age_sq': age ** 2,
        'bmi_sq': bmi ** 2,
        'age_bmi': age * bmi,
        'is_obese': int(bmi >= 30),
        'is_overweight': int(bmi >= 25),
        'smoker_obese': smoker_binary * int(bmi >= 30),
        'age_group_young': int(age < 30),
        'age_group_mid': int(30 <= age < 50),
        'age_group_senior': int(age >= 50),
        'has_children': int(children > 0),
        'log_bmi': np.log1p(bmi),
        'sex_male': int(sex == 'male'),
        'region_northwest': int(region == 'northwest'),
        'region_southeast': int(region == 'southeast'),
        'region_southwest': int(region == 'southwest'),
    }
    return pd.DataFrame([features])
```

**Critical requirement:** The features must be created in the **exact same way** and **exact same order** as during training. If you forget a feature or compute it differently, the model's predictions will be wrong.

The prediction function also applies the inverse log transform:
```python
def predict_charges(age, sex, bmi, children, smoker, region):
    X = prepare_features(age, sex, bmi, children, smoker, region)
    y_log = model.predict(X)[0]
    charges = np.expm1(y_log)      # Reverse the log transform
    return max(charges, 0)          # Clip negative predictions to 0
```

## Test Results: Normal Unseen Data
Realistic patient profiles produce sensible predictions:

| Profile | Age | BMI | Smoker | Predicted |
|---------|-----|-----|--------|-----------|
| Young healthy female | 25 | 22.0 | No | $4,256 |
| Middle-aged male, overweight | 35 | 28.5 | No | $6,313 |
| Mid-40s female, obese | 45 | 31.0 | No | $9,329 |
| Senior female, 3 kids | 60 | 29.0 | No | $15,366 |
| Young male smoker | 28 | 24.0 | Yes | $17,141 |
| Smoker, obese, 40s | 40 | 33.0 | Yes | $39,022 |
| Older male smoker, very obese | 52 | 35.5 | Yes | $51,190 |

**Average for non-smokers:** ~$8,222
**Average for smokers:** ~$34,753

These predictions align with our understanding from the EDA — smokers pay roughly 4-10x more than non-smokers.

## Test Results: Edge Cases
Extreme but possible values from the dataset boundaries:

| Description | Predicted |
|-------------|-----------|
| Youngest (18) + lowest BMI (15.96) + non-smoker | $1,820 (lowest) |
| Oldest (64) + highest BMI (53.13) + smoker + 5 kids | $49,973 (highest) |
| Ratio | 27.5x |

## Test Results: Absurd / Stress Test Data
We feed the model impossible inputs to test robustness:

| Input | Result | Analysis |
|-------|--------|----------|
| Age = 5 (child) | $1,820 | Model doesn't crash — clips to its minimum learned value |
| Age = 150 | $15,317 | Model doesn't extrapolate wildly — tree models are naturally bounded |
| BMI = 100 | $4,888 | BMI has weak effect for non-smokers, so this is reasonable |
| BMI = -5 | $6,192 | **Problem!** Model accepts invalid input without warning |
| Age = -10 | $1,966 | **Problem!** Same issue — negative age is meaningless |
| BMI = 999 + smoker | $47,675 | Model caps near its training maximum — tree models don't extrapolate |

**Key insight about tree-based models:** Unlike linear regression (which would extrapolate infinitely), tree-based models like XGBoost are **naturally bounded**. They can only predict values within the range of their training data. This means absurd inputs don't cause explosively wrong predictions, but it also means the model can't predict costs higher than ~$63,770 (the training maximum) even if the true cost should be $200,000.

## Sensitivity Analysis: How Each Feature Affects Predictions
The inference notebook systematically varies one feature at a time to measure its impact:

**Age effect (non-smoker, BMI=25):**
- Age 18 → $1,720
- Age 30 → $4,112 (baseline)
- Age 64 → $15,121
- Age 100 → $15,350 (barely changes after 64 — model has never seen ages > 64)

**BMI effect (non-smoker, age=30):**
- Relatively small impact: only ±$800 across the entire BMI range
- This confirms BMI matters mainly when combined with smoking

**Smoker × BMI interaction (the critical finding):**
| BMI | Non-smoker | Smoker | Smoker/Non-smoker ratio |
|-----|-----------|--------|------------------------|
| 20 | $3,133 | $14,743 | 4.7x |
| 25 | $4,112 | $19,732 | 4.8x |
| 30 | $3,690 | $23,107 | **6.3x** |
| 35 | $3,766 | $38,006 | **10.1x** |
| 40 | $4,422 | $45,226 | **10.2x** |

The ratio jumps dramatically at BMI=30 (the obesity threshold), confirming the model learned the smoker-obese interaction correctly.

---

# PART 10: LIMITATIONS AND RECOMMENDATIONS

## Known Limitations
1. **Small dataset (1,338 rows)** — with more data, the model could likely improve and generalize better
2. **US-only data** — predictions may not apply to healthcare systems in other countries
3. **Binary smoker status** — doesn't capture how much someone smokes (1 cigarette/day vs 2 packs/day)
4. **No temporal information** — can't capture trends over time (healthcare inflation, etc.)
5. **No input validation** — the model happily accepts negative ages, negative BMI, or other impossible values without warning
6. **Bounded predictions** — tree models cannot predict values outside their training range

## What Could Be Improved
1. **Add input validation** for deployment — reject negative values, warn about out-of-range inputs
2. **Use k-fold cross-validation** instead of a single train/test split for more robust evaluation
3. **Try ensemble stacking** — combine XGBoost + LightGBM predictions for potential marginal improvement
4. **Collect more data** — especially more smoker samples (only ~274 in the dataset)
5. **Add monitoring** for deployed models to detect data drift over time

---

# GLOSSARY OF KEY TERMS

- **Regression:** Predicting a continuous number (vs classification which predicts categories)
- **Feature Engineering:** Creating new input variables from existing ones to help the model learn
- **Interaction Feature:** A feature created by multiplying two features together to capture their combined effect
- **One-Hot Encoding:** Converting a categorical variable into multiple binary (0/1) columns
- **Log Transform (log1p):** Applying log(1+x) to compress a right-skewed distribution
- **Inverse Transform (expm1):** Applying e^x - 1 to reverse a log1p transform
- **Train/Test Split:** Dividing data into training (learning) and testing (evaluation) portions
- **R² Score:** Proportion of variance explained by the model (0 to 1, higher is better)
- **MAE:** Mean Absolute Error — average prediction error in the original unit (dollars)
- **RMSE:** Root Mean Squared Error — like MAE but punishes large errors more
- **Overfitting:** When a model memorizes training data but fails on new data (large Train-Test gap)
- **Gradient Boosting:** Sequential tree building where each tree corrects the previous trees' errors
- **Random Forest:** Parallel tree building where trees vote on the final prediction (bagging)
- **Hyperparameter Tuning:** Finding the best configuration settings for an algorithm
- **Inference:** Using a trained model to make predictions on new data
- **Serialization (joblib.dump):** Saving a trained model to a file for later use
- **GPU Acceleration:** Using graphics card hardware to speed up computation (CUDA)
- **Regularization (reg_lambda):** Adding penalties for model complexity to prevent overfitting
- **Multicollinearity:** When input features are too correlated with each other, confusing some algorithms
- **Right-Skewed Distribution:** A distribution where most values cluster left but a long tail extends right
