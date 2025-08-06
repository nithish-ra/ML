# 💰 Loan Sanction Amount Prediction

This machine learning project predicts loan sanction amounts based on customer and property details. It uses:
- 🔹 **Linear Regression** for baseline comparison
- 🔹 **Support Vector Regression (SVR)** with hyperparameter tuning

> 📊 Dataset used: [Predict Loan Amount Dataset](https://www.kaggle.com/datasets/phileinsophos/predict-loan-amount-data) from Kaggle

---

## 🧾 Project Goals

- Understand relationships between features and sanctioned loan amounts.
- Handle missing values, outliers, and categorical encoding.
- Perform feature selection using Random Forest.
- Train and compare two regression models:
  - Linear Regression
  - SVR with RBF, Polynomial, and Linear kernels (tuned via GridSearchCV)

---

## 📦 Tech Stack

- `Python 3`
- `pandas`, `numpy`
- `seaborn`, `matplotlib`
- `scikit-learn` (regression models, scaling, CV, GridSearch)

---

## 📊 Workflow

### 1. **Data Preprocessing**
- Dropped irrelevant columns (`Customer ID`, `Name`, etc.)
- Handled missing values:
  - Numeric → Median
  - Categorical → Mode
- Encoded categoricals using `LabelEncoder`
- Clipped outliers (1st–99th percentile)
- Feature scaling via `StandardScaler`

### 2. **Exploratory Data Analysis**
- Boxplots for outliers
- Correlation heatmaps
- Distribution plots
- Scatter plots for target vs top features

### 3. **Feature Selection**
- Used `RandomForestRegressor` to rank features
- Top 10 features were selected for modeling

### 4. **Modeling**

#### 🔹 Linear Regression
- Evaluated using:
  - R² and Adjusted R²
  - MAE, MSE, RMSE
  - 5-Fold Cross-Validation
- Visualized: Actual vs Predicted scatter plots

#### 🔹 Support Vector Regression (SVR)
- Pipeline with scaling + SVR
- GridSearchCV over:
  - Kernels: `rbf`, `poly`, `linear`
  - `C`, `epsilon`, `gamma`, `degree`
- Evaluated on:
  - Validation set
  - Cross-validation (R²)
  - Test set performance
- Visualized predictions

---

## 🔁 Model Comparison

| Metric         | Linear Regression | SVR (Best Kernel) |
|----------------|------------------:|------------------:|
| Validation R²  | ~0.57             | ~0.16             |
| CV R² (mean)   | ~0.55             | ~0.09             |
| Test R²        | ~0.55             | ~0.15             |

> ✅ **Linear Regression** performed better than SVR in this experiment.

---

## 📈 Visualizations
- 📦 Boxplots (before/after outlier treatment)
- 🔥 Correlation heatmap
- 🔍 Predicted vs Actual scatter plots
- 🧠 Feature importance bar chart

---


## 📊 Results Summary - LR

The model's performance was evaluated on the validation set, yielding the following key metrics:

* **Mean Absolute Error (MAE):** 21,479.00
* **Root Mean Squared Error (RMSE):** 30,346.77
* **R² Score:** 0.5744
* **Adjusted R² Score:** 0.5737

The cross-validation results confirmed this performance, with an average R² score of **0.57**, indicating that the model generalizes well to unseen data.



