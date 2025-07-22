# Loan Amount Prediction using Linear Regression

This repository contains the code and results for an experiment on predicting loan sanction amounts using a Linear Regression model. The project is part of the Machine Learning Algorithms Laboratory curriculum.

**Author:** Nithish Ra (3122237001033)

**View the Notebook:** [Google Colab](https://colab.research.google.com/drive/1n0g0KsdB0dNnbC9o3tz2XlrkPUwnK8-J)

---

## 🎯 Objective

The primary goal of this experiment is to build and evaluate a Linear Regression model to predict the loan amount sanctioned to a user based on various personal and financial features. The project also focuses on interpreting the model's results to understand the key factors influencing loan sanction decisions.

---

## ⚙️ Methodology

The project follows a standard machine learning pipeline:

1.  **Data Loading & Preprocessing:**
    * Irrelevant columns were dropped.
    * Missing values were handled by filling numerical columns with the mean and categorical columns with the mode.
    * Categorical features were converted to numerical format using `LabelEncoder`.

2.  **Exploratory Data Analysis (EDA):**
    * Visualizations such as histograms, scatter plots, and heatmaps were used to understand data distributions and feature relationships.
    * Boxplots were used to identify outliers in key numerical features.

3.  **Feature Engineering:**
    * Outliers in features like `Income (USD)` and `Loan Amount Request (USD)` were treated by clipping values at the 1st and 99th percentiles to reduce their skewing effect on the model.

4.  **Model Training & Evaluation:**
    * The dataset was split into 80% for training and 20% for validation.
    * Features were standardized using `StandardScaler`.
    * A Linear Regression model was trained on the preprocessed data.
    * The model's performance was evaluated using K-Fold Cross-Validation (K=5) to ensure robustness.

---

## 📊 Results Summary

The model's performance was evaluated on the validation set, yielding the following key metrics:

* **Mean Absolute Error (MAE):** 21,479.00
* **Root Mean Squared Error (RMSE):** 30,346.77
* **R² Score:** 0.5744
* **Adjusted R² Score:** 0.5737

The cross-validation results confirmed this performance, with an average R² score of **0.57**, indicating that the model generalizes well to unseen data.

### Key Visualizations

| Actual vs. Predicted Loan Amount | Feature Coefficients |
| :------------------------------: | :------------------: |
|  ![Actual vs Predicted Plot](https://i.imgur.com/r6wz0dF.png)   | ![Feature Coefficients Plot](https://i.imgur.com/8Q9Y1xN.png) |
| *The model's predictions align closely with the actual values, showing a strong linear relationship.* | *`Loan Amount Request (USD)` is the most influential feature, followed by `Credit Score`.* |

---

## 🧠 Learning Outcomes

This experiment provided hands-on experience in:
* Implementing a complete regression pipeline, from data cleaning to model evaluation.
* Applying feature engineering techniques like outlier treatment.
* Understanding the importance of cross-validation for robust model assessment.
* Interpreting model coefficients to understand feature importance and their impact on predictions.

---

## 🛠️ Libraries & Tools

* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab
