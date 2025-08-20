# Ensemble Techniques for Breast Cancer Classification

This project focuses on applying and comparing multiple ensemble learning techniques for the classification of breast cancer tumors (Benign vs. Malignant). The dataset used is the Wisconsin Diagnostic Breast Cancer dataset, which contains 30 real-valued features extracted from digitized images of fine needle aspirates of breast masses.

# Objective

The primary aim is to demonstrate how ensemble methods improve predictive performance compared to single learners. The study includes:

Decision Tree as a baseline model.

Ensemble methods such as AdaBoost, Gradient Boosting, Random Forest, XGBoost, and Stacking.

Evaluation of models based on classification accuracy, F1 score, confusion matrix, and ROC-AUC.

Preprocessing and Data Handling

Labels are encoded into binary form: Malignant = 1, Benign = 0.

Features are standardized using z-score normalization to ensure uniform scale.

Data is split into training (80%) and test (20%) sets, maintaining class balance with stratified sampling.

Exploratory analysis includes class balance visualization and correlation heatmaps for feature interdependence.

# Models and Their Roles
## Decision Tree

Serves as the baseline model.

Hyperparameters (criterion, depth, minimum samples per split/leaf) tuned via GridSearchCV.

Offers interpretability but suffers from high variance when used alone.

## AdaBoost

Combines multiple weak learners (shallow decision trees) sequentially.

Later learners focus on misclassified instances from previous rounds.

Hyperparameters tuned include number of estimators and learning rate.

## Gradient Boosting

Builds trees sequentially, but unlike AdaBoost, it optimizes residual errors through gradient descent.

Hyperparameters tuned include number of estimators, learning rate, depth, and subsampling ratio.

## XGBoost

An optimized gradient boosting framework designed for speed and regularization.

Adds regularization (L1 and L2) to control overfitting.

Hyperparameters tuned include depth, learning rate, number of trees, gamma, subsample ratio, and column sampling.

## Random Forest

Ensemble of decision trees trained on bootstrapped samples with feature bagging.

Reduces variance and improves generalization.

Hyperparameters tuned include number of estimators, maximum depth, feature selection strategy, and splitting rules.

## Stacking Classifier

Combines predictions from multiple base learners (SVM, Naïve Bayes, Decision Tree, KNN).


# Evaluation Approach

# Models are compared using:

Accuracy and F1 score on the test set.

Confusion matrices to capture misclassification patterns.

ROC-AUC for overall discriminatory ability.

5-Fold Cross-Validation to assess robustness and generalizability.

# Key Insights

Decision Tree provides a simple benchmark but is prone to overfitting.

Boosting methods (AdaBoost, Gradient Boosting, XGBoost) generally improve accuracy and stability.

Random Forest balances bias and variance effectively and often achieves strong results.

Stacking leverages model diversity, showing the benefit of combining linear and non-linear learners.

Cross-validation ensures that observed performance is consistent and not due to random train-test splits.
