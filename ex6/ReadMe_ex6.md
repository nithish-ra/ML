# Experiment 6: Dimensionality Reduction and Model Evaluation (With and Without PCA)

## Objective
The objective of this experiment is to study the effect of **dimensionality reduction using Principal Component Analysis (PCA)** on the performance of various machine learning classifiers. Models are trained and validated under two conditions:
- Without PCA (original feature space)
- With PCA (reduced feature space)

The experiment applies **5-fold cross-validation** with hyperparameter tuning and compares results across models.

---

## Dataset
- **Name:** Wisconsin Diagnostic Breast Cancer (WDBC) dataset  
- **Source:** UCI Machine Learning Repository / scikit-learn built-in dataset  
- **Samples:** 569  
- **Features:** 30 numeric features (cell nuclei characteristics)  
- **Target Classes:**  
  - 0 = Malignant  
  - 1 = Benign  

---

## Preprocessing
1. Standardization of features using `StandardScaler`.
2. Application of PCA retaining **95% variance**.  
3. Scree plot plotted to justify the number of components retained.

---

## Models Evaluated
1. Support Vector Machine (SVM)  
2. Naive Bayes  
3. k-Nearest Neighbors (KNN)  
4. Logistic Regression  
5. Decision Tree  
6. Random Forest  
7. AdaBoost  
8. Gradient Boosting  
9. XGBoost  
10. Stacking Classifier (ensemble of Random Forest and SVM with Logistic Regression as meta-learner)

---

## Procedure
1. Preprocess and standardize the dataset.  
2. Train and validate models using **5-fold stratified cross-validation**.  
3. Perform **hyperparameter tuning** using `GridSearchCV` for each model.  
4. Record results for both No-PCA and PCA settings.  
5. Evaluate performance using:  
   - Accuracy  
   - F1-Score  
   - Confusion Matrix  
   - ROC and Precision-Recall Curves  

---

## Results
- **Fold-wise accuracies** are recorded for each model under both settings.  
- **Comparison table** shows mean accuracy and standard deviation (No-PCA vs PCA).  
- **Confusion matrices** and **classification reports** are generated for all models.  
- **ROC and PR curves** plotted for selected models (SVM, Random Forest, XGBoost).  

---

## Observations
- PCA improved performance for some models (e.g., KNN, Logistic Regression) by reducing noise and dimensionality.  
- Ensemble methods such as Random Forest and XGBoost showed stable performance regardless of PCA.  
- PCA sometimes reduced accuracy for tree-based methods, as they do not rely heavily on linear feature transformations.  
- PCA reduced variance across folds, leading to more stable results for linear models.  
- Stacking classifier remained robust under both conditions.

---

## Conclusion
PCA can improve model stability and accuracy in high-dimensional spaces, particularly for linear classifiers. However, for ensemble models like Random Forest and XGBoost, PCA provides little to no improvement. Stacking demonstrated robustness, confirming that ensembles can handle raw high-dimensional data effectively.

