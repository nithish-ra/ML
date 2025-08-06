# 📬 Spam Email Classification using Machine Learning

implements and compares various classification algorithms — **Naive Bayes (Gaussian, Multinomial, Bernoulli)**, **K-Nearest Neighbors (KNN)**, and **Support Vector Machines (SVM)** — on the popular [Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase).

> 🚀 Objective: Classify emails as **Spam** or **Ham** based on word frequency, character frequency, and capitalization statistics.

---

## 🗂️ Dataset

- **Source:** UCI Spambase Dataset via Kaggle
- **Features:** 57 (word frequencies, special characters, capital run length)
- **Target:** `spam` column (1 = Spam, 0 = Ham)

---

## 📌 Models Implemented

### 🔹 1. Naive Bayes
Implemented three variants:
- **GaussianNB** – for continuous features. Hyperparameter: `var_smoothing`
- **MultinomialNB** – for word/char frequency. Hyperparameters: `alpha`, `fit_prior`
- **BernoulliNB** – for binary feature presence. Hyperparameters: `alpha`, `binarize`, `fit_prior`

Each model was:
- Trained using `GridSearchCV` (5-fold CV)
- Evaluated on Validation and Test sets
- Plotted with Confusion Matrix and ROC Curve

---

### 🔹 2. K-Nearest Neighbors (KNN)
- Uses distance-based classification
- Tuned using:
  - `n_neighbors`: [3, 5, 7, 9]
  - `weights`: ['uniform', 'distance']
- Evaluated using:
  - Cross-validation
  - Confusion matrix & ROC
- Feature scaling applied using `StandardScaler`

---

### 🔹 3. Support Vector Machine (SVM)
Four kernel types evaluated:
- **Linear**
- **Polynomial**
- **RBF**
- **Sigmoid**

Each kernel was tuned using `GridSearchCV` over:
- `C`: [0.1, 1, 10]
- `gamma`: ['scale', 'auto']
- `degree` (for Polynomial): [2, 3, 4]

The best kernel was chosen based on **F1 score**, retrained on the full training set, and tested on the hold-out set.

---

## 🔧 Tools & Libraries
- `scikit-learn` (SVM, Naive Bayes, KNN, GridSearchCV)
- `matplotlib`, `seaborn` (visualizations)
- `pandas`, `numpy` (data manipulation)
- `roc_curve`, `auc`, `confusion_matrix`, `classification_report`

---

## 🧪 Evaluation Metrics
Each model was evaluated using:
- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ ROC Curve & AUC Score
- ✅ Confusion Matrix

---

## 📊 Sample Output (from SVM GridSearch)

Best Kernel: RBF
Best Params: {'C': 10, 'gamma': 'scale'}
Test Accuracy: 0.9565
Test F1 Score: 0.9621
