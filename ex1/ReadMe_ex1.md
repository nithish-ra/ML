Machine Learning Lab - Experiment 1
Course: ICS1512 - Machine Learning Algorithms
Student: Nithish Ra (Reg No: 3122237001033)
College: Sri Sivasubramaniya Nadar College of Engineering, Chennai
Batch: 2023–2028 (B.E. CSE)
Academic Year: 2025–2026 (Odd Semester)

Experiment 1: Working with Python Packages
Objective:
Explore and implement basic functionalities of essential Python libraries used in Machine Learning:

NumPy
Pandas
SciPy
Scikit-learn
Matplotlib
Concepts Covered:
NumPy – Numerical Computing
import numpy as np a = np.array([1, 2, 3]) b = np.zeros((2, 2)) c = np.ones((3, 1)) d = np.arange(0, 10, 2) e = np.reshape(a, (3, 1))

Pandas – Data Preprocessing
import pandas as pd df = pd.read_csv("students.csv") df["Age"] = df["Age"].fillna(df["Age"].mean()) df["Passed"] = df["Result"].map({"Yes": 1, "No": 0}) df["Score"] = (df["Score"] - df["Score"].min()) / (df["Score"].max() - df["Score"].min())

SciPy – Mathematical Computing
from scipy import integrate def f(x): return x**2 area, _ = integrate.quad(f, 0, 5) print("Area under curve:", area)

Scikit-learn – Machine Learning Workflow
from sklearn.datasets import load_iris from sklearn.linear_model import LogisticRegression from sklearn.model_selection import train_test_split from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True) X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) model = LogisticRegression(max_iter=200) model.fit(X_train, y_train) accuracy = accuracy_score(y_test, model.predict(X_test)) print("Accuracy:", accuracy)

Matplotlib – Data Visualization
import matplotlib.pyplot as plt x = [1, 2, 3, 4, 5] y = [10, 20, 25, 30, 40] plt.plot(x, y, marker='o') plt.title("Simple Line Plot") plt.xlabel("X-axis") plt.ylabel("Y-axis") plt.grid(True) plt.show()

Learning Outcomes:

Mastered core data manipulation using NumPy and Pandas

Understood scientific computations with SciPy

Built basic ML models using Scikit-learn

Visualized data effectively using Matplotlib

🔹 1. Iris Dataset ML Task Type: Supervised Learning – Classification

Why: The dataset has labeled species (Setosa, Versicolor, Virginica), and we are classifying based on petal and sepal measurements.

Feature Selection Techniques:

ANOVA (Analysis of Variance)

SelectKBest

Suitable Algorithms:

K-Nearest Neighbors (KNN)

Logistic Regression

Decision Tree Classifier

Support Vector Machine (SVM)

🔹 2. Loan Amount Prediction ML Task Type: Supervised Learning – Regression

Why: The target variable (loan amount) is a continuous numeric value predicted using applicant data like income and credit history.

Feature Selection Techniques:

Correlation analysis

Mutual Information

Suitable Algorithms:

Linear Regression

XGBoost Regressor

🔹 3. Predicting Diabetes ML Task Type: Supervised Learning – Binary Classification

Why: The task is to classify patients as Diabetic or Non-Diabetic based on health metrics like BMI, glucose level, etc.

Feature Selection Techniques:

Chi-Square Test

f-classif (ANOVA for classification)

Suitable Algorithms:

Logistic Regression

Random Forest Classifier

🔹 4. Classification of Email Spam ML Task Type: Supervised Learning – Binary Classification

Why: Emails are labeled as either “Spam” or “Not Spam” — a binary classification task using text features like frequency of words.

Feature Selection Techniques:

Chi-Square Test

TF-IDF Feature Selection

Suitable Algorithms:

Naive Bayes Classifier (great for text)

Support Vector Machine (SVM)

🔹 5. Handwritten Character Recognition / MNIST ML Task Type: Supervised Learning – Multiclass Classification

Why: Each image corresponds to one of 10 digits (0–9), and the goal is to classify them. This is a high-dimensional image recognition task.

Feature Selection Techniques:

PCA (Principal Component Analysis)

Variance Thresholding

Suitable Algorithms:

Convolutional Neural Networks (CNNs – preferred)

Random Forest

Support Vector Machine (SVM)
