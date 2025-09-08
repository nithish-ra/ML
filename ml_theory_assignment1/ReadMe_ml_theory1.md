##  Overview

This assignment explores the use of **linear models** for both regression and classification.  
The work is divided into two parts:

1. **Regression – Mobile Phone Price Prediction**  
   Using the mobile phone dataset, linear regression is implemented through both **closed-form** and **gradient descent** approaches.  
   - Closed-form solution computes parameters using the matrix pseudoinverse.  
   - Gradient descent iteratively updates parameters to minimize error.  
   - Ridge regression (L2 regularization) is applied to handle overfitting and improve generalization.  
   - Standardization of features is shown to significantly affect stability and performance.  
   - Feature importance is analyzed by examining ridge regression coefficients.  

   Key results:  
   - Both closed-form and gradient descent achieve similar prediction performance.  
   - Ridge regression improves robustness against overfitting.  
   - Standardization ensures better performance across different regularization strengths.  

2. **Classification – Bank Note Authentication**  
   Logistic regression is applied to the banknote authentication dataset.  
   - Models were trained with and without L2 regularization.  
   - Accuracies on train and test sets were compared across varying regularization strengths (λ).  
   - A 3D visualization of features illustrates class separability.  
   - Outliers were intentionally introduced to study their effect.  

   Key results:  
   - L2 regularization helps balance model generalization and avoids overfitting.  
   - Train/test accuracy varies with λ, showing the trade-off between underfitting and overfitting.  
   - Outliers negatively impact generalization, even if training accuracy remains high.  

---

##  Conclusion

- **Linear regression** is effective but sensitive to feature scaling and requires regularization for stability.  
- **Ridge regression** balances bias and variance while highlighting the most important features influencing price.  
- **Logistic regression with L2 regularization** provides robust classification performance.  
- **Outliers** reduce generalization ability, emphasizing the importance of proper preprocessing.  
