# Experiment 5: Perceptron vs Multilayer Perceptron (A/B Experiment)

## Objective
To implement and compare the performance of two models on the English Handwritten Characters Dataset:
- **Model A:** Single-Layer Perceptron Learning Algorithm (PLA)  
- **Model B:** Multilayer Perceptron (MLP) with hidden layers and nonlinear activations  

This experiment highlights the limitations of simple linear models and the effectiveness of neural networks in learning complex decision boundaries.

---

## Dataset
- **English Handwritten Characters Dataset**
- ~3,410 images across **62 classes** (digits `0–9`, uppercase `A–Z`, lowercase `a–z`).
- Each image is resized to **28×28**, flattened, and normalized.

---

## Methodology

### Preprocessing
1. Images converted to grayscale, resized, and normalized.
2. Labels encoded into integers.

### PLA (Single-Layer Perceptron)
- Uses **step activation function**.
- Implements binary classification: **“Is the character ‘0’ or not?”**
- Weight update rule: `w = w + η(y − ŷ)x`.
- Evaluated using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curve.

### MLP (Multilayer Perceptron)
- Architecture:  
  `Input (784) → Dense(256, ReLU) → Dropout → Dense(128, ReLU) → Dropout → Dense(62, Softmax)`
- Optimizer: **Adam**
- Loss: **Categorical Cross-Entropy**
- Trained with validation split, tracks accuracy and loss over epochs.
- Evaluated on **all 62 classes** using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curves (micro/macro average).

---

## Results & Insights
- **PLA:** Works only for linearly separable data. Performance is limited since the dataset is complex.  
- **MLP:** Learns non-linear decision boundaries, achieves much higher accuracy across all classes.  
- Hyperparameter tuning (optimizer, learning rate, hidden layers) has a significant impact on performance.  
- Overfitting risk is mitigated with **dropout** and validation monitoring.

---

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrices (PLA binary vs MLP multi-class)  
- ROC Curves (PLA binary, MLP micro/macro)  
- Training vs Validation accuracy/loss plots  

---

## Observations
1. PLA underperforms compared to MLP because it cannot capture complex non-linear boundaries.  
2. Adam optimizer provides faster convergence than SGD in this experiment.  
3. More hidden layers do not always guarantee better performance; tuning depth and dropout is crucial.  
4. MLP shows signs of overfitting if dropout/regularization is not applied.  

---

## Conclusion
This experiment demonstrates the gap between traditional linear models (PLA) and modern neural networks (MLP). While PLA struggles with real-world, non-linear data, MLPs excel when properly tuned, making them essential for tasks like handwritten character recognition.
