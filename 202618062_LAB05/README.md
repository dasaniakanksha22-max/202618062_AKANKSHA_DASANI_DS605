# DS605: Fundamentals of Machine Learning
## Lab Assignment 5: Machine Learning with Scikit-learn and From Scratch

**Student Name:** Akanksha Dasani  
**Student ID:** 202618062  
**Course:** DS605 - Fundamentals of Machine Learning  
**Dataset:** UCI Productivity Prediction of Garment Employees  

---

## 1. Project Overview

The goal of this assignment is to implement both regression and classification pipelines on the UCI Garment Worker Productivity dataset using two complementary approaches:
1. **Scikit-learn Implementation (Part A):** Standard pipeline using built-in estimators, transformers, and evaluation utilities.
2. **From-Scratch Implementation (Part B):** Complete manual reconstruction using only NumPy and Pandas (no machine learning libraries).
3. **Benchmarking & Optimization (Part C):** Fair side-by-side comparison of execution times, predictive metrics, and implementing algorithmic optimizations (L2 regularization and convergence early stopping).

### Task Definitions:
* **Regression:** Predict the continuous target `actual_productivity` using Linear Regression.
* **Classification:** Predict the binary target `MeetsTarget` using Logistic Regression, where:
  $$\text{MeetsTarget} = \begin{cases} 1 & \text{if } \text{actual\_productivity} \ge \text{targeted\_productivity} \\ 0 & \text{otherwise} \end{cases}$$
* **Data Leakage Constraint:** As required, `actual_productivity` is strictly excluded from the input feature set for classification to prevent trivial target leakage.

---

## 2. Dataset Preprocessing & Cleaning

Before model training, several anomalies in the raw dataset were identified and resolved:
* **Inconsistent Categorical Values:** The `department` column contained trailing spaces (e.g., `'finishing '` vs `'finishing'`) and spelling errors (`'sweing'`). These were standardized to uniform categories (`'sewing'` and `'finishing'`).
* **Missing Value Imputation:** The `wip` (Work In Progress) feature contained numerous missing values. In garment manufacturing, finishing departments do not carry sewing WIP. Therefore, missing WIP values were logically imputed with `0`.
* **Redundant Features:** The `date` column was removed because day-of-week (`day`) and quarterly temporal variations (`quarter`) were already captured as distinct features.
* **Categorical Encoding:** One-hot encoding was applied to `['quarter', 'department', 'day', 'team']` with the first category dropped (`drop_first=True`) to prevent the dummy variable trap.
* **Feature Scaling:** Z-score standardization ($\frac{x - \mu}{\sigma}$) was fitted exclusively on the training partition and subsequently applied to the test partition to prevent data snooping.
* **Split Reproducibility:** A fixed 80/20 train-test split (`random_state=42`) was shared across both Scikit-learn and scratch implementations.

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Linear Regression (Closed-Form Solution)
For the regression task, an intercept column of ones was prepended to the feature matrix ($X_b = [\mathbf{1}, X]$). The weight vector was computed using the Ordinary Least Squares (OLS) Normal Equation:
$$w = (X_b^T X_b)^{-1} X_b^T y$$
In NumPy, `np.linalg.pinv` was utilized to guarantee numerical stability against potential multicollinearity.

### 3.2 Logistic Regression (Batch Gradient Descent)
For binary classification, the linear output $z = X_b w$ was mapped to a probability via the sigmoid function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Weights were iteratively updated using batch gradient descent to minimize binary cross-entropy loss:
$$\nabla_w J = \frac{1}{m} X_b^T (\sigma(X_b w) - y)$$
$$w := w - \alpha \nabla_w J$$
Predictions were thresholded at $p \ge 0.5$.

### 3.3 Optimizations (Part C)
* **L2 Regularization (Ridge Penalty):** Added a shrinkage penalty ($\lambda$) to both Linear and Logistic regression to prevent overfitting, without penalizing the bias term.
* **Convergence-Based Early Stopping:** Monitored the Euclidean norm of weight updates ($||\Delta w||$). If $||\Delta w|| < 10^{-5}$, iterations terminate early, eliminating unnecessary epochs.

---

## 4. Experimental Results

*(Note: Values recorded on a fixed 80/20 split with random_state=42)*

### Regression Performance (Target: `actual_productivity`)

| Implementation | MAE | RMSE | $R^2$ Score | Training Time (s) | Prediction Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Scikit-learn LinearRegression** | 0.1065 | 0.1442 | 0.2584 | ~0.0021 | ~0.0003 |
| **Manual (From Scratch - OLS)** | 0.1065 | 0.1442 | 0.2584 | ~0.0016 | ~0.0002 |
| **Manual Optimized (Ridge L2)** | 0.1058 | 0.1436 | 0.2641 | ~0.0018 | ~0.0002 |

### Classification Performance (Target: `MeetsTarget`)

| Implementation | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Prediction Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scikit-learn LogisticRegression** | 0.7583 | 0.8148 | 0.8684 | 0.8408 | ~0.0085 | ~0.0003 |
| **Manual (From Scratch - GD 1500)** | 0.7458 | 0.8065 | 0.8553 | 0.8302 | ~0.0380 | ~0.0003 |
| **Manual Optimized (L2 + Early Stop)** | 0.7625 | 0.8159 | 0.8750 | 0.8444 | ~0.0124 | ~0.0003 |

---

## 5. Observations & Discussion

1. **Mathematical Equivalence in Regression:**  
   The manual closed-form Linear Regression produced metrics identical to Scikit-learn's `LinearRegression` down to numerical precision limits ($< 10^{-12}$). Both methods solve the identical convex optimization problem analytically via matrix factorization.

2. **Understanding Runtime Discrepancies:**  
   * For **Linear Regression**, the manual NumPy implementation was equally fast (and occasionally slightly faster) because it bypassed Scikit-learn's internal object validations and directly invoked underlying BLAS/LAPACK C libraries via `np.linalg.pinv`.
   * For **Logistic Regression**, Scikit-learn's default solver (`L-BFGS`) relies on second-order Hessian approximations compiled in C/Cython, allowing it to converge in substantially fewer iterations. The baseline manual implementation relied on first-order Gradient Descent executed inside a Python loop.

3. **Impact of Optimization:**  
   Introducing early stopping based on parameter convergence cut manual gradient descent training time by more than 60%, stopping once the update norm dipped below the threshold. Adding L2 regularization improved out-of-sample generalization slightly, matching and slightly exceeding the baseline Scikit-learn score on the test set.

---

## 6. Repository Structure

```text
├── garments_worker_productivity.csv   # Dataset
├── garment_productivity.ipynb         # Complete, runnable Jupyter Notebook
└── README.md                          # Project documentation and benchmarks