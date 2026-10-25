# DS605: Fundamentals of Machine Learning — Lab Assignment 8
## Decision Trees and Support Vector Machines from Scratch

**Dataset:** [UCI Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) (1,372 samples, 4 continuous features, 2 classes)  
**Submission:** Public GitHub Repository  

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Part A: Decision Tree from Scratch](#part-a-decision-tree-from-scratch)
4. [Part B: Linear Soft-Margin SVM from Scratch](#part-b-linear-soft-margin-svm-from-scratch)
5. [Part C: Kernel SVM & Nonlinear Boundaries](#part-c-kernel-svm--nonlinear-boundaries)
6. [Part D: Final Comparison & Performance Analysis](#part-d-final-comparison--performance-analysis)
7. [Comprehensive Discussion & Observations](#comprehensive-discussion--observations)
8. [How to Run the Code](#how-to-run-the-code)

---

## 📖 Project Overview
This laboratory assignment implements two fundamental machine learning algorithms from scratch using only **NumPy** and **Pandas**:
1. **Decision Tree Classifier** supporting both Gini Impurity and Entropy criteria, recursive greedy splitting, stopping rules, and tree visualization.
2. **Linear Soft-Margin Support Vector Machine (SVM)** using Hinge Loss and Subgradient Descent optimization.

Both from-scratch algorithms are compared directly against **Scikit-Learn** reference models. The workflow includes:
- 2D decision boundary visualization on two features (*Variance* and *Skewness*).
- Margin and margin-violator analysis for SVM.
- Non-linear boundary exploration using Polynomial and RBF kernels on a synthetic dataset (`make_moons`).
- Full 4-feature benchmark evaluating Accuracy, Precision, Recall, F1-Score, and Training/Inference latency.

---

## 📁 Repository Structure
```text
.
├── data_banknote_authentication.txt  # UCI Banknote dataset (1,372 rows)
├── DS605_Lab8_Decision_Tree_SVM.ipynb # Main runnable notebook containing all experiments
├── README.md                         # Comprehensive lab report and analysis