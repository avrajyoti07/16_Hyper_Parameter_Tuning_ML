# Machine Learning Hyperparameter Tuning & Model Selection

This repository contains practical implementations and exercises for hyperparameter tuning and model selection using **scikit-learn**. The project transitions from basic train-test splitting to manual cross-validation loops, before leveraging automated tools like `GridSearchCV` and `RandomizedSearchCV` to systematically discover the best machine learning models and configurations.

## 📌 Project Overview

When building machine learning models, default parameters rarely yield optimal performance. This notebook demonstrates how to efficiently fine-tune hyperparameters using the classic **Iris Dataset**. Additionally, it implements a structured framework to compare completely different algorithms (SVM, Random Forest, and Logistic Regression) side-by-side to choose the best overall model for a given dataset.

---

## 🛠️ Key Techniques Covered

* **Train/Test Split Validation**: The initial baseline method to evaluate model accuracy.
* **K-Fold Cross-Validation (`cross_val_score`)**: Eliminating dataset split bias by evaluating performance across 5 distinct folds.
* **Manual Parameter Grids**: Writing custom nested loops over parameter arrays to understand the underlying mechanics of tuning.
* **Grid Search CV (`GridSearchCV`)**: Automating exhaustive parameter searches inside a clean, parallelizable framework.
* **Randomized Search CV (`RandomizedSearchCV`)**: Optimizing computationally expensive models by randomly sampling a fixed number of parameter combinations.
* **Multi-Model Evaluation**: Passing dictionaries of multiple estimators and their respective parameter grids to dynamically rank entirely different model classes.

---

## 📊 Models & Parameter Grids Evaluated

The final script runs an automated comparison across the following setups:

| Model | Hyperparameters Tested |
| :--- | :--- |
| **Support Vector Machine (SVM)** | `C`: [1, 10, 20], `kernel`: ['rbf', 'linear'] |
| **Random Forest Classifier** | `n_estimators`: [10, 50, 100], `max_features`: ['sqrt', 'log2'] |
| **Logistic Regression** | `C`: [1, 5, 10], `solver`: ['liblinear', 'lbfgs'] |

---

## 📈 Summary of Results

From the multi-model search space, the configurations scored as follows on the Iris dataset:

* 🥇 **SVM** achieved the highest accuracy (**98.0%**) with parameters: `{'C': 1, 'kernel': 'rbf'}`.
* 🥈 **Logistic Regression** reached **97.3%** accuracy using: `{'C': 1, 'solver': 'lbfgs'}`.
* 🥉 **Random Forest** achieved **96.6%** accuracy using: `{'max_features': 'sqrt', 'n_estimators': 50}`.

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
