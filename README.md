# Cardiovascular Disease Predictor (Data Analytics Project)

This project builds a machine learning pipeline to **predict cardiovascular disease (CVD)** using structured patient data. It covers the full end-to-end workflow: data cleaning, feature engineering, exploratory data analysis (EDA), model development, evaluation, and final prediction generation for an unseen test set.

The work is based on my FDA Assignment 3 and uses the **Cardiovascular Disease dataset** from Kaggle.

---

## Project Objectives

- Use structured patient data to **predict cardiovascular disease risk** (binary classification).
- Build a **robust, well-validated, and explainable** machine learning pipeline.
- Compare multiple classification algorithms and justify the final model choice.
- Generate **final predictions** for an unseen test dataset in a format compatible with Kaggle submission requirements.
- Reflect good **data science practice**: cleanliness, reproducibility, evaluation, and interpretability.

---

## Dataset

- **Source:**  
  Kaggle – *Cardiovascular Disease Dataset*  
  https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

- **Size:**  
  ~56,000 records (each row represents an individual patient).

- **Target variable:**
  - `cardio` – binary indicator (0/1) of cardiovascular disease.

- **Features:**

  **Continuous variables**
  - `age` – provided in days; converted to **years** for interpretability.
  - `height` – height in centimetres.
  - `weight` – weight in kilograms.
  - `ap_hi` – systolic blood pressure (mmHg).
  - `ap_lo` – diastolic blood pressure (mmHg).

  **Categorical / ordinal variables**
  - `gender` – 1 = female, 2 = male.
  - `cholesterol` – 1 = normal, 2 = above normal, 3 = well above normal.
  - `glucose` – 1 = normal, 2 = above normal, 3 = well above normal.

  **Binary variables (0/1)**
  - `smoke` – smoking status.
  - `alco` – alcohol consumption.
  - `active` – physical activity level.

The dataset required careful cleaning to correct inconsistent values, treat outliers (especially blood pressure), and ensure clinically plausible ranges.

---

## Project Scope & Pipeline

The project implements a complete CVD prediction workflow:

1. **Data Cleaning & Preprocessing**
   - Handled inconsistencies, unrealistic values, and outliers (e.g. extreme blood pressure).
   - Checked and addressed missing values (minimal in this dataset).
   - Converted `age` from days to years.
   - Ensured label encoding and data types were appropriate for modelling.

2. **Feature Engineering & Selection**
   - Basic feature transformations (e.g. handling skew, scaling).
   - Feature selection via:
     - **Recursive Feature Elimination (RFE)**
     - **Chi-square test**
   - Goal: reduce noise, improve interpretability, and focus on the most informative predictors.

3. **Exploratory Data Analysis (EDA)**
   - Distribution of the target variable (`cardio`).
   - Summary statistics and distributions for continuous features.
   - Relationships between risk factors (e.g. cholesterol, blood pressure, lifestyle) and CVD.
   - Identification of potential bias or imbalance in the dataset.

4. **Data Scaling**
   - Continuous features scaled so that distance-based and gradient-based models (e.g. KNN, SVM, MLP) behave properly and no single feature dominates.

5. **Model Training & Comparison**
   The following classification models were trained and evaluated:

   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbours (KNN)
   - Naive Bayes
   - Support Vector Machine (SVM)
   - Gradient Boosting
   - Multi-Layer Perceptron (MLP neural network)

   Each model was evaluated using:
   - Confusion matrix
   - ROC curve / ROC-AUC
   - Precision, recall, F1-score
   - Overall accuracy

6. **Hyperparameter Tuning**
   - **GridSearchCV** with 5-fold cross-validation was used for systematic tuning.
   - Focus on improving generalisation rather than just maximising training accuracy.

7. **Model Evaluation & Interpretation**
   - Comparison across all models based on validation metrics.
   - Interpretation using confusion matrices and ROC-AUC curves.
   - Consideration of clinical relevance: sensitivity (recall), specificity, and trade-offs.

8. **Final Prediction Generation**
   - The selected model was applied to the unseen **test dataset**.
   - Predictions were exported to a CSV file (`submission_final.csv`) in a format suitable for Kaggle submission.

---

## Best Model & Key Results

Although the **Multi-Layer Perceptron (MLP)** achieved the **highest ROC-AUC** on the validation set, the project ultimately selected **Gradient Boosting** as the **primary deployment model** because it provided:

- Strong and stable ROC-AUC (around the low 0.80s).
- Balanced sensitivity and specificity.
- Consistent performance across cross-validation folds.
- Better interpretability than a neural network in this context.

**Tuned Gradient Boosting hyperparameters (via GridSearchCV):**

- `n_estimators = 200`
- `max_depth = 4`
- `learning_rate = 0.1`

This configuration offered a good balance between complexity and generalisation, avoiding overfitting while capturing important non-linear relationships.

The final models demonstrated:
- Reliable discrimination between patients with and without CVD.
- Clinically meaningful behaviour when inspecting confusion matrices and ROC curves.

---

## Repository Structure

Recommended layout for this project on GitHub:

```text
.
├── data/
│   ├── train_subset.csv              # Training data used for model development
│   ├── test_kaggle_features.csv      # Features for unseen Kaggle/test data
│   ├── submission_sample.csv         # Sample submission file from Kaggle
│   └── submission_final.csv          # Final model predictions
├── notebooks/
│   ├── assignment.ipynb                              # Main assignment / analysis notebook
│   ├── FINAL_CVD_PREDICTION_TRAINING_A3_JUPYTER_NOTEBOOK.ipynb  # Full training pipeline
│   └── PREDICTION_FINAL_SUBMISSION.ipynb            # Notebook for generating final submission
├── models/
│   ├── best_model_tuned_Gradient_Boosting.pkl       # Saved Gradient Boosting model
│   └── model_features.pkl                           # Features used by the model
├── report/
│   └── FDA_A3_25674250.docx                         # Full written report for the assignment
├── README.md
└── .gitignore
