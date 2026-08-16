# Industrial AI Data Platform

An end-to-end **Industrial AI and Predictive Maintenance platform** that connects **Snowflake directly to Python Machine Learning pipelines** to predict industrial equipment failures, compare ML models, explain model decisions using SHAP, and generate failure predictions from Snowflake data.

The project is designed around a realistic industrial data workflow rather than a notebook-only experiment. Industrial equipment data is stored and prepared in **Snowflake**, accessed directly from Python, processed using a reproducible ML pipeline, and used to train and evaluate predictive maintenance models.

---

## 🚀 Project Overview

Unexpected equipment failures can cause production downtime, maintenance costs, and operational disruptions.

Predictive maintenance uses historical equipment and sensor data to identify patterns associated with failures before they occur.

This project builds an **Industrial AI Data Platform** that demonstrates this workflow:

```text
                    Industrial Data
                          │
                          ▼
                     Snowflake
                          │
                          │ Direct Connector
                          ▼
                       Python
                          │
                          ▼
                  Feature Processing
                          │
                          ▼
               Machine Learning Models
                    │           │
                    ▼           ▼
          Logistic Regression   XGBoost
                    │           │
                    └─────┬─────┘
                          ▼
                  Model Evaluation
                          │
                          ▼
                    Best Model
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
            SHAP Analysis      Predictions
                 │                 │
                 ▼                 ▼
          Explainability      Failure Output
```

The current implementation uses **Snowflake as the data source** and Python scripts for the complete Machine Learning workflow.

---

# 🎯 Objectives

The main objectives of the project are to:

* Build a cloud-based industrial data platform using Snowflake.
* Create ML-ready industrial failure datasets.
* Connect Snowflake directly to Python.
* Develop a reproducible Machine Learning pipeline.
* Train multiple failure-classification models.
* Compare model performance using multiple evaluation metrics.
* Identify the strongest-performing model.
* Explain model predictions using SHAP.
* Generate equipment failure predictions from Snowflake data.
* Store models and experiment results for reproducibility.
* Establish a foundation for predictive maintenance applications.

---

# 🏭 Industrial Use Case

The platform focuses on **industrial equipment failure prediction**.

Instead of waiting for equipment to fail, Machine Learning can identify patterns in industrial sensor and operational data that indicate an increased likelihood of failure.

### Traditional Maintenance

```text
Equipment
    ↓
Failure
    ↓
Unexpected Downtime
    ↓
Maintenance
    ↓
Production Loss
```

### Predictive Maintenance

```text
Industrial Data
      ↓
   Snowflake
      ↓
 Machine Learning
      ↓
Failure Prediction
      ↓
Early Intervention
      ↓
Reduced Downtime
```

The objective is to support a transition from **reactive maintenance** to **data-driven predictive maintenance**.

---

# ☁️ Data Platform

Snowflake is used as the central cloud data warehouse for the Machine Learning datasets.

The project uses two dedicated Snowflake tables:

```text
APS_FAILURE_ML_TRAIN
APS_FAILURE_ML_TEST
```

### Dataset Distribution

| Dataset          |      Rows |
| ---------------- | --------: |
| Training Dataset |     1,600 |
| Testing Dataset  |       400 |
| **Total**        | **2,000** |

The datasets are accessed directly from Snowflake using the **Snowflake Connector for Python**.

There is no requirement to manually download the Snowflake ML datasets and place them into the project.

---

# 📊 Dataset Characteristics

The current ML pipeline uses:

* **2,000 total observations**
* **1,600 training observations**
* **400 testing observations**
* **20 selected features**

The datasets are balanced for the failure target:

| Dataset  | Non-Failure | Failure |
| -------- | ----------: | ------: |
| Training |         800 |     800 |
| Testing  |         200 |     200 |

This balanced distribution provides a suitable basis for evaluating binary classification performance.

---

# 🔄 End-to-End Data Pipeline

The implemented pipeline follows this architecture:

                    INDUSTRIAL DATA
                         │
                         ▼
                ┌─────────────────┐
                │  Data Ingestion  │
                │  & Data Loading  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     Snowflake    │
                │  Cloud Data      │
                │    Platform      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Data Preparation │
                │ & Data Quality   │
                │     Checks       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Feature Selection│
                │   20 Features    │
                └────────┬────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Machine Learning   │
              │       Pipeline       │
              └──────────┬───────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
     ┌─────────────────┐    ┌─────────────────┐
     │    Logistic     │    │     XGBoost     │
     │    Regression   │    │      Model      │
     │    Baseline     │    │                 │
     └────────┬────────┘    └────────┬────────┘
              │                     │
              └──────────┬──────────┘
                         ▼
                ┌─────────────────┐
                │ Model Evaluation │
                │ Accuracy         │
                │ Precision        │
                │ Recall           │
                │ F1-Score         │
                │ ROC-AUC          │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Model Selection │
                │    XGBoost       │
                └────────┬────────┘
                         │
             ┌───────────┴────────────┐
             ▼                        ▼
    ┌─────────────────┐      ┌─────────────────┐
    │ SHAP Explainable│      │ Model Inference │
    │       AI        │      │   & Prediction  │
    └────────┬────────┘      └────────┬────────┘
             │                        │
             │                        ▼
             │               ┌─────────────────┐
             │               │ Failure / No    │
             │               │ Failure         │
             │               │ Predictions     │
             │               └────────┬────────┘
             │                        │
             └────────────┬───────────┘
                          ▼
                ┌─────────────────────┐
                │   Results Storage   │
                │ Models / Predictions│
                │ SHAP / Metrics      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Dashboard       │
                │                     │
                │ • Failure KPIs      │
                │ • Predictions       │
                │ • Model Metrics     │
                │ • Feature Importance│
                │ • SHAP Insights     │
                │ • Failure Trends    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Monitoring       │
                │                     │
                │ • Data Quality      │
                │ • Model Performance │
                │ • Prediction Trends │
                │ • Drift Monitoring  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Maintenance /       │
                │ Business Decision   │
                └─────────────────────┘

# 🤖 Machine Learning Models

Two classification models have currently been implemented.

## 1. Logistic Regression

Logistic Regression was implemented as the **baseline model**.

It provides an interpretable and computationally efficient reference point for comparing more advanced models.

### Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **92.25%** |
| Precision | **96.17%** |
| Recall    | **88.00%** |
| F1-Score  | **91.91%** |
| ROC-AUC   | **97.12%** |

---

## 2. XGBoost

**XGBoost** was implemented as the more advanced tree-based Machine Learning model.

The current implementation uses **XGBoost 2.1.4**.

### Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **98.25%** |
| Precision | **98.49%** |
| Recall    | **98.00%** |
| F1-Score  | **98.25%** |
| ROC-AUC   | **99.81%** |

XGBoost substantially outperformed the Logistic Regression baseline across the evaluated metrics.

---

# 🏆 Model Comparison

The current model comparison is:

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     92.25% |     96.17% |     88.00% |     91.91% |     97.12% |
| **XGBoost**         | **98.25%** | **98.49%** | **98.00%** | **98.25%** | **99.81%** |

### Best Performing Model

**XGBoost** is currently the best-performing model.

Compared with Logistic Regression, XGBoost improved:

* Accuracy by **6.00 percentage points**
* Precision by **2.32 percentage points**
* Recall by **10.00 percentage points**
* F1-score by **6.34 percentage points**
* ROC-AUC by **2.69 percentage points**

The improvement in recall is particularly important for failure prediction because correctly identifying actual failures is a critical requirement in predictive maintenance.

---

# 🔍 Model Explainability with SHAP

Model performance alone does not explain **why** a model predicts an equipment failure.

To improve interpretability, the project implements **SHAP (SHapley Additive exPlanations)**.

SHAP was successfully executed on the **400 Snowflake test observations**.

### SHAP Outputs

The project generated:

```text
results/shap/shap_feature_importance.csv
results/shap/shap_feature_importance.png
results/shap/shap_summary_plot.png
```

The analysis identifies the features that have the strongest influence on model predictions.

### Top Features Identified

The current SHAP analysis identified the following important features:

```text
AA_000
BJ_000
CI_000
AP_000
CK_000
CC_000
DN_000
BH_000
BY_000
AN_000
```

This provides an additional layer of interpretability for the Industrial AI pipeline.

---

# 🔮 Prediction / Inference Pipeline

The project also includes a dedicated prediction pipeline.

The trained XGBoost model can be used to generate predictions from the Snowflake-based test data.

The prediction results are stored as:

```text
results/snowflake_predictions.csv
```

This separates the workflow into two important stages:

```text
Training
   ↓
Model
   ↓
Saved Model
   ↓
Inference
   ↓
Predictions
```

This structure makes the project closer to a practical ML application rather than a single training experiment.

---

# 💾 Saved Models and Results

Trained models and experiment outputs are stored locally for reproducibility.

### XGBoost Model

```text
results/models/snowflake_xgboost.pkl
```

### XGBoost Results

```text
results/snowflake_xgboost_results.txt
```

### Predictions

```text
results/snowflake_predictions.csv
```

### SHAP Results

```text
results/shap/
```

This allows the trained model, predictions, and explainability outputs to be reused without retraining the model every time.

---

# 📁 Project Structure

The current project is organized around Python scripts and modular ML components rather than Jupyter notebooks.

```text
Industrial-AI-Data-Platform/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── processed/
│   │   ├── scania/
│   │   ├── train_categorical_reduced.csv
│   │   ├── train_date_reduced.csv
│   │   └── train_numeric_reduced.csv
│   │
│   └── raw/
│       └── scania/
│           ├── aps_failure_description.txt
│           ├── aps_failure_test_set.csv
│           ├── aps_failure_training_set_clean.csv
│           └── aps_failure_training_set.csv
│
├── data_quality/
│   ├── check_scania_quality.py
│   └── preprocess_scania.py
│
├── ingestion/
│   └── load_scania_data.py
│
├── ml/
│   ├── explain_model.py
│   ├── explain_snowflake_model.py
│   ├── predict_snowflake.py
│   ├── train_baseline.py
│   ├── train_snowflake_model.py
│   ├── train_snowflake_xgboost.py
│   └── train_xgboost.py
│
├── results/
│   ├── models/
│   │   ├── snowflake_logistic_regression.pkl
│   │   └── snowflake_xgboost.pkl
│   │
│   ├── shap/
│   │   ├── shap_feature_importance.csv
│   │   ├── shap_feature_importance.png
│   │   └── shap_summary_plot.png
│   │
│   ├── baseline_results.txt
│   ├── shap_bar.png
│   ├── shap_feature_importance.csv
│   ├── shap_individual.csv
│   ├── shap_summary.png
│   ├── snowflake_logistic_regression_results.txt
│   ├── snowflake_predictions.csv
│   ├── snowflake_xgboost_results.txt
│   ├── threshold_analysis.csv
│   └── xgboost_results.txt
│
├── snowflake/
│   └── scania_ai_failure_ml.sql
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   └── test_example.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

> **Note:** The project does not depend on Jupyter notebooks for the implemented ML pipeline. The main ML workflow is executed through Python scripts.

---

# 🧩 Main ML Scripts

The ML directory contains separate scripts for the major pipeline stages.

### `train_snowflake_model.py`

Trains the baseline Logistic Regression model using data loaded directly from Snowflake.

```text
Snowflake
   ↓
Training Data
   ↓
Logistic Regression
   ↓
Evaluation
```

### `train_snowflake_xgboost.py`

Trains the XGBoost model using the same Snowflake training and testing tables and the same selected features.

```text
Snowflake
   ↓
Training Data
   ↓
XGBoost
   ↓
Evaluation
   ↓
Saved Model
```

### `explain_snowflake_model.py`

Performs SHAP-based model explainability using the trained model and Snowflake test data.

```text
Snowflake Test Data
        ↓
   Trained Model
        ↓
       SHAP
        ↓
Feature Importance
```

### `predict_snowflake.py`

Loads the trained model and generates predictions from Snowflake data.

```text
Snowflake
    ↓
Test Data
    ↓
Saved XGBoost Model
    ↓
Predictions
    ↓
snowflake_predictions.csv
```

---

# 🔐 Security

Snowflake credentials must not be committed to GitHub.

Sensitive connection information should be stored using environment variables or another secure local configuration mechanism.

Typical Snowflake configuration includes:

```text
SNOWFLAKE_USER
SNOWFLAKE_PASSWORD
SNOWFLAKE_ACCOUNT
SNOWFLAKE_WAREHOUSE
SNOWFLAKE_DATABASE
SNOWFLAKE_SCHEMA
```

Real credentials should never be placed directly inside Python scripts or committed to the repository.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Industrial AI Data Platform"
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Technology Stack

## Programming

* Python
* SQL

## Cloud Data Platform

* Snowflake

## Machine Learning

* Scikit-learn
* XGBoost

## Explainable AI

* SHAP

## Data Processing

* Pandas
* NumPy

## Database Connectivity

* Snowflake Connector for Python

## Development Tools

* VS Code
* Git
* GitHub

The project uses a **Python-script-based workflow** rather than relying on Jupyter Notebook execution.

---

# ▶️ Running the Pipeline

## 1. Configure Snowflake

Set up the required Snowflake connection credentials securely.

## 2. Verify the Snowflake Tables

The following tables should be available:

```text
APS_FAILURE_ML_TRAIN
APS_FAILURE_ML_TEST
```

## 3. Train the Baseline Model

Run:

```bash
python ml/train_snowflake_model.py
```

This trains and evaluates the Logistic Regression baseline.

## 4. Train XGBoost

Run:

```bash
python ml/train_snowflake_xgboost.py
```

This trains the XGBoost model and saves the trained model and evaluation results.

## 5. Run SHAP Explainability

Run:

```bash
python ml/explain_snowflake_model.py
```

This generates the SHAP feature-importance outputs.

## 6. Generate Predictions

Run:

```bash
python ml/predict_snowflake.py
```

The prediction results are saved to:

```text
results/snowflake_predictions.csv
```

---

# 📈 Current Results

The completed ML experiments produced:

### Logistic Regression

```text
Accuracy:   92.25%
Precision:  96.17%
Recall:     88.00%
F1-Score:   91.91%
ROC-AUC:    97.12%
```

### XGBoost

```text
Accuracy:   98.25%
Precision:  98.49%
Recall:     98.00%
F1-Score:   98.25%
ROC-AUC:    99.81%
```

### Best Model

```text
XGBoost
Accuracy: 98.25%
ROC-AUC:  99.81%
```

---

# 💡 Why This Project Matters

This project demonstrates more than simply training a Machine Learning model.

It combines:

```text
Cloud Data Warehouse
        +
Data Engineering
        +
Machine Learning
        +
Model Comparison
        +
Explainable AI
        +
ML Inference
```

This makes the project relevant to real-world **Industrial AI, Data Science, Machine Learning, Predictive Maintenance, and Data Engineering** use cases.

---

# 🎓 Key Learning Outcomes

Through this project, the following practical concepts were implemented:

* Industrial predictive maintenance
* Snowflake data warehousing
* Direct Snowflake-to-Python data pipelines
* SQL-based data preparation
* Machine Learning classification
* Logistic Regression
* XGBoost
* Model comparison
* Classification metrics
* SHAP explainability
* Feature importance analysis
* ML model persistence
* Model inference
* Prediction generation
* Reproducible Python ML workflows
* Cloud data and ML integration

---

# 🚀 Future Extensions

The current implementation provides a strong foundation for further Industrial AI development.

Possible future extensions include:

* Additional ML model benchmarking
* Hyperparameter optimization
* Advanced feature engineering
* Model monitoring
* Data-quality monitoring
* Automated model retraining
* ML experiment tracking
* Predictive maintenance dashboards
* Automated failure alerts
* Remaining Useful Life (RUL) prediction
* Prescriptive maintenance recommendations

These are extensions to the current platform and are not part of the completed baseline implementation.

---

# 📌 Project Status

## Completed

* [x] Industrial AI predictive-maintenance use case
* [x] Snowflake data platform
* [x] `APS_FAILURE_ML_TRAIN` table
* [x] `APS_FAILURE_ML_TEST` table
* [x] 1,600 training records
* [x] 400 testing records
* [x] Balanced training and testing datasets
* [x] 20 selected ML features
* [x] Direct Snowflake-to-Python pipeline
* [x] Logistic Regression baseline
* [x] XGBoost model
* [x] Model performance comparison
* [x] Accuracy evaluation
* [x] Precision evaluation
* [x] Recall evaluation
* [x] F1-score evaluation
* [x] ROC-AUC evaluation
* [x] XGBoost model persistence
* [x] Snowflake-based inference pipeline
* [x] Prediction output generation
* [x] SHAP explainability
* [x] SHAP feature-importance analysis
* [x] SHAP visualization outputs

## Current Best Result

**XGBoost — 98.25% Accuracy and 99.81% ROC-AUC**

---

# 👩‍💻 Author

**Manoj Shankar BADATALA**

Master's in Data Intelligence
ISEP — Paris, France

---

# ⭐ Project Summary

**Industrial AI Data Platform** is an end-to-end predictive maintenance solution that integrates **Snowflake, Python, Machine Learning, XGBoost, and SHAP** to predict and explain industrial equipment failures.

The platform directly retrieves **2,000 industrial records** from Snowflake, consisting of **1,600 training records and 400 testing records**, and uses **20 selected features** for failure prediction.

Two Machine Learning models were evaluated. Logistic Regression achieved **92.25% accuracy**, while XGBoost achieved **98.25% accuracy and 99.81% ROC-AUC**, making XGBoost the current best-performing model.

The project additionally implements **SHAP explainability** and a separate **prediction/inference pipeline**, providing a complete foundation for an Industrial AI predictive-maintenance platform.
