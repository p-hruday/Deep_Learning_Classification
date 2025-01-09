# German Credit Card Worthiness Classification Using Deep Learning

This project focuses on building and evaluating machine learning and deep learning models for classifying credit risks using the German Credit dataset. The primary objective is to predict whether a customer is a good or bad credit risk based on various demographic, financial, and behavioral attributes. The workflow includes data preprocessing, exploratory analysis, model development, hyperparameter tuning, and saving artifacts for deployment.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [Detailed Section Descriptions](#detailed-section-descriptions)
   - 4.1 Data Exploration
   - 4.2 Data Preprocessing
   - 4.3 Exploratory Data Analysis
   - 4.4 Model Development
   - 4.5 Hyperparameter Tuning
   - 4.6 Model Evaluation
   - 4.7 Neural Network Model Evaluation
   - 4.8 Saving Artifacts
5. [Results](#results)

---

## Introduction

This project aims to address a critical classification task: predicting the creditworthiness of customers using advanced machine learning and deep learning techniques. It employs the German Credit dataset, which contains 1,000 instances and 20 attributes (a mix of categorical and numerical). The workflow encompasses data exploration, preprocessing, and model evaluation to build robust and accurate models.

## Dataset

The German Credit dataset is designed to evaluate the creditworthiness of customers based on multiple attributes. It consists of two versions: one with categorical and symbolic attributes and another fully numerical for algorithm compatibility. Key aspects of the dataset:

- **Number of Instances**: 1,000
- **Number of Attributes**:
  - Original Dataset: 20 attributes (7 numerical, 13 categorical)
- **Attributes**:
  - Demographic data (e.g., age, employment status, and housing)
  - Financial indicators (e.g., checking account status, savings, and credit amount)
  - Behavioral features (e.g., payment history, number of existing credits)
- **Target Variable**:
  - Binary Classification:
    - Good Credit Risk (1)
    - Bad Credit Risk (0)
- **Cost Matrix**:
  Misclassification has an associated cost, emphasizing the importance of minimizing false positives for bad credit risks.

Attribute details and mappings are provided in the dataset documentation. The data preprocessing steps ensure compatibility for model training, handling missing values, encoding categorical features, and scaling numerical features.

## Project Workflow

The project workflow is structured into eight main sections:
1. Data Exploration
2. Data Preprocessing
3. Exploratory Data Analysis
4. Model Development
5. Hyperparameter Tuning
6. Model Evaluation
7. Neural Network Model Evaluation
8. Saving Artifacts

## Detailed Section Descriptions

### 4.1 Data Exploration
**Purpose**: To load and understand the structure of the dataset.
- Loaded the dataset using libraries like Pandas and NumPy.
- Inspected data types, missing values, and unique value distributions.

### 4.2 Data Preprocessing
**Purpose**: To prepare the data for analysis and modeling.
- Addressed missing values through appropriate imputation techniques.
- Encoded categorical variables using One-Hot Encoding and Label Encoding.
- Normalized numerical variables for improved model performance.
- Split the dataset into training, validation, and test sets.

### 4.3 Exploratory Data Analysis
**Purpose**: To uncover patterns and relationships within the data.
- Visualized feature distributions using histograms, bar plots, and scatterplots.
- Analyzed correlations and feature importance.
- Identified outliers using box plots and addressed them if necessary.

### 4.4 Model Development
**Purpose**: To develop predictive models.
- Implemented baseline models (e.g., Random Forest, SVM).
- Developed and trained a neural network for enhanced performance.

### 4.5 Hyperparameter Tuning
**Purpose**: To optimize model performance.
- Used techniques like GridSearchCV and RandomizedSearchCV.
- Identified optimal parameters for Random Forest, SVM, and the neural network.

### 4.6 Model Evaluation
**Purpose**: To assess the models' performance on unseen data.
- Evaluated models using metrics like accuracy, precision, recall, and F1-score.
- Analyzed confusion matrices to identify misclassification patterns.

### 4.7 Neural Network Model Evaluation
**Purpose**: To evaluate the deep learning model's performance.
- Fine-tuned the neural network for improved predictions.
- Generated detailed classification reports.

### 4.8 Saving Artifacts
**Purpose**: To save the trained models and preprocessing pipelines for future use.
- Saved the best models and scalers using joblib or pickle for reproducibility.

## Results

Key results of the project:
- **Random Forest Model**:
  - Accuracy: 95%
  - F1-Score: 0.95 (Macro Avg)
- **SVM Model**:
  - Accuracy: 96%
  - F1-Score: 0.95 (Macro Avg)
- **Neural Network Model**:
  - Accuracy: 96%
  - F1-Score: 0.96 (Macro Avg)

These metrics demonstrate the robustness of the models, making them viable for deployment in real-world credit risk prediction scenarios.

---
