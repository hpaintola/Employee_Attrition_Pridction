# Employee Attrition Prediction Pipeline

## Project Overview
This project demonstrates an end-to-end data pipeline to predict employee attrition using machine learning. The pipeline consists of four stages: Data Ingestion, Data Transformation, Feature Engineering, and Model Training. Designed with scalability and reproducibility in mind, this project provides actionable insights for improving workforce retention strategies


## Business Problem
Employee attrition presents a significant challenge for organizations, leading to:

- Increased recruitment costs.
- Productivity losses due to turnover.
- Disruption in operational efficiency.
Organizations need to predict which employees are at risk of leaving and identify the underlying reasons for attrition. This knowledge enables proactive interventions to retain talent, enhance job satisfaction, and optimize workforce management.

This project provides a solution by leveraging machine learning to:

1. Predict employees at risk of attrition.
2. Analyze the drivers of employee turnover.
3. Suggest data-driven strategies to reduce churn and improve retention


## Goal of the Project
The primary goal is to develop a robust, scalable machine learning pipeline to predict employee attrition and empower organizations to:

- Understand attrition drivers: Analyze factors like job satisfaction, work-life balance, and compensation.
- Reduce churn: Identify high-risk employees and suggest proactive measures.
- Support data-driven HR decisions: Enable strategic planning to improve workforce stability.


## Project Structure
├── data_ingestion.ipynb         # Handles data extraction and ingestion <br>
├── data_transformation.ipynb   # Cleans and transforms raw data <br>
├── feature_engineering.ipynb   # Prepares data for model training <br>
├── model_training.ipynb        # Trains and evaluates the predictive model <br>



## Modules Overview
### 1. Data Ingestion
File: `data_ingestion.ipynb`
This module automates the process of downloading, extracting, and preparing raw data.

Key Steps:
- Data Retrieval: Fetches the dataset using the Kaggle API.
- Data Extraction: Extracts relevant files from ZIP archives.
- Data Storage: Saves raw data to the Bronze Layer for further processing.<br>

Highlights:
- Scalable ingestion pipeline for large datasets.
- Robust error handling for file and connectivity validation.

###  2. Data Transformation
File: `data_transformation.ipynb`
This module processes and cleans raw data to prepare it for downstream analysis.

Key Steps:
- Load Data: Reads raw data from the Bronze Layer using PySpark.
- Schema Validation: Ensures data consistency with predefined schemas.
- Data Cleaning: Handles missing values, removes outliers, and standardizes data.<br>
  
Highlights:
- Distributed processing using PySpark.
- Ensures data readiness for feature engineering.
  
### 3. Feature Engineering
File: `feature_engineering.ipynb`
This module prepares features for training machine learning models.

Key Steps:
- Feature Encoding:
- Ordinal encoding for columns like Work-Life Balance and Job Satisfaction.
- Nominal encoding for categorical variables such as Job Role and Gender.
- Feature Selection: Filters numeric and categorical columns relevant for modeling.<br>
  
Highlights:
- Optimized feature set for better model performance.
- Encodes and prepares both categorical and numeric data.

 ### 4. Model Training
File: `model_training.ipynb`
This module trains machine learning models and evaluates their performance.

Key Steps:

- Model Training:
Implements Logistic Regression and Random Forest classifiers using PySpark.
- Predictions: Generates predictions and probabilities for employee attrition.
- Evaluation:
Calculates metrics such as Area Under ROC (AUC) and Accuracy.
- Feature Importance:
Extracts feature importance using the trained Random Forest model.<br>
Highlights:

- Model Comparison: Compares different algorithms for optimal results.
- Feature Insights: Ranks feature importance to explain drivers of attrition.<br>
Sample Outputs:

- Area Under ROC (AUC): Evaluates the classifier's ability to distinguish between classes.
Feature Importance: Highlights features like Job Satisfaction, Work-Life Balance, and Overtime as key predictors.<br>

### Technologies Used
- PySpark: For distributed data processing and machine learning.
- Kaggle API: For data retrieval.
- Parquet: For efficient data storage.
- Jupyter Notebook: For development and experimentation.


