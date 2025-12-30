# Diabetes Onset Prediction using Pima Indian Data

## Overview
This project focuses on identifying the likelihood of diabetes onset by analyzing patient health metrics. By utilizing the Pima Indians Diabetes Dataset, I developed a system that can assist in predicting whether a patient is at risk based on historical medical data.

## Problem Statement
The goal of this project is to predict diabetes onset based on medical and health features such as glucose levels, BMI, and age using a cleaned medical dataset and classification models.

## Dataset
The project uses the **Pima Indians Diabetes Dataset**, which includes several diagnostic variables for female patients. Key features analyzed include:
* **Pregnancies**: Number of times pregnant.
* **Glucose**: Plasma glucose concentration.
* **Blood Pressure**: Diastolic blood pressure.
* **Skin Thickness**: Triceps skinfold thickness.
* **Insulin**: 2-Hour serum insulin.
* **BMI**: Body mass index.
* **Diabetes Pedigree Function**: A score based on family history.
* **Age**: Patient age in years.
* **Outcome**: Target variable (0 for No Diabetes, 1 for Diabetes).

## Approach
The implementation followed a structured data science pipeline:

### 1. Data Cleaning
* Identified invalid "0" values in critical columns like Glucose and BMI.
* Replaced these invalid entries with the median value of the respective column to ensure statistical accuracy.

### 2. Model Development
I predicted diabetes by training and comparing two primary algorithms:
* **Support Vector Machine (SVM)**: Used for its ability to find the optimal boundary between classes.
* **Decision Tree**: Implemented to provide clear, rule-based logic for predictions.

### 3. Feature Scaling
* Applied `StandardScaler` to normalize the data, ensuring all features contributed equally to the model's accuracy.

### 4. User Interface
* Developed a desktop application using **Tkinter** to allow users to input diagnostic measurements and receive an immediate prediction.

## Results
Through model tuning and data cleaning, the following results were achieved:
* **Decision Tree Accuracy**: 78.39%
* **SVM Accuracy**: ~76%

The Decision Tree model was integrated into the final GUI for its balanced performance and reliability.



## Technologies Used
* **Python**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Tkinter**

## Project Timeline
November 2023 – December 2023
