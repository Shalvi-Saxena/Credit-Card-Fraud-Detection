# Credit Card Fraud Detection Project

## Overview

This project aims to tackle the pressing issue of credit card fraud by employing machine learning algorithms to distinguish between fraudulent and legitimate transactions. Using a dataset that spans from January 1, 2019, to December 31, 2020, we analyze transactions from 1,000 customers and 800 merchants, featuring over 1.6 million records. The dataset, rich with both numerical and categorical data, has been meticulously prepared for this task, leveraging the Sparkov Data Generation tool and the 'faker' Python library to simulate realistic credit card transaction patterns.

## Dataset

The dataset is on ([Kaggle])(https://www.kaggle.com/datasets/kartik2112/fraud-detection/data?select=fraudTrain.csv)). It encompasses a mix of 22 features from both legitimate and fraudulent credit card transactions, carefully generated to reflect real-world transaction dynamics. For the purpose of fraud detection, we have narrowed down to 12 significant features after removing redundancies:

- **Trans_date_trans_time**: Date and time of transaction
- **Category**: Purchase category
- **Amt**: Amount of payment
- **Last**: Customer's last name
- **Gender**: Customer's gender (M/F)
- **Lat**: Customer's latitude
- **Long**: Customer's longitude
- **City_pop**: City population
- **Job**: Customer's occupation
- **Merch_lat**: Merchant's latitude
- **Merch_long**: Merchant's longitude
- **is_fraud**: Indicator of fraud (0 for non-fraudulent, 1 for fraudulent)

## Research Questions
Primary Research Question: Can fraudulent credit card transactions be predicted?
Secondary Research Questions:
- **RQ 1**: Which features are most influential in predicting fraudulent transactions?
- **RQ 2**: How do different machine learning models compare in terms of efficiency and effectiveness in detecting fraudulent transactions?
- **RQ 3**: What is the impact of hyperparameter tuning on the performance of the selected classifiers?

## Methodology

To address the challenges posed by the dataset, including class imbalance, we employ undersampling and split the data into training and testing sets in an 80-20 ratio. Our analysis pipeline includes:

1. **Feature Selection**: Utilize non-linear correlation for identifying relevant features.
2. **Model Training**: Train Logistic Regression, Random Forest Classifier, and XGBoost models on the prepared dataset.
3. **Hyperparameter Tuning**: Conduct an ablation study to fine-tune the models for optimal performance.
4. **Evaluation**: Assess model performance using precision, recall, and F1 score as primary KPIs, with a focus on accurately detecting fraudulent transactions.
