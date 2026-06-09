# Task 1: Data Analysis and Preprocessing Summary

## Completed Activities

### Fraud Data Analysis

* Loaded and validated Fraud_Data.csv
* Assessed missing values and duplicates
* Performed class imbalance analysis
* Analyzed purchase value, age, browser, source, and gender distributions
* Created time-based features:

  * Hour of day
  * Day of week
  * Time since signup
* Performed IP-to-country mapping using IpAddress_to_Country.csv
* Generated processed dataset

### Credit Card Dataset Analysis

* Loaded and inspected creditcard_data.csv
* Assessed missing values and duplicates
* Evaluated class imbalance
* Analyzed transaction amount distribution
* Performed correlation analysis with fraud labels

### Feature Engineering

* Created transaction frequency feature
* Applied numerical scaling using StandardScaler
* Applied one-hot encoding for categorical variables
* Generated final feature-engineered dataset

## Output Files

* data/processed/fraud_data_processed.csv
* data/processed/fraud_feature_engineered.csv

## Scripts

* scripts/fraud_data_preprocessing.py
* scripts/creditcard_eda.py
* scripts/feature_engineering.py
