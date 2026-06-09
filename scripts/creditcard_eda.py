import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    credit_df = pd.read_csv("data/raw/creditcard_data.csv")
    print("Credit card dataset loaded successfully.")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    raise
except Exception as e:
    print(f"Unexpected error while loading credit card data: {e}")
    raise
# Load dataset
credit_df = pd.read_csv("data/raw/creditcard_data.csv")

# Basic inspection
print("Dataset Shape:", credit_df.shape)
print("\nMissing Values:")
print(credit_df.isnull().sum())

print("\nDuplicate Records:")
print(credit_df.duplicated().sum())

# Class distribution
print("\nClass Distribution:")
print(credit_df["Class"].value_counts())

print("\nClass Percentages:")
print(credit_df["Class"].value_counts(normalize=True) * 100)

# Fraud vs Legitimate Transactions
plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=credit_df)
plt.title("Fraud vs Legitimate Transactions")
plt.savefig("fraud_vs_legitimate.png")
plt.close()

# Amount Distribution
plt.figure(figsize=(8,5))
sns.histplot(credit_df["Amount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.savefig("amount_distribution.png")
plt.close()

# Correlation Analysis
corr_with_target = (
    credit_df.corr(numeric_only=True)["Class"]
    .sort_values(ascending=False)
)

print("\nTop Features Correlated with Fraud:")
print(corr_with_target.head(15))

plt.figure(figsize=(8,6))
corr_with_target.drop("Class").head(10).plot(kind="barh")
plt.title("Top Features Correlated with Fraud")
plt.savefig("fraud_correlations.png")
plt.close()

print("Credit Card EDA completed successfully.")

import pandas as pd

try:
    credit_df = pd.read_csv("data/raw/creditcard_data.csv")
    print("Credit card dataset loaded successfully.")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    raise
except Exception as e:
    print(f"Unexpected error while loading credit card data: {e}")
    raise