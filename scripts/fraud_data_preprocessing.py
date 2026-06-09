import pandas as pd
import numpy as np
try:
    fraud_df = pd.read_csv("data/raw/Fraud_Data.csv")
    ip_df = pd.read_csv("data/raw/IpAddress_to_Country.csv")
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    raise
except Exception as e:
    print(f"Unexpected error: {e}")
    raise
# Load datasets
fraud_df = pd.read_csv("data/raw/Fraud_Data.csv")
ip_df = pd.read_csv("data/raw/IpAddress_to_Country.csv")

# Data inspection
print("Fraud Data Shape:", fraud_df.shape)
print("\nMissing Values:")
print(fraud_df.isnull().sum())

print("\nDuplicates:", fraud_df.duplicated().sum())

# Datetime conversion
fraud_df["signup_time"] = pd.to_datetime(fraud_df["signup_time"])
fraud_df["purchase_time"] = pd.to_datetime(fraud_df["purchase_time"])

# Time-based features
fraud_df["hour_of_day"] = fraud_df["purchase_time"].dt.hour
fraud_df["day_of_week"] = fraud_df["purchase_time"].dt.day_name()

fraud_df["time_since_signup"] = (
    fraud_df["purchase_time"] -
    fraud_df["signup_time"]
).dt.total_seconds() / 3600

# IP conversion
fraud_df["ip_int"] = fraud_df["ip_address"].astype(float).astype(np.int64)

ip_df["lower_bound_ip_address"] = ip_df["lower_bound_ip_address"].astype(np.int64)
ip_df["upper_bound_ip_address"] = ip_df["upper_bound_ip_address"].astype(np.int64)

# Sort before merge
fraud_df = fraud_df.sort_values("ip_int")
ip_df = ip_df.sort_values("lower_bound_ip_address")

# Country mapping
fraud_geo = pd.merge_asof(
    fraud_df,
    ip_df,
    left_on="ip_int",
    right_on="lower_bound_ip_address",
    direction="backward"
)

fraud_geo = fraud_geo[
    fraud_geo["ip_int"] <= fraud_geo["upper_bound_ip_address"]
]

# Save processed dataset
fraud_geo.to_csv(
    "data/processed/fraud_data_processed.csv",
    index=False
)

print("Fraud preprocessing completed successfully.")