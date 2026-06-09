import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load processed dataset
fraud_df = pd.read_csv("data/processed/fraud_data_processed.csv")

# Transaction frequency feature
transaction_frequency = (
    fraud_df.groupby("user_id")
    .size()
    .reset_index(name="transaction_count")
)

fraud_df = fraud_df.merge(
    transaction_frequency,
    on="user_id",
    how="left"
)

# Numerical scaling
numeric_features = [
    "purchase_value",
    "age",
    "time_since_signup",
    "transaction_count"
]

scaler = StandardScaler()

fraud_df[numeric_features] = scaler.fit_transform(
    fraud_df[numeric_features]
)

# Categorical encoding
categorical_features = [
    "source",
    "browser",
    "sex",
    "country",
    "day_of_week"
]

fraud_encoded = pd.get_dummies(
    fraud_df,
    columns=categorical_features,
    drop_first=True
)

# Save final dataset
fraud_encoded.to_csv(
    "data/processed/fraud_feature_engineered.csv",
    index=False
)

print("Feature engineering completed successfully.")