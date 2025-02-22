from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def preprocess_data(data):
    label_enc = LabelEncoder()
    data["users"]["IncomeLevel"] = label_enc.fit_transform(data["users"]["IncomeLevel"])
    data["users"]["Location"] = label_enc.fit_transform(data["users"]["Location"])
    data["users"]["Gender"] = label_enc.fit_transform(data["users"]["Gender"])

    # Merge datasets
    df = data["transactions"].merge(data["users"], on="CustomerID").merge(data["policies"], on="PolicyID").merge(data["user_interactions"], on="CustomerID")

    # Normalize numerical features
    scaler = StandardScaler()
    df[['PremiumAmount', 'PolicyDuration', 'WebsiteVisits', 'CallCenterInteractions']] = scaler.fit_transform(
        df[['PremiumAmount', 'PolicyDuration', 'WebsiteVisits', 'CallCenterInteractions']])

    # Separate the merged DataFrame into individual DataFrames
    users_df = data["users"]
    transactions_df = data["transactions"]
    policies_df = data["policies"]
    interactions_df = data["user_interactions"]

    return users_df, transactions_df, policies_df, interactions_df

# Test Preprocessing
if __name__ == "__main__":
    from retrieve_data import fetch_data
    raw_data = fetch_data()
    users_df, transactions_df, policies_df, interactions_df = preprocess_data(raw_data)
    print(users_df.head())
    print(transactions_df.head())
    print(policies_df.head())
    print(interactions_df.head())
