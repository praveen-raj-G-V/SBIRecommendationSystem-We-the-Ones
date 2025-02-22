import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans

os.environ["OMP_NUM_THREADS"] = "2"

def train_kmeans(users_df, transactions_df, policies_df, interactions_df):
   

   
    users_df["CustomerID"] = users_df["CustomerID"].astype(str)
    transactions_df["CustomerID"] = transactions_df["CustomerID"].astype(str)
    transactions_df["PolicyID"] = transactions_df["PolicyID"].astype(str)
    policies_df["PolicyID"] = policies_df["PolicyID"].astype(str)
    interactions_df["CustomerID"] = interactions_df["CustomerID"].astype(str)

   
    transactions_with_policies = transactions_df.merge(
        policies_df[['PolicyID', 'PremiumAmount', 'PolicyDuration']],
        on="PolicyID", how="left"
    )

   
    avg_premium_duration = transactions_with_policies.groupby("CustomerID").agg(
        {"PremiumAmount": "mean", "PolicyDuration": "mean"}).reset_index()

   
    users_df = users_df.merge(avg_premium_duration, on="CustomerID", how="left")

   
    users_df = users_df.merge(
        interactions_df[['CustomerID', 'WebsiteVisits', 'CallCenterInteractions']], 
        on="CustomerID", how="left"
    )

   
    users_df.fillna(0, inplace=True)

   
    X = users_df[['Age', 'IncomeLevel', 'PremiumAmount', 'PolicyDuration', 'WebsiteVisits', 'CallCenterInteractions']]

    kmeans = KMeans(n_clusters=3, random_state=42)
    X['Cluster'] = kmeans.fit_predict(X)

   
    os.makedirs("models", exist_ok=True)


    joblib.dump(kmeans, 'models/kmeans.pkl')

   
    joblib.dump(X.columns.tolist(), 'models/kmeans_feature_names.pkl')

   

if __name__ == "__main__":
    from retrieve_data import fetch_data
    from preprocess import preprocess_data

    raw_data = fetch_data()
    users_df, transactions_df, policies_df, interactions_df = preprocess_data(raw_data)

   
    train_kmeans(users_df, transactions_df, policies_df, interactions_df)
