import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans

# ✅ Fix Memory Leak Issue (KMeans on Windows)
os.environ["OMP_NUM_THREADS"] = "2"

def train_kmeans(users_df, transactions_df, policies_df, interactions_df):
    """ Trains K-Means model and saves both model & feature names. """

    # ✅ Convert CustomerID to string for consistency
    users_df["CustomerID"] = users_df["CustomerID"].astype(str)
    transactions_df["CustomerID"] = transactions_df["CustomerID"].astype(str)
    transactions_df["PolicyID"] = transactions_df["PolicyID"].astype(str)
    policies_df["PolicyID"] = policies_df["PolicyID"].astype(str)
    interactions_df["CustomerID"] = interactions_df["CustomerID"].astype(str)

    # ✅ Merge transactions with policies to get `PremiumAmount` & `PolicyDuration`
    transactions_with_policies = transactions_df.merge(
        policies_df[['PolicyID', 'PremiumAmount', 'PolicyDuration']],
        on="PolicyID", how="left"
    )

    # ✅ Aggregate to get **average PremiumAmount & PolicyDuration** per Customer
    avg_premium_duration = transactions_with_policies.groupby("CustomerID").agg(
        {"PremiumAmount": "mean", "PolicyDuration": "mean"}).reset_index()

    # ✅ Merge the aggregated values with `users_df`
    users_df = users_df.merge(avg_premium_duration, on="CustomerID", how="left")

    # ✅ Merge interactions to get `WebsiteVisits` & `CallCenterInteractions`
    users_df = users_df.merge(
        interactions_df[['CustomerID', 'WebsiteVisits', 'CallCenterInteractions']], 
        on="CustomerID", how="left"
    )

    # ✅ Fill missing values with 0 (for users with no transactions or interactions)
    users_df.fillna(0, inplace=True)

    # ✅ Select relevant features for clustering
    X = users_df[['Age', 'IncomeLevel', 'PremiumAmount', 'PolicyDuration', 'WebsiteVisits', 'CallCenterInteractions']]

    # ✅ Train K-Means model
    kmeans = KMeans(n_clusters=3, random_state=42)
    X['Cluster'] = kmeans.fit_predict(X)

    # ✅ Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # ✅ Save trained model
    joblib.dump(kmeans, 'models/kmeans.pkl')

    # ✅ Save feature names for consistency during prediction
    joblib.dump(X.columns.tolist(), 'models/kmeans_feature_names.pkl')

    print("✅ K-Means Model & Features Saved!")

# ✅ Run Training
if __name__ == "__main__":
    from retrieve_data import fetch_data
    from preprocess import preprocess_data

    raw_data = fetch_data()
    users_df, transactions_df, policies_df, interactions_df = preprocess_data(raw_data)

    # ✅ Train and save the K-Means model
    train_kmeans(users_df, transactions_df, policies_df, interactions_df)
