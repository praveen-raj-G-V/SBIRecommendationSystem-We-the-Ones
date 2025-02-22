import pandas as pd
from preprocess import preprocess_data
from retrieve_data import fetch_data
from train_decision_tree import train_decision_tree
from train_collaborative_filtering import train_cf_model
from train_kmeans import train_kmeans

print("🔹 Loading Data...")
raw_data = fetch_data()

# ✅ Unpack data properly
users_df, transactions_df, policies_df, interactions_df = preprocess_data(raw_data)

# ✅ Merge users_df with transactions_df to get purchased policies
merged_df = transactions_df.merge(users_df, on="CustomerID", how="left")

# ✅ Merge with policies_df to include policy details
merged_df = merged_df.merge(policies_df, on="PolicyID", how="left")

# ✅ Now, select the correct features for training
X = merged_df[['Age', 'IncomeLevel', 'Gender', 'Location', 'PremiumAmount', 'PolicyDuration']]
y = merged_df['PolicyID']  # Target variable (Policies)

# Train all models
train_decision_tree(X, y)  # Saves feature names
train_cf_model(transactions_df)  # Uses transactions to train CF model
train_kmeans(users_df, transactions_df, policies_df, interactions_df)  # Uses customer & policy data for clustering

print("🎉 All AI Models Trained Successfully!")
