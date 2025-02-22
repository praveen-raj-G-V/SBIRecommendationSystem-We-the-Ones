import pandas as pd
from preprocess import preprocess_data
from retrieve_data import fetch_data
from train_decision_tree import train_decision_tree
from train_collaborative_filtering import train_cf_model
from train_kmeans import train_kmeans

print("🔹 Loading Data...")
raw_data = fetch_data()


users_df, transactions_df, policies_df, interactions_df = preprocess_data(raw_data)


merged_df = transactions_df.merge(users_df, on="CustomerID", how="left")


merged_df = merged_df.merge(policies_df, on="PolicyID", how="left")


X = merged_df[['Age', 'IncomeLevel', 'Gender', 'Location', 'PremiumAmount', 'PolicyDuration']]
y = merged_df['PolicyID']  

train_decision_tree(X, y)  
train_cf_model(transactions_df) 
train_kmeans(users_df, transactions_df, policies_df, interactions_df)  


