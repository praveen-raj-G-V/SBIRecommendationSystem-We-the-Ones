from surprise import Dataset, Reader, SVD
import joblib
import pandas as pd

def create_ratings_from_transactions(df):
    
    df['Rating'] = df.groupby(['CustomerID', 'PolicyID'])['PolicyID'].transform('count')
    df['Rating'] = df['Rating'].clip(1, 5) 
    return df

def train_cf_model(transactions_df):
   

    reader = Reader(rating_scale=(1, 5))

    
    transactions_df["CustomerID"] = transactions_df["CustomerID"].astype(int)
    transactions_df["PolicyID"] = transactions_df["PolicyID"].astype(int)


    transactions_df = create_ratings_from_transactions(transactions_df)

    
    ratings = Dataset.load_from_df(transactions_df[['CustomerID', 'PolicyID', 'Rating']], reader)
    trainset = ratings.build_full_trainset()
    
   
    cf_model = SVD(random_state=42)  
    cf_model.fit(trainset)

    
    joblib.dump(cf_model, 'models/collaborative_filtering.pkl')
   
