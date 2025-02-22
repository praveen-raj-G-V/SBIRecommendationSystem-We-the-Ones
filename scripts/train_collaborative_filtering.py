from surprise import Dataset, Reader, SVD
import joblib
import pandas as pd

def create_ratings_from_transactions(df):
    """ Create a rating column based on user interactions with policies. """
    df['Rating'] = df.groupby(['CustomerID', 'PolicyID'])['PolicyID'].transform('count')
    df['Rating'] = df['Rating'].clip(1, 5)  # Ensure ratings are between 1 and 5
    return df

def train_cf_model(transactions_df):
    """ Trains the Collaborative Filtering model using past transactions. """

    reader = Reader(rating_scale=(1, 5))

    # ✅ Ensure IDs are treated as integers
    transactions_df["CustomerID"] = transactions_df["CustomerID"].astype(int)
    transactions_df["PolicyID"] = transactions_df["PolicyID"].astype(int)

    # ✅ Generate Ratings from Transactions
    transactions_df = create_ratings_from_transactions(transactions_df)

    # ✅ Load dataset into Surprise library
    ratings = Dataset.load_from_df(transactions_df[['CustomerID', 'PolicyID', 'Rating']], reader)
    trainset = ratings.build_full_trainset()
    
    # ✅ Train SVD model
    cf_model = SVD(random_state=42)  # Add random state for consistent results
    cf_model.fit(trainset)

    # ✅ Save Model
    joblib.dump(cf_model, 'models/collaborative_filtering.pkl')
    print("✅ Collaborative Filtering Model Trained & Saved!")
