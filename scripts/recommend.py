import joblib
import pandas as pd
from retrieve_data import fetch_data  # ✅ Fetch data from MySQL
from preprocess import preprocess_data  # ✅ Preprocess data

# ✅ Load trained AI Models
dt_model = joblib.load('models/decision_tree.pkl')
cf_model = joblib.load('models/collaborative_filtering.pkl')

def recommend_policies_decision_tree(customer_id, top_n=5):
    """ Uses Decision Tree to recommend policies based on customer demographics. """

    # ✅ Fetch & preprocess data from MySQL
    raw_data = fetch_data()
    users_df, _, policies_df, _ = preprocess_data(raw_data)

    # ✅ Convert CustomerID to string for consistency
    users_df["CustomerID"] = users_df["CustomerID"].astype(str)

    # ✅ Fetch user data
    user_data = users_df[users_df["CustomerID"] == str(customer_id)]

    if user_data.empty:
        print(f"⚠ No data found for Customer ID {customer_id}. Cannot make predictions.")
        return []

    print(f"\n✅ Running Decision Tree for Customer ID {customer_id}...")

    # ✅ Select features for prediction
    feature_cols = ['Age', 'IncomeLevel', 'Gender', 'Location']
    user_features = user_data[feature_cols]

    # ✅ Convert categorical data to numerical using one-hot encoding
    user_features = pd.get_dummies(user_features)

    # ✅ Align with model training features
    model_features = joblib.load('models/dt_feature_names.pkl')  # Load saved feature names from training

    # ✅ Create missing columns in one step using pd.concat()
    missing_cols = set(model_features) - set(user_features.columns)  # Find missing columns
    missing_df = pd.DataFrame(0, index=user_features.index, columns=list(missing_cols))  # Create missing columns

    # ✅ Join all missing columns at once (avoids fragmentation)
    user_features = pd.concat([user_features, missing_df], axis=1)

    # ✅ Reorder columns to match training order
    user_features = user_features[model_features]  # Ensures correct column order for predictions

    # ✅ Predict top policies
    policy_probs = dt_model.predict_proba(user_features)[0]
    top_policy_indices = policy_probs.argsort()[-top_n:][::-1]  # Get top N policies
    top_policies = [dt_model.classes_[idx] for idx in top_policy_indices]

    print(f"\n✅ Predicted Top Policies for {customer_id}: {top_policies}")

    # ✅ Retrieve full policy details
    recommendations = []
    for policy_id in top_policies:
        policy_id = int(policy_id)  # Convert to integer
        policy_details = policies_df[policies_df["PolicyID"] == policy_id]

        if policy_details.empty:
            continue  # Skip if policy not found

        policy_data = {
            "PolicyID": policy_id,
            "PolicyName": policy_details.iloc[0]["PolicyName"],
            "PolicyType": policy_details.iloc[0]["PolicyType"],
            "PremiumAmount": policy_details.iloc[0]["PremiumAmount"],
            "PolicyDuration": policy_details.iloc[0]["PolicyDuration"],
            "Source": "Decision Tree"  # ✅ Ensure 'Source' field is always added
        }

        recommendations.append(policy_data)

    return recommendations

def recommend_policies_collaborative_filtering(customer_id, top_n=5):
    """ Uses Collaborative Filtering to recommend policies based on similar users' purchases. """

    # ✅ Fetch & preprocess data from MySQL
    raw_data = fetch_data()
    _, transactions_df, policies_df, _ = preprocess_data(raw_data)

    # ✅ Convert CustomerID to string for consistency
    transactions_df["CustomerID"] = transactions_df["CustomerID"].astype(str)

    # ✅ Get policies user has already interacted with
    user_transactions = transactions_df[transactions_df["CustomerID"] == customer_id]
    purchased_policies = set(user_transactions["PolicyID"].unique())  # Policies already purchased

    # ✅ Predict policy recommendations using CF model
    policy_scores = []
    for policy_id in policies_df["PolicyID"]:
        if policy_id in purchased_policies:
            continue  # ✅ Skip already purchased policies

        pred = cf_model.predict(customer_id, policy_id)
        policy_scores.append((policy_id, pred.est))  # Store (policy_id, estimated rating)

    # ✅ Sort policies by highest predicted rating
    policy_scores.sort(key=lambda x: x[1], reverse=True)
    top_policies = [policy[0] for policy in policy_scores[:top_n]]

    print(f"\n✅ Predicted Top Policies (CF) for {customer_id}: {top_policies}")

    # ✅ Retrieve full policy details
    recommendations = []
    for policy_id in top_policies:
        policy_details = policies_df[policies_df["PolicyID"] == policy_id]

        if policy_details.empty:
            continue  # Skip if policy not found

        policy_data = {
            "PolicyID": policy_id,
            "PolicyName": policy_details.iloc[0]["PolicyName"],
            "PolicyType": policy_details.iloc[0]["PolicyType"],
            "PremiumAmount": policy_details.iloc[0]["PremiumAmount"],
            "PolicyDuration": policy_details.iloc[0]["PolicyDuration"],
            "AI Score": round(policy_scores[top_policies.index(policy_id)][1], 2),
            "Source": "Collaborative Filtering"  # ✅ Ensure 'Source' field is always added
        }
        recommendations.append(policy_data)

    return recommendations


# ✅ Test the function
# ✅ Test the function
if __name__ == "__main__":
    test_customer_id = input("Enter Customer ID: ")  # ✅ Input Customer ID

    dt_recommendations = recommend_policies_decision_tree(test_customer_id)
    cf_recommendations = recommend_policies_collaborative_filtering(test_customer_id)

    # ✅ Merge both recommendation lists
    all_recommendations = dt_recommendations + cf_recommendations

    if all_recommendations:
        print(f"\n✅ Recommended Policies for {test_customer_id}:")
        for policy in all_recommendations:
            print(f"   - Policy ID: {policy['PolicyID']} - {policy['PolicyName']} ({policy['PolicyType']})")
            print(f"     - Premium Amount: ₹{policy['PremiumAmount']}/year")
            print(f"     - Policy Duration: {policy['PolicyDuration']} years")
            print(f"     - Suggested By: {policy.get('Source', 'Unknown')}\n")  # ✅ Ensures no 'Source' KeyError
    else:
        print(f"⚠ No policy recommendations available for {test_customer_id}.")