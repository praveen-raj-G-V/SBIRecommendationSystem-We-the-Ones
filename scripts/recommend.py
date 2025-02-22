import joblib
import pandas as pd
from retrieve_data import fetch_data  
from preprocess import preprocess_data  


dt_model = joblib.load('models/decision_tree.pkl')
cf_model = joblib.load('models/collaborative_filtering.pkl')

def recommend_policies_decision_tree(customer_id, top_n=5):
   
   
    raw_data = fetch_data()
    users_df, _, policies_df, _ = preprocess_data(raw_data)

   
    users_df["CustomerID"] = users_df["CustomerID"].astype(str)

    
    user_data = users_df[users_df["CustomerID"] == str(customer_id)]

    if user_data.empty:
        print(f"⚠ No data found for Customer ID {customer_id}. Cannot make predictions.")
        return []

   

    
    feature_cols = ['Age', 'IncomeLevel', 'Gender', 'Location']
    user_features = user_data[feature_cols]

    
    user_features = pd.get_dummies(user_features)

   
    model_features = joblib.load('models/dt_feature_names.pkl')  
    
    missing_cols = set(model_features) - set(user_features.columns)  
    missing_df = pd.DataFrame(0, index=user_features.index, columns=list(missing_cols)) 

    
    user_features = pd.concat([user_features, missing_df], axis=1)

   
    user_features = user_features[model_features]  

    
    policy_probs = dt_model.predict_proba(user_features)[0]
    top_policy_indices = policy_probs.argsort()[-top_n:][::-1]  
    top_policies = [dt_model.classes_[idx] for idx in top_policy_indices]

   
    recommendations = []
    for policy_id in top_policies:
        policy_id = int(policy_id)  
        policy_details = policies_df[policies_df["PolicyID"] == policy_id]

        if policy_details.empty:
            continue  
        policy_data = {
            "PolicyID": policy_id,
            "PolicyName": policy_details.iloc[0]["PolicyName"],
            "PolicyType": policy_details.iloc[0]["PolicyType"],
            "PremiumAmount": policy_details.iloc[0]["PremiumAmount"],
            "PolicyDuration": policy_details.iloc[0]["PolicyDuration"],
            "Source": "Decision Tree" 
        }

        recommendations.append(policy_data)

    return recommendations

def recommend_policies_collaborative_filtering(customer_id, top_n=5):
   


    raw_data = fetch_data()
    _, transactions_df, policies_df, _ = preprocess_data(raw_data)


    transactions_df["CustomerID"] = transactions_df["CustomerID"].astype(str)

    
    user_transactions = transactions_df[transactions_df["CustomerID"] == customer_id]
    purchased_policies = set(user_transactions["PolicyID"].unique()) 

    
    policy_scores = []
    for policy_id in policies_df["PolicyID"]:
        if policy_id in purchased_policies:
            continue  
        pred = cf_model.predict(customer_id, policy_id)
        policy_scores.append((policy_id, pred.est))  
   
    policy_scores.sort(key=lambda x: x[1], reverse=True)
    top_policies = [policy[0] for policy in policy_scores[:top_n]]

   

    
    recommendations = []
    for policy_id in top_policies:
        policy_details = policies_df[policies_df["PolicyID"] == policy_id]

        if policy_details.empty:
            continue  
        policy_data = {
            "PolicyID": policy_id,
            "PolicyName": policy_details.iloc[0]["PolicyName"],
            "PolicyType": policy_details.iloc[0]["PolicyType"],
            "PremiumAmount": policy_details.iloc[0]["PremiumAmount"],
            "PolicyDuration": policy_details.iloc[0]["PolicyDuration"],
              
        }
        recommendations.append(policy_data)

    return recommendations



if __name__ == "__main__":
    test_customer_id = input("Enter Customer ID: ")  

    dt_recommendations = recommend_policies_decision_tree(test_customer_id)
    cf_recommendations = recommend_policies_collaborative_filtering(test_customer_id)

  
    all_recommendations = dt_recommendations + cf_recommendations

    if all_recommendations:
       
        for policy in all_recommendations:
            print(f"   - Policy ID: {policy['PolicyID']} - {policy['PolicyName']} ({policy['PolicyType']})")
            print(f"     - Premium Amount: ₹{policy['PremiumAmount']}/year")
            print(f"     - Policy Duration: {policy['PolicyDuration']} years")
            print(f"     - Suggested By: {policy.get('Source', 'Unknown')}\n")  
    else:
        print(f"⚠ No policy recommendations available for {test_customer_id}.")
