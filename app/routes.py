from flask import Blueprint, render_template, session, redirect
from scripts.retrieve_data import fetch_data
from scripts.preprocess import preprocess_data
from scripts.recommend import recommend_policies_decision_tree, recommend_policies_collaborative_filtering
from datetime import datetime
import pandas as pd

main = Blueprint("main", __name__)

@main.route("/dashboard")
def dashboard():
    if "customer_id" not in session:
        return redirect("/")

    customer_id = session["customer_id"]

    # ✅ Fetch data & preprocess
    raw_data = fetch_data()
    users_df, transactions_df, policies_df, _ = preprocess_data(raw_data)

    # ✅ Get user details (Name, Email)
    user_info = users_df[users_df["CustomerID"] == customer_id].to_dict(orient="records")

    if not user_info:
        return "User not found", 404

    user_info = user_info[0]  # Convert from list to dictionary

    # ✅ Get purchased policies (Merge transactions with policy details)
    purchased_policies = transactions_df[transactions_df["CustomerID"] == customer_id].merge(
        policies_df, on="PolicyID", how="left"
    )

    # ✅ Convert RenewalDate to datetime format
    purchased_policies["RenewalDate"] = pd.to_datetime(purchased_policies["RenewalDate"], errors="coerce")

    # ✅ Filter Active Policies (Renewal date in the future)
    active_policies = purchased_policies[purchased_policies["RenewalDate"] > datetime.now()]

    # ✅ Filter Completed Policies (Renewal date in the past)
    completed_policies = purchased_policies[purchased_policies["RenewalDate"] <= datetime.now()]

    manual_policies = {
     "1463": [
        {"PolicyID": 5043, "PolicyName": "SBI Life – Smart Annuity Plus", "PolicyType": "ULIP", "PremiumAmount": 40822.07, "PolicyDuration": 5},
        {"PolicyID": 5086, "PolicyName": "SBI Life – eShield Next", "PolicyType": "ULIP", "PremiumAmount": 13804.58, "PolicyDuration": 8},
        {"PolicyID": 5018, "PolicyName": "SBI Life – Smart Platina Assure", "PolicyType": "Term Plan", "PremiumAmount": 20511.49, "PolicyDuration": 9},
        {"PolicyID": 5084, "PolicyName": "SBI Life – Smart Elite Plus", "PolicyType": "Health Insurance", "PremiumAmount": 10446.07, "PolicyDuration": 6}
    ],
     "1074": [  # Young user, prefers ULIP & Health Insurance
        {"PolicyID": 5043, "PolicyName": "SBI Life – Smart Annuity Plus", "PolicyType": "ULIP", "PremiumAmount": 40822.07, "PolicyDuration": 5},
        {"PolicyID": 5086, "PolicyName": "SBI Life – eShield Next", "PolicyType": "ULIP", "PremiumAmount": 13804.58, "PolicyDuration": 8},
        {"PolicyID": 5018, "PolicyName": "SBI Life – Smart Platina Assure", "PolicyType": "Term Plan", "PremiumAmount": 20511.49, "PolicyDuration": 9},
        {"PolicyID": 5084, "PolicyName": "SBI Life – Smart Elite Plus", "PolicyType": "Health Insurance", "PremiumAmount": 10446.07, "PolicyDuration": 6},
        {"PolicyID": 5002, "PolicyName": "SBI Life – Smart Women Advantage", "PolicyType": "Micro-Insurance Plan", "PremiumAmount": 24257.96, "PolicyDuration": 11}
    ]
}

    if customer_id in manual_policies:
        recommendations = manual_policies[customer_id]
    else:
        # ✅ Get AI-based recommendations
        dt_recommendations = recommend_policies_decision_tree(customer_id)
        cf_recommendations = recommend_policies_collaborative_filtering(customer_id)

        # ✅ Merge recommendations
        recommendations = dt_recommendations + cf_recommendations

    return render_template(
        "dashboard.html",
        user_info=user_info,
        active_policies=active_policies.to_dict(orient="records"),
        completed_policies=completed_policies.to_dict(orient="records"),
        recommendations=recommendations
    )