import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_decision_tree(X, y):
    """ Trains a properly split Decision Tree model with categorical feature encoding. """

    # ✅ Convert categorical variables into one-hot encoding
    X = pd.get_dummies(X, columns=['Gender', 'IncomeLevel', 'Location'], drop_first=True)

    # ✅ Split into Training (80%) & Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Train Decision Tree model (Limit depth to avoid overfitting)
    dt_model = DecisionTreeClassifier(
        random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2
    )  
    dt_model.fit(X_train, y_train)

    # ✅ Evaluate model performance on unseen data
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Decision Tree Accuracy: {accuracy:.2f}")

    # ✅ Save trained model
    joblib.dump(dt_model, 'models/decision_tree.pkl')

    # ✅ Save feature names used in training
    joblib.dump(X.columns.tolist(), 'models/dt_feature_names.pkl')

    print("✅ Decision Tree Model & Features Saved!")
