import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_decision_tree(X, y):
   
    X = pd.get_dummies(X, columns=['Gender', 'IncomeLevel', 'Location'], drop_first=True)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    dt_model = DecisionTreeClassifier(
        random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2
    )  
    dt_model.fit(X_train, y_train)

   
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")

   
    joblib.dump(dt_model, 'models/decision_tree.pkl')

    
    joblib.dump(X.columns.tolist(), 'models/dt_feature_names.pkl')

    
