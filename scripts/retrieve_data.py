import pandas as pd
from db_connection import connect_db


def fetch_data():
    conn = connect_db()
    if conn is None:
        return

    query_dict = {
        "users": "SELECT * FROM Users;",
        "transactions": "SELECT * FROM Transactions;",
        "policies": "SELECT * FROM Policies;",
        "user_interactions": "SELECT * FROM UserInteractions;"
    }

    dfs = {}
    for name, query in query_dict.items():
        dfs[name] = pd.read_sql(query, conn)
        print(f" Retrieved {name} data.")

    conn.close()
    return dfs

# Test data retrieval
if __name__ == "__main__":
    data = fetch_data()
    print(data["users"].head()) 
