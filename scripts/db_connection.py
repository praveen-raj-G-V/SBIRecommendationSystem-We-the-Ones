import mysql.connector

def connect_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Praj77258@",
            database="SBILifeAI"
        )
        print(" MySQL Connection Successful!")
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Test Connection
if __name__ == "__main__":
    conn = connect_db()
    if conn:
        conn.close()
