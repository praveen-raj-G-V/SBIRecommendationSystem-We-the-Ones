import hashlib
from flask import Blueprint, render_template, request, redirect, url_for, session
import pymysql
from config import Config

auth = Blueprint("auth", __name__)

# Function to connect to MySQL
def get_db_connection():
    return pymysql.connect(
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor
    )

# Function to hash a password using SHA-256
def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# Route: Login Page
@auth.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        customer_id = request.form["customer_id"]
        entered_password = request.form["password"]  # User-entered password

        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch stored password hash from database
        cursor.execute("SELECT PasswordHash FROM users WHERE CustomerID=%s", (customer_id,))
        user = cursor.fetchone()
        conn.close()

        if user:
            stored_hash = user["PasswordHash"]  # Retrieve the stored hash

            # Hash the entered password and compare it with stored hash
            if hash_password(entered_password) == stored_hash:
                session["customer_id"] = customer_id
                return redirect(url_for("main.dashboard"))  # Redirect to Dashboard
            else:
                return render_template("login.html", error="Invalid credentials!")
        else:
            return render_template("login.html", error="User not found!")

    return render_template("login.html")
