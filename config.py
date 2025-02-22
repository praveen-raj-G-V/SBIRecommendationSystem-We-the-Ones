import sys
import os
import secrets
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
sys.path.append(SCRIPTS_DIR)

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(16))  # Generates a secure secret key
    MYSQL_HOST = "localhost"
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "Praj77258@"
    MYSQL_DB = "SBILifeAI"
