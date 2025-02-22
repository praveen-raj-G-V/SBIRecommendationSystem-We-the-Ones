import os
from flask import Flask
from config import Config
from app.routes import main
from app.auth import auth

def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.abspath("templates"),  # Ensure Flask finds templates
        static_folder=os.path.abspath("static")  # Ensure Flask finds static files
    )

    app.config.from_object(Config)  # Load configurations

    # Register Blueprints (for modular routes)
    app.register_blueprint(main)
    app.register_blueprint(auth)

    return app
