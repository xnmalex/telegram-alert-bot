from flask import Flask
import logging
import sys
from .routes import webhook, screener

def create_app():
    app = Flask(__name__)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Import and register blueprints, if any
    app.register_blueprint(webhook)
    app.register_blueprint(screener)
    
    return app
