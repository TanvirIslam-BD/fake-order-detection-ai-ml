from .app import app
from flask import Flask
from .routes import routes

__all__ = ["app"]

def create_app():
    app = Flask(__name__)
    app.register_blueprint(routes, url_prefix="/")

    return app
