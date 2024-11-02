from .app import app
from flask import Flask
from .routes.routes import main
from .api.api import api

__all__ = ["app"]

def create_app():
    app = Flask(__name__)
    app.register_blueprint(main, url_prefix="/")
    app.register_blueprint(api, url_prefix="/")

    return app
