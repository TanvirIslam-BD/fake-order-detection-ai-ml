from flask import request, Blueprint
from app import app
from app.handlers.predict import predict_handler

# Create a blueprint for the API routes
api = Blueprint('api', __name__)

@api.route("/api/v1/predict", methods=["POST"])
def predict():
    return predict_handler(request)
