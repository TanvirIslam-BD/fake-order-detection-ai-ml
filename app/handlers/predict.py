import os
import joblib
import pandas as pd
from flask import request, jsonify
from sklearn.pipeline import Pipeline
from app.constants import MODEL_PATH

def extract_datetime_data_json(data):
    if "Date & Time" not in data:
        return data
    else:
        data_and_time = pd.to_datetime(data["Date & Time"])
        data["year"] = data_and_time.year
        data["month"] = data_and_time.month
        data["day"] = data_and_time.day
        data["hour"] = data_and_time.hour
        data["minute"] = data_and_time.minute
    return data

def predict_handler(request: request, path: str = os.getenv("MODEL_PATH", MODEL_PATH)):
    model: Pipeline = joblib.load(path)
    statuses = {0: "Order not genuine", 1: "Genuine order"}
    request_json = request.get_json()
    request_json = extract_datetime_data_json(request_json)
    data_frame = pd.DataFrame.from_records([request_json])
    target_vector_y_predict = model.predict(data_frame)
    return jsonify(dict(prediction=statuses[int(target_vector_y_predict[0])]))




