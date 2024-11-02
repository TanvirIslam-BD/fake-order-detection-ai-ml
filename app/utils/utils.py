import os

import joblib
import pandas as pd
from flask import json

from app.constants import TRAIN_HISTORY_PATH, features_file, MODEL_PATH, METRICS_PATH


# Custom transformer to extract datetime features
def extract_datetime_features(data_frame):
    data_frame["Date & Time"] = pd.to_datetime(data_frame["Date & Time"])
    data_frame["year"] = data_frame["Date & Time"].dt.year
    data_frame["month"] = data_frame["Date & Time"].dt.month
    data_frame["day"] = data_frame["Date & Time"].dt.day
    data_frame["hour"] = data_frame["Date & Time"].dt.hour
    data_frame["minute"] = data_frame["Date & Time"].dt.minute
    return data_frame.drop(columns=["Date & Time"])

def extract_datetime_data_json(data):
    data_and_time = pd.to_datetime(data["Date & Time"])
    data["year"] = data_and_time.dt.year
    data["month"] = data_and_time.dt.month
    data["day"] = data_and_time.dt.day
    data["hour"] = data_and_time.dt.hour
    data["minute"] = data_and_time.dt.minute
    return data


def save_training_history(entry):
    # Load existing history or initialize new list
    if os.path.exists(TRAIN_HISTORY_PATH):
        with open(TRAIN_HISTORY_PATH, "r") as f:
            history = json.load(f)
    else:
        history = []

    # Add new entry to history
    history.append(entry)

    # Save updated history back to file
    with open(TRAIN_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

def save_features_to_json(features):
    with open(features_file, 'w') as f:
        json.dump(features, f)

def load_features_from_json():
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            return json.load(f)
    return []

# Load model if it exists
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Save the model and metrics
def save_model(model, metrics):
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)