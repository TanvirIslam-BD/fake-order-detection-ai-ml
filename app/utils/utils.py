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
def save_model(model):
    joblib.dump(model, MODEL_PATH)

def save_last_trained_features(categorical_features, numeric_features, date_features):
    last_trained_features = {
        "categorical": categorical_features,
        "numerical": numeric_features,
        "date": date_features
    }
    with open('last_trained_features.json', 'w') as f:
        json.dump(last_trained_features, f)

def load_last_trained_features():
    if os.path.exists('last_trained_features.json'):
        with open('last_trained_features.json', 'r') as f:
            return json.load(f)
    else:
        return None  # Return None if the file does not exist


def validate_and_select_features(new_file_path):
    # Load the last trained features
    last_trained_features = load_last_trained_features()

    if last_trained_features is None:
        print("No previously trained features found.")
        return None

    # Load the new CSV file
    new_data = pd.read_csv(new_file_path)

    # Get the required fields
    categorical_fields = last_trained_features['categorical']
    numeric_fields = last_trained_features['numerical']
    date_fields = last_trained_features['date']

    # Check for missing fields
    missing_fields = []
    for field in categorical_fields + numeric_fields + date_fields:
        if field not in new_data.columns:
            missing_fields.append(field)

    if missing_fields:
        print(f"Warning: The following required fields are missing in the new data: {', '.join(missing_fields)}")
    else:
        print("All required fields are present.")

    # Optionally, select only the fields that are present
    selected_features = new_data[categorical_fields + numeric_fields + date_fields].copy()
    return selected_features
