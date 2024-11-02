import base64
import io
import os
import pickle

from flask import  render_template, redirect
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import json
from datetime import datetime
from typing import List
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import re
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)

from .errors import errors
from .handlers.predict import predict_handler


app = Flask(__name__)

# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
app.register_blueprint(errors)
# Constants

np.random.seed(42)

# Constants

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
TIMESTAMP_FMT = "%m-%d-%Y, %H:%M:%S"

MODEL_PATH = "data/pipeline.pkl"  # Path to save/load model
METRICS_PATH = "data/metrics.json"  # Path to save metrics

# Path to save training history
TRAIN_HISTORY_PATH = "data/train_history.json"

# Global variable to store the last trained model's feature columns
last_trained_features = []
features_file = 'last_trained_features.json'

LABEL: str = "Genuine Order"

NUMERIC_FEATURES: List[str] = [
    "Amount (Total Price)",
    "Tickets (Quantity)",
    "Coupon amount",
]

PRICE_FEATURES: List[str] = [
    "Amount (Total Price)",
    "Coupon amount",
]

CATEGORICAL_FEATURES: List[str] = [
    "Customer Name",
    "Currency",
    "Country",
    "Booking type",
    "Payment"
]

# Additional date features
DATE_FEATURES: List[str] = [
    "year",
    "month",
    "day",
    "hour",
    "minute"
]

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

# Create the pipeline function
def create_pipeline(categorical_features: List[str], numeric_features: List[str],learning_rate, max_iter, max_leaf_nodes, min_samples_leaf):

    # Use StandardScaler for numeric features if scaling is desired (not strictly necessary for tree-based models)
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    # One-hot encode categorical features
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant")), ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
    )

    # Combine the transformations for price, datetime, numeric, and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf
    )

    # Use HistGradientBoostingClassifier
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

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

def train_model(label_column, file_path, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf):
    # Load data
    data = pd.read_csv(file_path)
    X = data.drop(label_column, axis=1)  # Replace 'target' with your actual target column
    y = data[label_column]  # Replace 'target' with your actual target column

    categorical_features = data.get("categorical_features", CATEGORICAL_FEATURES)
    numeric_features = data.get("numeric_features", NUMERIC_FEATURES + DATE_FEATURES)

    # Clean the price data directly
    data['Amount (Total Price)'] = data['Amount (Total Price)'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x.strip())) if isinstance(x, str) else x
    )

    # Clean the price data directly
    data['Coupon amount'] = data['Coupon amount'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x.strip())) if isinstance(x, str) else x
    )

    data_frame = extract_datetime_features(data)

    features = data_frame[categorical_features + numeric_features]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)


    model = load_model()  # Load existing model if present

    model = model or create_pipeline(categorical_features=categorical_features, numeric_features=numeric_features,
                            learning_rate=learning_rate,
                            max_iter=max_iter,
                            max_leaf_nodes=max_leaf_nodes,
                            min_samples_leaf=min_samples_leaf)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and store metrics
    train_accuracy = accuracy_score(y_train, model.predict(X_train)) * 100
    test_accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, -1])

    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "roc_auc": roc_auc,
        "timestamp": datetime.now().strftime(TIMESTAMP_FMT)
    }

    save_model(model, metrics)  # Save the updated model and metrics


    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')  # Use 'binary' if binary classification, 'macro' or 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Calculate the confusion matrix
    confusion = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for easy JSON handling

    # Add classification report as a JSON serializable dictionary
    report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

    # Save plot as base64 to display in HTML
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    roc_img = base64.b64encode(img.getvalue()).decode('utf8')

    # Calculate permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    # Normalize to get percentage importance
    total_importance = sum(result.importances_mean)

    # Check for zero total importance to avoid division by zero
    if total_importance > 0:
        feature_importance_data = [
            (feature, (importance / total_importance) * 100)  # Convert to percentage
            for feature, importance in zip(X.columns, result.importances_mean)
        ]
    else:
        # If total_importance is zero, assign zero importance to all features
        feature_importance_data = [(feature, 0) for feature in X.columns]

    # Ensure importance values are non-negative
    feature_importance_data = [(feature, max(importance, 0)) for feature, importance in feature_importance_data]

    # Sort the feature importance data by percentage importance in descending order
    feature_importance_data = sorted(feature_importance_data, key=lambda x: x[1], reverse=True)

    # Gather metrics and information for history
    entry = {
        "timestamp": datetime.now().strftime(TIMESTAMP_FMT),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "confusion_matrix": confusion,
        "roc_auc": roc_auc,
        "report": report
    }

    # Save this training session's details to history
    save_training_history(entry)

    # Save features to JSON file
    last_trained_features = categorical_features + numeric_features
    save_features_to_json(last_trained_features)

    # Return model and metrics
    return model, accuracy, precision, recall, f1, confusion, report, roc_img, feature_importance_data


@app.route("/api/v1/predict", methods=["POST"])
def predict_api():
    return predict_handler(request)

@app.route("/", methods=['GET'])
def index():
    return (render_template("index.html"))

@app.route("/test-model-form", methods=['GET'])
def test_model_form():
    return (render_template("test.html"))

@app.route('/get_model_features', methods=['GET'])
def get_model_features():
    features = load_features_from_json()
    return jsonify(features)

@app.route("/training-history", methods=["GET"])
def training_history():
    return render_template("history.html")

@app.route("/api/training-history", methods=["GET"])
def get_training_history():
    if os.path.exists(TRAIN_HISTORY_PATH):
        with open(TRAIN_HISTORY_PATH, "r") as f:
            history = json.load(f)
    else:
        history = []

    return jsonify(history), 200

@app.route('/train-action', methods=['GET', 'POST'])
def train_action():
    if request.method == 'POST':
        # Get form data
        file = request.files['file']
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        # Get hyperparameters from the form
        learning_rate = float(request.form['learning_rate'])
        max_iter = int(request.form['max_iter'])
        max_leaf_nodes = int(request.form['max_leaf_nodes'])
        min_samples_leaf = int(request.form['min_samples_leaf'])
        label_column = request.form['labelColumn'].strip()

        # Train the model and get metrics
        model, accuracy, precision, recall, f1, confusion, report, roc_img, feature_importance_data = train_model(label_column, file_path, learning_rate, max_iter, max_leaf_nodes,
                                                             min_samples_leaf)

        return redirect('/training-history')
    return render_template('train.html')






