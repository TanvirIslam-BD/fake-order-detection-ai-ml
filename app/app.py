import base64
import io
import os
import pickle

from flask import Response, render_template, url_for, flash, redirect
from flask_cors import CORS
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance

from .handlers import predict as predict_handler

import json
import time
from datetime import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from services import PriceTransformer
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from .model import train_model

app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
# app.register_blueprint(errors)
# Constants
np.random.seed(42)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


TIMESTAMP_FMT = "%m-%d-%Y, %H:%M:%S"

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
def create_pipeline(categorical_features: List[str], numeric_features: List[str],
                    learning_rate,
                    max_iter,
                    max_leaf_nodes,
                    min_samples_leaf):

    # Use StandardScaler for numeric features if scaling is desired (not strictly necessary for tree-based models)
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    # One-hot encode categorical features
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant")), ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
    )

    # # Preprocess datetime features
    # datetime_transformer = FunctionTransformer(extract_datetime_features, validate=False)
    #
    # # Preprocess the price data
    # price_transformer = PriceTransformer()

    # Combine the transformations for price, datetime, numeric, and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            # ("price", price_transformer, PRICE_FEATURES),
            # ("datetime", datetime_transformer, DATE_FEATURES),
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Use HistGradientBoostingClassifier
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf
    ))])

# Train route
@app.route("/api/train", methods=["POST"])
def train_api():
    data = request.json
    # path = data.get("path")
    # path = os.getenv("MODEL_PATH", "data/pipeline.pkl"),
    # model_path = os.getenv("MODEL_PATH", "data/pipeline.pkl"),
    # metrics_path  = os.getenv("METRICS_PATH", "data/metrics.json"),
    path = r"data/eb-order-data-final.csv"
    model_path = r"data/pipeline.pkl"
    metrics_path = r"data/pmetrics.json"
    # model_path = data.get("model_path", "data/pipeline.pkl")
    # metrics_path = data.get("metrics_path", "data/metrics.json")
    test_size = data.get("test_size", 0.2)
    dump = data.get("dump", True)

    categorical_features = data.get("categorical_features", CATEGORICAL_FEATURES)
    numeric_features = data.get("numeric_features", NUMERIC_FEATURES + DATE_FEATURES)
    label = data.get("label", LABEL)

    start = time.time()

    # Load dataset
    print(f"read_csv path {path}")
    data_frame = pd.read_csv(path)

    # Clean the price data directly
    data_frame['Amount (Total Price)'] = data_frame['Amount (Total Price)'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x.strip())) if isinstance(x, str) else x
    )

   # Clean the price data directly
    data_frame['Coupon amount'] = data_frame['Coupon amount'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x.strip())) if isinstance(x, str) else x
    )

    data_frame = extract_datetime_features(data_frame)

    features = data_frame[categorical_features + numeric_features]
    target = data_frame[label]

    # Train test split by scikit-learn
    # tx: The training set for the input features.
    # vx: The test/validation set for the input features.
    # ty: The training set for the target (label).
    # vy: The test/validation set for the target (label).
    tx, vx, ty, vy = train_test_split(features, target, test_size=test_size, random_state=42)

    # Assuming X is a sparse matrix

    # Create model and train

    # model the pipeline object

    # price_transformer = PriceTransformer()
    # tx  = price_transformer.transform(data_frame)
    # print("price transformed_data")
    # print(tx)
    # print(tx.dtypes)  # Check the data types of the columns
    # print(tx.isnull().sum())  # Check for NaN values in the transformed data

    model = create_pipeline(categorical_features=categorical_features, numeric_features=numeric_features)
    model.fit(tx, ty)

    end = time.time()

    # Calculate metrics

    #Score on train set
    train_accuracy = accuracy_score(model.predict(tx), ty) * 100

    # Score on test set
    test_accuracy = accuracy_score(model.predict(vx), vy) * 100

    #  ROC AUC score on test set
    roc_auc = roc_auc_score(vy, model.predict_proba(vx)[:, -1])

    metrics = dict(
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        roc_auc=roc_auc,
        elapsed_time=end - start,
        timestamp=datetime.now().strftime(TIMESTAMP_FMT),
    )

    # Save model and metrics
    if dump:
        joblib.dump(model, model_path)
        json.dump(metrics, open(metrics_path, "w"))

    return jsonify(metrics), 200

@app.route("/api/v1/predict", methods=["POST"])
def predict_api():
    return predict_handler(request)

@app.route("/", methods=['GET'])
def index():
    return (render_template("index.html"))


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


    # Initialize HistGradientBoostingClassifier with provided hyperparameters
    # model = HistGradientBoostingClassifier(
    #     learning_rate=learning_rate,
    #     max_iter=max_iter,
    #     max_leaf_nodes=max_leaf_nodes,
    #     min_samples_leaf=min_samples_leaf
    # )
    model = create_pipeline(categorical_features=categorical_features, numeric_features=numeric_features,
                            learning_rate=learning_rate,
                            max_iter=max_iter,
                            max_leaf_nodes=max_leaf_nodes,
                            min_samples_leaf=min_samples_leaf)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,
                                average='binary')  # Use 'binary' if binary classification, 'macro' or 'weighted' for multiclass
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

    # Return model and metrics
    return model, accuracy, precision, recall, f1, confusion, report, roc_img, feature_importance_data

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

        return render_template('results.html',model=model, accuracy=accuracy, precision=precision, recall=recall,
                               f1=f1, confusion=confusion, report=report, roc_img=roc_img, feature_importance_data=feature_importance_data)

    return render_template('train.html')
