from flask import Response, render_template
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from services import PriceTransformer
import re

app = Flask(__name__)
# app.register_blueprint(errors)

# Constants
np.random.seed(42)
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
def create_pipeline(categorical_features: List[str], numeric_features: List[str]):

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
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", HistGradientBoostingClassifier(random_state=42))])



@app.route("/data-info", methods=["POST"])
def data_info():
    data = request.json
    path = r"data/eb-order-data-final.csv"

    # Load dataset
    print(f"read_csv path {path}")
    data_frame = pd.read_csv(path)
    print(f"Data types and non-null values {data_frame.info()}") # Get data types and non-null values

    print(f"statistical summary {data_frame.describe()}")
    print(f"Preview the first few rows {data_frame.head()}")

    return Response("OK", status=200)

# Train route
@app.route("/train", methods=["POST"])
def train():
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



@app.route("/predict", methods=["POST"])
def predict():
    return predict_handler(request)


# feel free to add as many handlers in here as you like!


@app.route("/health")
def health():
    return Response("OK", status=200)


@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")
