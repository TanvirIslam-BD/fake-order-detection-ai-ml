import os
from typing import List

# Constants

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
TIMESTAMP_FMT = "%m-%d-%Y, %H:%M:%S"

MODEL_PATH = "data/pipeline.pkl"  # Path to save/load model
METRICS_PATH = "data/metrics.json"  # Path to save metrics

# Path to save training history
TRAIN_HISTORY_PATH = "data/train_history.json"


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


DATE_FEATURES: List[str] = [
    "year",
    "month",
    "day",
    "hour",
    "minute"
]