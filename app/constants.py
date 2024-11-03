import os
from typing import List

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TIMESTAMP_FMT = "%m-%d-%Y, %H:%M:%S"

MODEL_PATH = "data/pipeline.pkl"  # Path to save/load model
METRICS_PATH = "data/metrics.json"  # Path to save metrics
TRAIN_HISTORY_PATH = "data/train_history.json"
features_file = 'data/last_trained_features.json'

last_trained_features = []

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
    "Booking type"
]


DATE_FEATURES: List[str] = [
    "year",
    "month",
    "day",
    "hour",
    "minute"
]

HYPERPARAMETERS  = {
    "learning_rate": 0.05,        # Moderately low learning rate for gradual learning
    "max_iter": 250,              # Sufficient iterations for convergence at this learning rate
    "max_leaf_nodes": 20,         # Balanced tree complexity
    "min_samples_leaf": 15        # Ensures each leaf has enough samples for generalization
}