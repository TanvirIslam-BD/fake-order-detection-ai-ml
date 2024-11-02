import base64
import re
import io
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

from app.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, DATE_FEATURES, TIMESTAMP_FMT
from app.model.model import create_pipeline
from app.utils.utils import extract_datetime_features, load_model, save_model, save_training_history, \
    save_features_to_json, save_last_trained_features
from datetime import datetime


def train_model(request, file_path):

    # Extract parameters from the request
    learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, label_column = extract_parameters(request)

    # Load and prepare data
    data, X, y, numeric_features, categorical_features = load_and_prepare_data(file_path, label_column, request)

    # Train the model and evaluate
    model, metrics, y_pred, y_test, X_test = train_and_evaluate_model(X, y, learning_rate, max_iter, max_leaf_nodes,
                                                              min_samples_leaf, categorical_features, numeric_features)

    # Generate metrics
    performance_metrics = calculate_metrics(X_test, y_test, y_pred, model)

    # Save training history and features
    save_training_data(metrics, performance_metrics, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf)

    return model, *performance_metrics.values()

def extract_parameters(request):
    learning_rate = float(request.form['learning_rate'])
    max_iter = int(request.form['max_iter'])
    max_leaf_nodes = int(request.form['max_leaf_nodes'])
    min_samples_leaf = int(request.form['min_samples_leaf'])
    label_column = request.form['labelColumn'].strip()
    return learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, label_column


def load_and_prepare_data(file_path, label_column, request):
    data = pd.read_csv(file_path)
    # Clean price columns
    clean_price_data(data)

    # Extract features from the request
    categorical_features = request.form.getlist('categoricalFields')
    date_features = request.form.getlist('dateFields')
    numeric_features = request.form.getlist('numericFields')
    numeric_features = numeric_features + DATE_FEATURES

    save_last_trained_features(categorical_features, numeric_features, date_features)

    # Extract datetime features
    data_frame = extract_datetime_features(data)
    X = data_frame[categorical_features + numeric_features]
    y = data[label_column]

    return data, X, y, numeric_features, categorical_features


def clean_price_data(data):
    data['Amount (Total Price)'] = data['Amount (Total Price)'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x.strip())) if isinstance(x, str) else x
    )
    data['Coupon amount'] = data['Coupon amount'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x.strip())) if isinstance(x, str) else x
    )


def train_and_evaluate_model(X, y, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, categorical_features, numeric_features):

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = load_model()  # Load existing model if present
    model = model or create_pipeline(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf
    )

    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print("Error during model fitting:", e)
        print("X_train columns:", X_train.columns.tolist())
        print("Expected categorical features:", categorical_features)
        print("Expected numeric features:", numeric_features)
        raise  # Re-raise the exception after logging

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)) * 100,
        "test_accuracy": accuracy_score(y_test, y_pred) * 100,
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, -1]),
        "timestamp": datetime.now().strftime(TIMESTAMP_FMT)
    }

    save_model(model, metrics)  # Save the updated model and metrics

    return model, metrics, y_pred, y_test, X_test


def calculate_metrics(X_test, y_test, y_pred, model):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    confusion = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Calculate permutation importance
    feature_importance_data = calculate_permutation_importance(model, X_test, y_test)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion": confusion,
        "report": report,
        "roc_auc": roc_auc,
        "feature_importance_data": feature_importance_data
    }


def plot_roc_curve(fpr, tpr, roc_auc):
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
    return base64.b64encode(img.getvalue()).decode('utf8')

def calculate_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    total_importance = sum(result.importances_mean)

    if total_importance > 0:
        return [
            (feature, (importance / total_importance) * 100)
            for feature, importance in zip(X_test.columns, result.importances_mean)
        ]
    else:
        return [(feature, 0) for feature in X_test.columns]

def save_training_data(metrics, performance_metrics, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf):
    entry = {
        "timestamp": datetime.now().strftime(TIMESTAMP_FMT),
        **performance_metrics,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
    }

    # Save this training session's details to history
    save_training_history(entry)

    # Save features to JSON file
    last_trained_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    save_features_to_json(last_trained_features)