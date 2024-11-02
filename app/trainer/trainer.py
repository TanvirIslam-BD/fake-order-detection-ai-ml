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
    save_features_to_json
from datetime import datetime


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