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
from typing import Tuple, List, Dict


def train_model(request, file_path):

    (learning_rate, max_iter, max_leaf_nodes,
     min_samples_leaf, label_column) = extract_parameters(request)

    (data, feature_matrix_x, target_vector_y,
     numeric_features, categorical_features) = load_and_prepare_data(file_path, label_column, request)

    (model, target_vector_y_test, feature_matrix_x_test) = initiate_model_training(feature_matrix_x, target_vector_y, learning_rate, max_iter, max_leaf_nodes,
                                                              min_samples_leaf, categorical_features, numeric_features)

    performance_metrics = calculate_model_performance(feature_matrix_x_test, target_vector_y_test, model)

    save_training_data(performance_metrics, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf)

    return model, *performance_metrics.values()

def extract_parameters(request):
    try:
        learning_rate = float(request.form['learning_rate'])
        max_iter = int(request.form['max_iter'])
        max_leaf_nodes = int(request.form['max_leaf_nodes'])
        min_samples_leaf = int(request.form['min_samples_leaf'])
        label_column = request.form['labelColumn'].strip()
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid parameter in request: {e}")

    return learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, label_column


def load_and_prepare_data( file_path: str,  label_column: str,  request: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], List[str]]:

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or invalid format.")

    clean_price_data(data)

    categorical_features = request.form.getlist('categoricalFields')
    date_features = request.form.getlist('dateFields')
    numeric_features = request.form.getlist('numericFields')

    missing_columns = [col for col in categorical_features + numeric_features + [label_column] if
                       col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")

    try:
        numeric_features = numeric_features + DATE_FEATURES
    except NameError:
        raise ValueError("DATE_FEATURES is not defined in the current scope.")

    save_last_trained_features(categorical_features, numeric_features, date_features)

    data_frame = extract_datetime_features(data)

    feature_matrix_x = data_frame[categorical_features + numeric_features]
    target_vector_y = data[label_column]

    return data, feature_matrix_x, target_vector_y, numeric_features, categorical_features


def clean_price_data(data):
    data['Amount (Total Price)'] = data['Amount (Total Price)'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x.strip())) if isinstance(x, str) else x
    )
    data['Coupon amount'] = data['Coupon amount'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', x.strip())) if isinstance(x, str) else x
    )


def initiate_model_training(feature_matrix_x, target_vector_y, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, categorical_features, numeric_features):

    (feature_matrix_x_train, feature_matrix_x_test,
     target_vector_y_train, target_vector_y_test) = train_test_split(feature_matrix_x, target_vector_y, test_size=0.2, random_state=42)

    model = load_model()  # Load existing model if present
    # model = model or create_pipeline(
    #     categorical_features=categorical_features,
    #     numeric_features=numeric_features,
    #     learning_rate=learning_rate,
    #     max_iter=max_iter,
    #     max_leaf_nodes=max_leaf_nodes,
    #     min_samples_leaf=min_samples_leaf
    # )

    model = create_pipeline(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf
    )

    try:
        model.fit(feature_matrix_x_train, target_vector_y_train)
    except ValueError as e:
        print("Error during model fitting:", e)
        print("X_train columns:", feature_matrix_x_train.columns.tolist())
        print("Expected categorical features:", categorical_features)
        print("Expected numeric features:", numeric_features)
        raise

    save_model(model)

    return model, target_vector_y_test, feature_matrix_x_test


def calculate_model_performance(feature_matrix_x_test, y_test, model):

    target_vector_y_prediction = model.predict(feature_matrix_x_test)

    accuracy = accuracy_score(y_test, target_vector_y_prediction)
    precision = precision_score(y_test, target_vector_y_prediction, average='binary')
    recall = recall_score(y_test, target_vector_y_prediction, average='binary')
    f1 = f1_score(y_test, target_vector_y_prediction, average='binary')
    confusion = confusion_matrix(y_test, target_vector_y_prediction).tolist()
    report = classification_report(y_test, target_vector_y_prediction, output_dict=True)

    y_pred_proba = model.predict_proba(feature_matrix_x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    feature_importance_data = calculate_permutation_importance(model, feature_matrix_x_test, y_test)

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

def save_training_data(performance_metrics, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf):
    entry = {
        "timestamp": datetime.now().strftime(TIMESTAMP_FMT),
        **performance_metrics,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
    }

    save_training_history(entry)

    last_trained_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    save_features_to_json(last_trained_features)