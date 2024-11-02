import io
import os

import pandas as pd
from flask import request, jsonify, render_template, json, redirect, Blueprint
from app import app
from app.constants import TRAIN_HISTORY_PATH
from app.trainer.trainer import train_model
from app.utils.utils import load_features_from_json

# Create the blueprint
main = Blueprint('main', __name__)

@main.route("/", methods=['GET'])
def index():
    return (render_template("index.html"))

@main.route("/test-model-form", methods=['GET'])
def test_model_form():
    return (render_template("test.html"))

@main.route('/get_model_features', methods=['GET'])
def get_model_features():
    features = load_features_from_json()
    return jsonify(features)

@main.route("/training-history", methods=["GET"])
def training_history():
    return render_template("history.html")


@main.route("/api/training-history", methods=["GET"])
def get_training_history():
    if os.path.exists(TRAIN_HISTORY_PATH):
        with open(TRAIN_HISTORY_PATH, "r") as f:
            history = json.load(f)
    else:
        history = []

    return jsonify(history), 200

@main.route('/train-action', methods=['GET', 'POST'])
def train_action():

    if request.method == 'POST':

        file = request.files['file']
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        learning_rate = float(request.form['learning_rate'])
        max_iter = int(request.form['max_iter'])
        max_leaf_nodes = int(request.form['max_leaf_nodes'])
        min_samples_leaf = int(request.form['min_samples_leaf'])
        label_column = request.form['labelColumn'].strip()

        model, accuracy, precision, recall, f1, confusion, report, roc_img, feature_importance_data = train_model(label_column, file_path, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf)

        return redirect('/training-history')

    return render_template('index.html')


@main.route("/data-set-info", methods=["POST"])
def get_data_set_info():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    column_names = df.columns.tolist()

    summary = {
        "head": df.head().to_html(),
        "info": info_str,
        "column_names": column_names,
        "describe": df.describe(include="all").to_html(),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": df.nunique().to_dict()
    }

    return jsonify(summary), 200


