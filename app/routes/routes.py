import os
from flask import Blueprint, request, jsonify, render_template, json, redirect
from app import app
from app.constants import TRAIN_HISTORY_PATH
from app.handlers.predict import predict_handler
from app.trainer.trainer import train_model
from app.utils.utils import load_features_from_json


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


