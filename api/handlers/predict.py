import os
from typing import Any, Callable

import flask
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def extract_datetime_data_json(data):
    data_and_time = pd.to_datetime(data["Date & Time"])
    data["year"] = data_and_time.year
    data["month"] = data_and_time.month
    data["day"] = data_and_time.day
    data["hour"] = data_and_time.hour
    data["minute"] = data_and_time.minute
    return data

def create_predict_handler(    path: str = os.getenv("MODEL_PATH", "data/pipeline.pkl"),) -> Callable[[flask.Request], flask.Response]:
    """
    This function loads a previously trained model and initialises response labels.

    If then wraps an 'inner' handler function (ensuring above model and response labels
    are in scope for the wrapped function, and that each is initialised exactly once at
    runtime).

    Parameters
    ----------
    path: str
        A path to the target model '.joblib' file.

    Returns
    -------

    """

    model: Pipeline = joblib.load(path)
    statuses = {0: "original", 1: "fake"}

    def handler(request: flask.Request) -> Any:
        request_json = request.get_json()
        request_json = extract_datetime_data_json(request_json)
        df = pd.DataFrame.from_records([request_json])
        yh = model.predict(df)
        return flask.jsonify(dict(order_type=statuses[int(yh[0])]))

    return handler


predict = create_predict_handler()
