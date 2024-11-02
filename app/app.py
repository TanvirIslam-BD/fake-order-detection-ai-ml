import numpy as np
from flask import Flask
from .errors import errors

app = Flask(__name__)
# app.register_blueprint(errors)

np.random.seed(42)




