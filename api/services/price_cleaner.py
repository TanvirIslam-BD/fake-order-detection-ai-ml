import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PriceCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer to clean the price column."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Clean the 'price' column
        X["price"] = X["price"].str.replace(r'[^0-9.]', '',
                                            regex=True)  # Remove any character except numeric values and decimal points
        X["price"] = pd.to_numeric(X["price"])  # Convert to numeric type
        return X
