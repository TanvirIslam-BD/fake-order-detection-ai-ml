import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class PriceTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to clean the price column."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Clean the 'Amount (Total Price)' column
        X["Amount (Total Price)"] = (
            X["Amount (Total Price)"]
            .replace(r'[^0-9.]', '', regex=True)  # Remove any character except numeric values and decimal points
            .replace(r'[$,]', '', regex=True)  # Remove $ and comma
            .replace(r'\s+', '', regex=True)  # Remove any whitespace
        )
        # Convert to numeric type
        X["Amount (Total Price)"] = pd.to_numeric(X["Amount (Total Price)"], errors='coerce')

        # Clean the 'Coupon amount' column
        X["Coupon amount"] = (
            X["Coupon amount"]
            .replace(r'[^0-9.]', '', regex=True)  # Remove any character except numeric values and decimal points
            .replace(r'[$,]', '', regex=True)  # Remove $ and comma
            .replace(r'\s+', '', regex=True)  # Remove any whitespace
        )
        # Convert to numeric type
        X["Coupon amount"] = pd.to_numeric(X["Coupon amount"], errors='coerce')

        return X


