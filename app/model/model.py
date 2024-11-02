from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Create the pipeline function
def create_pipeline(categorical_features: List[str], numeric_features: List[str],learning_rate, max_iter, max_leaf_nodes, min_samples_leaf):

    # Use StandardScaler for numeric features if scaling is desired (not strictly necessary for tree-based models)
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    # One-hot encode categorical features
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant")), ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
    )

    # Combine the transformations for price, datetime, numeric, and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf
    )

    # Use HistGradientBoostingClassifier
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
