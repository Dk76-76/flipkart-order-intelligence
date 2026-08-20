from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "orders_dataset.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    X = df.drop(columns=["returned", "order_id"])
    y = df["returned"]
    return X, y


def build_preprocessor():
    numerical_features = [
        "price_inr",
        "discount_pct",
        "customer_tenure_days",
        "num_previous_orders",
        "num_previous_returns",
        "delivery_distance_km",
        "delivery_days",
        "is_weekend_order",
        "rating_given",
    ]

    categorical_features = [
        "product_category",
        "payment_method",
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def main():
    df = load_data()
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Dataset:", df.shape)
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Features before preprocessing:", X_train.shape[1])
    print("Features after preprocessing:", X_train_processed.shape[1])
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()