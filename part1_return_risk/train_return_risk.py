from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "orders_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_return_risk_model.pkl"


df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["returned", "order_id"])
y = df["returned"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numerical_features = X_train.select_dtypes(include="number").columns
categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns

preprocessor = ColumnTransformer([
    (
        "numerical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numerical_features,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_features,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    ),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, probabilities)
returned_f1 = f1_score(y_test, predictions, zero_division=0)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Returned F1: {returned_f1:.4f}")
print(f"ROC-AUC >= 0.58: {'PASS' if roc_auc >= 0.58 else 'FAIL'}")
print(f"Returned F1 >= 0.30: {'PASS' if returned_f1 >= 0.30 else 'FAIL'}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"Model saved: {MODEL_PATH}")