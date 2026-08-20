from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "orders_dataset.csv"


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

model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy: {accuracy:.2%}")
print(f"F1-score (returned=1): {f1:.2f}")
print("Recall for returned=1: 0.00")

print(
    "\nHigh accuracy is misleading here because the baseline "
    "predicts every order as not returned. It gets many correct "
    "predictions, but has zero recall for returned orders."
)