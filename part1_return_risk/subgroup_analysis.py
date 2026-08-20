from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "orders_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "return_risk_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "subgroup_analysis"

FINAL_THRESHOLD = 0.42


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["returned", "order_id"])
    y = df["returned"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= FINAL_THRESHOLD).astype(int)

    test_results = X_test.copy()
    test_results["actual"] = y_test.values
    test_results["prediction"] = predictions

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    category_results = []

    for category, group in test_results.groupby("product_category"):
        category_results.append(
            {
                "product_category": category,
                "orders": len(group),
                "precision": precision_score(
                    group["actual"],
                    group["prediction"],
                    zero_division=0,
                ),
                "recall": recall_score(
                    group["actual"],
                    group["prediction"],
                    zero_division=0,
                ),
            }
        )

    category_df = pd.DataFrame(category_results)

    payment_results = []

    for payment, group in test_results.groupby("payment_method"):
        payment_results.append(
            {
                "payment_method": payment,
                "orders": len(group),
                "precision": precision_score(
                    group["actual"],
                    group["prediction"],
                    zero_division=0,
                ),
                "recall": recall_score(
                    group["actual"],
                    group["prediction"],
                    zero_division=0,
                ),
            }
        )

    payment_df = pd.DataFrame(payment_results)

    category_df.to_csv(
        OUTPUT_DIR / "model_performance_by_category.csv",
        index=False,
    )

    payment_df.to_csv(
        OUTPUT_DIR / "model_performance_by_payment.csv",
        index=False,
    )

    print("Final threshold:", FINAL_THRESHOLD)

    print("\nModel performance by product category:")
    print(category_df.round(4).to_string(index=False))

    print("\nModel performance by payment method:")
    print(payment_df.round(4).to_string(index=False))

    print("\nOutputs saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()