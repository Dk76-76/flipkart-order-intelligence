from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "orders_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "return_risk_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "threshold_tuning"


def main():
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

    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]

    default_predictions = (probabilities >= 0.50).astype(int)

    default_precision = precision_score(
        y_test, default_predictions, zero_division=0
    )
    default_recall = recall_score(
        y_test, default_predictions, zero_division=0
    )
    default_f1 = f1_score(
        y_test, default_predictions, zero_division=0
    )

    results = []

    for threshold in [round(0.10 + i * 0.02, 2) for i in range(41)]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "recall_gain": recall - default_recall,
        })

    results_df = pd.DataFrame(results)

    best_f1 = results_df.loc[results_df["f1"].idxmax()]

    candidates = results_df[
        results_df["recall_gain"] >= 0.15
    ].sort_values(
        ["f1", "recall"],
        ascending=False,
    )

    if candidates.empty:
        raise RuntimeError(
            "No threshold gives the required recall improvement."
        )

    operational = candidates.iloc[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        OUTPUT_DIR / "random_forest_threshold_results.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "random_forest_threshold_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Random Forest Threshold Analysis\n")
        file.write(f"Default threshold: 0.50\n")
        file.write(f"Default precision: {default_precision:.4f}\n")
        file.write(f"Default recall: {default_recall:.4f}\n")
        file.write(f"Default F1: {default_f1:.4f}\n")
        file.write(
            f"F1-optimal threshold: {best_f1['threshold']:.2f}\n"
        )
        file.write(
            f"Operational threshold: {operational['threshold']:.2f}\n"
        )
        file.write(
            f"Operational precision: {operational['precision']:.4f}\n"
        )
        file.write(
            f"Operational recall: {operational['recall']:.4f}\n"
        )
        file.write(
            f"Operational F1: {operational['f1']:.4f}\n"
        )
        file.write(
            f"Recall improvement: "
            f"{operational['recall_gain'] * 100:.2f} percentage points\n"
        )

    print(f"Default F1: {default_f1:.4f}")
    print(f"F1-optimal threshold: {best_f1['threshold']:.2f}")
    print(f"Operational threshold: {operational['threshold']:.2f}")
    print(f"Operational precision: {operational['precision']:.4f}")
    print(f"Operational recall: {operational['recall']:.4f}")
    print(
        f"Recall improvement: "
        f"{operational['recall_gain'] * 100:.2f} percentage points"
    )


if __name__ == "__main__":
    main()