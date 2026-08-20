from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "orders_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "return_risk_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainability"


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
    test_auc = roc_auc_score(y_test, probabilities)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    permutation_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": permutation.importances_mean,
        "std": permutation.importances_std,
    })

    permutation_df = permutation_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    permutation_df.to_csv(
        OUTPUT_DIR / "permutation_importance.csv",
        index=False,
    )

    top_features = importance_df.head(10).sort_values("importance")

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Return Risk Features")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "top_feature_importance.png",
        dpi=150,
    )
    plt.close()

    with open(
        OUTPUT_DIR / "explainability_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("MODEL EXPLAINABILITY SUMMARY\n\n")
        file.write(f"Test ROC-AUC: {test_auc:.4f}\n\n")

        file.write("Top 5 features:\n")

        for i, row in importance_df.head(5).iterrows():
            file.write(
                f"{i + 1}. {row['feature']}: "
                f"{row['importance']:.4f}\n"
            )

        file.write("\nPermutation importance:\n")

        for i, row in permutation_df.head(5).iterrows():
            file.write(
                f"{i + 1}. {row['feature']}: "
                f"{row['importance']:.4f}\n"
            )

    print(f"ROC-AUC: {test_auc:.4f}")
    print("Top 5 features:")

    for i, row in importance_df.head(5).iterrows():
        print(
            f"{i + 1}. {row['feature']}: "
            f"{row['importance']:.4f}"
        )

    print(f"Outputs saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()