from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "orders_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Return rate: {df['returned'].mean():.2%}")

print("\nReturn rate by product category:")
print(
    df.groupby("product_category")["returned"]
    .mean()
    .sort_values(ascending=False)
    .map(lambda x: f"{x:.2%}")
)

print("\nReturn rate by payment method:")
print(
    df.groupby("payment_method")["returned"]
    .mean()
    .sort_values(ascending=False)
    .map(lambda x: f"{x:.2%}")
)


for column in ["product_category", "payment_method"]:
    return_rate = (
        df.groupby(column)["returned"]
        .mean()
        .sort_values(ascending=False)
    )

    return_rate.plot(kind="bar")
    plt.title(f"Return Rate by {column}")
    plt.ylabel("Return Rate")
    plt.xlabel(column)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"return_rate_by_{column}.png",
        dpi=150
    )
    plt.close()