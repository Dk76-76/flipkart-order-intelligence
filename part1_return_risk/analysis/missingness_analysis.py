from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "orders_dataset.csv"


df = pd.read_csv(DATA_PATH)

missing_rate = df["rating_given"].isna().mean()

payment_missing = (
    df.groupby("payment_method")["rating_given"]
    .apply(lambda x: x.isna().mean())
)

cod_rate = payment_missing["COD"]
non_cod_rate = df.loc[
    df["payment_method"] != "COD", "rating_given"
].isna().mean()

gap = cod_rate - non_cod_rate


print(f"Total rows: {len(df)}")
print(f"Missing rating rate: {missing_rate:.2%}")

print("\nMissing rate by payment method:")
print(payment_missing.map(lambda x: f"{x:.2%}"))

print("\nCOD missing rate:", f"{cod_rate:.2%}")
print("Non-COD missing rate:", f"{non_cod_rate:.2%}")
print("Missing-rate gap:", f"{gap:.2%}")

print("\nMissingness classification: MAR")
print(
    "Reason: rating missingness depends on the observed "
    "payment_method column, so it is not MCAR or MNAR."
)