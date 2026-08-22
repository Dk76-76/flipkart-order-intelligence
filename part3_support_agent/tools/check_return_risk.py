from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "return_risk_model.pkl"

RISK_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 0.65

FEATURES = [
    "product_category",
    "price_inr",
    "discount_pct",
    "payment_method",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]


def check_return_risk(order_features: dict) -> dict:
    missing_features = [
        feature
        for feature in FEATURES
        if feature not in order_features
    ]

    if missing_features:
        raise ValueError(
            "Missing required order features: "
            + ", ".join(missing_features)
        )

    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame(
        [[order_features[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    return_probability = model.predict_proba(
        input_data
    )[0, 1]

    if return_probability < RISK_THRESHOLD:
        risk_bucket = "Low"
    elif return_probability < HIGH_RISK_THRESHOLD:
        risk_bucket = "Medium"
    else:
        risk_bucket = "High"

    return {
        "return_probability": round(
            float(return_probability),
            4,
        ),
        "risk_bucket": risk_bucket,
    }


if __name__ == "__main__":
    order_features = {
        "product_category": "Footwear",
        "price_inr": 2572.0,
        "discount_pct": 23.8,
        "payment_method": "Prepaid_Card",
        "customer_tenure_days": 17,
        "num_previous_orders": 3,
        "num_previous_returns": 0,
        "delivery_distance_km": 604.6,
        "delivery_days": 1,
        "is_weekend_order": 0,
        "rating_given": 2.0,
    }

    result = check_return_risk(order_features)

    print("Return probability:", result["return_probability"])
    print("Risk bucket:", result["risk_bucket"])