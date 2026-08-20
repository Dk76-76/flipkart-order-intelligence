from part3_support_agent.mock_llm.mock_llm import (
    mock_llm,
    validate_response,
)


print("POLICY TEST")
print("-" * 40)

policy_result = mock_llm(
    intent="policy",
    retrieved_context=[
        {
            "text": "Apparel products can be returned within 7 days of delivery.",
            "score": 0.6955,
        }
    ],
)

print(policy_result)
print(validate_response(policy_result))


print()
print("RETURN RISK TEST")
print("-" * 40)

risk_result = mock_llm(
    intent="return_risk",
    tool_result={
        "return_probability": 0.4489,
        "risk_bucket": "Medium",
    },
)

print(risk_result)
print(validate_response(risk_result))


print()
print("IMAGE CLASSIFICATION TEST")
print("-" * 40)

image_result = mock_llm(
    intent="image_classification",
    tool_result={
        "class": "Pullover",
        "confidence": 0.9757,
    },
)

print(image_result)
print(validate_response(image_result))