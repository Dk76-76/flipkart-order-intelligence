from part3_support_agent.mock_llm.prompt import get_system_prompt


ALLOWED_SOURCES = {
    "policy_kb",
    "return_risk_tool",
    "image_classifier_tool",
}


def mock_llm(
    intent: str,
    retrieved_context: list | None = None,
    tool_result: dict | None = None,
) -> dict:
    """
    Deterministic response generator.

    This function does not make any network calls and does not
    require an API key.
    """

    # Load the prompt so the 4S and role instructions are part
    # of the response-generation configuration.
    get_system_prompt()

    retrieved_context = retrieved_context or []
    tool_result = tool_result or {}

    if intent == "return_risk":
        if not tool_result:
            return {
                "answer": "Return-risk information is unavailable.",
                "source": "return_risk_tool",
                "confidence": 0.0,
            }

        probability = tool_result["return_probability"]
        risk_bucket = tool_result["risk_bucket"]

        answer = (
            f"Return probability is {probability:.4f}. "
            f"Risk level is {risk_bucket}."
        )

        return {
            "answer": answer,
            "source": "return_risk_tool",
            "confidence": 1.0,
        }

    if intent == "image_classification":
        if not tool_result:
            return {
                "answer": "Product classification is unavailable.",
                "source": "image_classifier_tool",
                "confidence": 0.0,
            }

        predicted_class = tool_result["class"]
        confidence = float(tool_result["confidence"])

        answer = (
            f"The product is classified as "
            f"{predicted_class} with "
            f"{confidence:.4f} confidence."
        )

        return {
            "answer": answer,
            "source": "image_classifier_tool",
            "confidence": round(confidence, 4),
        }

    if intent == "policy":
        if not retrieved_context:
            return {
                "answer": "I could not find a relevant policy.",
                "source": "policy_kb",
                "confidence": 0.0,
            }

        best_result = retrieved_context[0]

        answer = best_result["text"]
        confidence = float(best_result["score"])

        return {
            "answer": answer,
            "source": "policy_kb",
            "confidence": round(confidence, 4),
        }

    return {
        "answer": "I could not determine the appropriate support response.",
        "source": "policy_kb",
        "confidence": 0.0,
    }


def validate_response(response: dict) -> dict:
    """
    Validate the fixed response schema required by the project.
    """

    required_fields = {
        "answer",
        "source",
        "confidence",
    }

    if set(response.keys()) != required_fields:
        raise ValueError(
            "Response must contain exactly: "
            "answer, source, confidence"
        )

    if response["source"] not in ALLOWED_SOURCES:
        raise ValueError(
            f"Invalid source: {response['source']}"
        )

    if not isinstance(response["answer"], str):
        raise TypeError("answer must be a string")

    if not isinstance(response["confidence"], (int, float)):
        raise TypeError("confidence must be numeric")

    return response