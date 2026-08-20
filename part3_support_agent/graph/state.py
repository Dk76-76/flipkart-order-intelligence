from typing import TypedDict


class SupportState(TypedDict, total=False):
    query: str
    conversation_history: list
    intent: str

    order_features: dict
    image_path: str

    retrieved_context: list
    tool_result: dict

    answer: str
    source: str
    confidence: float

    blocked: bool
    guardrail_message: str

    grounding_score: float
    grounding_threshold: float