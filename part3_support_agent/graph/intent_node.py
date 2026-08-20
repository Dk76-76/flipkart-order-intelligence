from part3_support_agent.graph.state import SupportState


def intent_node(state: SupportState) -> SupportState:
    query = state["query"].lower()
    history = state.get("conversation_history", [])

    if "return risk" in query or "return probability" in query:
        intent = "return_risk"

    elif (
        "image" in query
        or "product category" in query
        or "classify" in query
    ):
        intent = "image_classification"

    elif (
        "that order" in query
        or "this order" in query
        or "same order" in query
    ):
        previous_intent = None

        for message in reversed(history):
            if isinstance(message, dict) and message.get("intent"):
                previous_intent = message["intent"]
                break

        intent = previous_intent or "policy"

    else:
        intent = "policy"

    state["intent"] = intent

    return state