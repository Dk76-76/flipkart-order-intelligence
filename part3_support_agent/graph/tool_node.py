from part3_support_agent.graph.state import SupportState
from part3_support_agent.tools.check_return_risk import check_return_risk
from part3_support_agent.tools.classify_product_image import classify_product_image


def tool_node(state: SupportState) -> SupportState:
    intent = state["intent"]

    if intent == "return_risk":
        result = check_return_risk(
            state["order_features"]
        )

    elif intent == "image_classification":
        result = classify_product_image(
            state["image_path"]
        )

    else:
        result = {}

    state["tool_result"] = result

    return state