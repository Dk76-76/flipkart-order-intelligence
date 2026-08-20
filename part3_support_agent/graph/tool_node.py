from part3_support_agent.graph.state import SupportState
from part3_support_agent.tools.check_return_risk import check_return_risk
from part3_support_agent.tools.classify_product_image import (
    classify_product_image,
)


def tool_node(state: SupportState) -> SupportState:
    intent = state["intent"]

    if intent == "return_risk":
        order_features = state.get("order_features")

        if not order_features:
            state["tool_result"] = {
                "error": "Order details are required for return-risk analysis."
            }
            return state

        result = check_return_risk(order_features)

    elif intent == "image_classification":
        image_path = state.get("image_path")

        if not image_path:
            state["tool_result"] = {
                "error": (
                    "Please upload a product image using the "
                    "Product Classifier before requesting image classification."
                )
            }
            return state

        result = classify_product_image(image_path)

    else:
        result = {}

    state["tool_result"] = result

    return state