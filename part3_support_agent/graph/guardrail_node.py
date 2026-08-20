from part3_support_agent.graph.state import SupportState
from part3_support_agent.guardrails import check_input


def guardrail_node(state: SupportState) -> SupportState:
    query = state["query"]

    result = check_input(query)

    state["blocked"] = result["blocked"]
    state["guardrail_message"] = result["reason"]

    if result["blocked"]:
        state["answer"] = (
            "I can't process this request because it contains "
            "an instruction that conflicts with the support workflow."
        )
        state["source"] = "policy_kb"
        state["confidence"] = 0.0

    return state