from part3_support_agent.graph.state import SupportState
from part3_support_agent.guardrails import check_groundedness
from part3_support_agent.mock_llm.mock_llm import (
    mock_llm,
    validate_response,
)


def response_node(state: SupportState) -> SupportState:
    if state.get("blocked", False):
        response = {
            "answer": (
                "I can't process this request because it contains "
                "an instruction that conflicts with the support workflow."
            ),
            "source": "policy_kb",
            "confidence": 0.0,
        }

        response = validate_response(response)

    elif state["intent"] == "policy":
        results = state.get("retrieved_context", [])

        grounding = check_groundedness(results)

        state["grounding_score"] = grounding["score"]
        state["grounding_threshold"] = grounding["threshold"]

        if not grounding["grounded"]:
            response = {
                "answer": (
                    "I could not find a sufficiently relevant policy "
                    "in the knowledge base to answer this question."
                ),
                "source": "policy_kb",
                "confidence": 0.0,
            }
        else:
            response = mock_llm(
                intent="policy",
                retrieved_context=results,
                tool_result={},
            )

            response = validate_response(response)

    else:
        response = mock_llm(
            intent=state["intent"],
            retrieved_context=state.get("retrieved_context", []),
            tool_result=state.get("tool_result", {}),
        )

        response = validate_response(response)

    state["answer"] = response["answer"]
    state["source"] = response["source"]
    state["confidence"] = response["confidence"]

    history = state.get("conversation_history", [])

    history.append(
        {
            "query": state["query"],
            "intent": state.get("intent"),
            "answer": state["answer"],
        }
    )

    state["conversation_history"] = history

    return state