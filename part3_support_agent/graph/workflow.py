from langgraph.graph import StateGraph, START, END

from part3_support_agent.graph.state import SupportState
from part3_support_agent.graph.guardrail_node import guardrail_node
from part3_support_agent.graph.intent_node import intent_node
from part3_support_agent.graph.retrieval_node import retrieval_node
from part3_support_agent.graph.tool_node import tool_node
from part3_support_agent.graph.response_node import response_node


def route_after_guardrail(state: SupportState) -> str:
    if state.get("blocked", False):
        return "response"

    return "intent"


def route_after_intent(state: SupportState) -> str:
    if state["intent"] == "policy":
        return "retrieval"

    return "tool"


graph_builder = StateGraph(SupportState)

graph_builder.add_node("guardrail", guardrail_node)
graph_builder.add_node("intent", intent_node)
graph_builder.add_node("retrieval", retrieval_node)
graph_builder.add_node("tool", tool_node)
graph_builder.add_node("response", response_node)

graph_builder.add_edge(START, "guardrail")

graph_builder.add_conditional_edges(
    "guardrail",
    route_after_guardrail,
    {
        "intent": "intent",
        "response": "response",
    },
)

graph_builder.add_conditional_edges(
    "intent",
    route_after_intent,
    {
        "retrieval": "retrieval",
        "tool": "tool",
    },
)

graph_builder.add_edge("retrieval", "response")
graph_builder.add_edge("tool", "response")

graph_builder.add_edge("response", END)

support_graph = graph_builder.compile()