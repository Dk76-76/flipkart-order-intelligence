from part3_support_agent.graph.state import SupportState
from part3_support_agent.rag.retriever import retrieve


def retrieval_node(state: SupportState) -> SupportState:
    query = state["query"]

    results = retrieve(
        query,
        top_k=3
    )

    state["retrieved_context"] = results

    return state