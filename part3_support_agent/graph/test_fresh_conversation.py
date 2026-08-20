from part3_support_agent.graph.workflow import support_graph


state = {
    "query": "What about that order?",
    "conversation_history": [],
}


result = support_graph.invoke(state)


print("FRESH CONVERSATION")
print("-" * 40)
print("Query:", result["query"])
print("Intent:", result["intent"])
print("Answer:", result["answer"])
print("History:", result["conversation_history"])