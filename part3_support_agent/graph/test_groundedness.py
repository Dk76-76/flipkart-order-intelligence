from part3_support_agent.graph.workflow import support_graph


state = {
    "query": "What is the policy for international drone deliveries?",
    "conversation_history": [],
}


result = support_graph.invoke(state)


print("UNGROUNDED POLICY TEST")
print("-" * 40)
print("Query:", state["query"])
print("Intent:", result.get("intent"))
print("Retrieved similarity:", result.get("grounding_score"))
print("Grounding threshold:", result.get("grounding_threshold"))
print("Answer:", result.get("answer"))
print("Source:", result.get("source"))
print("Confidence:", result.get("confidence"))

if (
    result.get("grounding_score", 0.0)
    < result.get("grounding_threshold", 0.50)
):
    print("Decision: REFUSED")
else:
    print("Decision: ANSWERED")