from part3_support_agent.graph.workflow import support_graph


state = {
    "query": "What product category is this image?",
    "conversation_history": [],
    "image_path": "data/sample_images/01_pullover.png",
}

result = support_graph.invoke(state)

print("Query:", result["query"])
print("Intent:", result["intent"])
print("Answer:", result["answer"])
print("Source:", result["source"])
print("Confidence:", result["confidence"])