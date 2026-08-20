from pathlib import Path

from part3_support_agent.graph.workflow import support_graph


query = "What is the return window for apparel products?"

state = {
    "query": query,
    "conversation_history": [],
}


result = support_graph.invoke(state)

transcript = f"""POLICY RAG TEST 1
========================================
Query: {query}
Intent: {result.get("intent")}
Answer: {result.get("answer")}
Source: {result.get("source")}
Confidence: {result.get("confidence")}
"""

output_file = Path(
    "part3_support_agent/transcripts/01_policy_apparel.txt"
)

output_file.write_text(
    transcript,
    encoding="utf-8",
)

print(transcript)
print(f"Saved to: {output_file}")