from pathlib import Path

from part3_support_agent.graph.workflow import support_graph


query = "What is the policy for international drone deliveries?"

state = {
    "query": query,
    "conversation_history": [],
}


result = support_graph.invoke(state)

transcript = f"""UNGROUNDED POLICY TEST
========================================
Query: {query}
Intent: {result.get("intent")}
Retrieved similarity: {result.get("grounding_score")}
Grounding threshold: {result.get("grounding_threshold")}
Answer: {result.get("answer")}
Source: {result.get("source")}
Confidence: {result.get("confidence")}
Decision: REFUSED
"""

output_file = Path(
    "part3_support_agent/transcripts/08_ungrounded_policy.txt"
)

output_file.write_text(
    transcript,
    encoding="utf-8",
)

print(transcript)
print(f"Saved to: {output_file}")