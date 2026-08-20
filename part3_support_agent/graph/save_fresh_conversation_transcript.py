from pathlib import Path

from part3_support_agent.graph.workflow import support_graph


state = {
    "query": "What about that order?",
    "conversation_history": [],
}


result = support_graph.invoke(state)

transcript = f"""FRESH CONVERSATION TEST
========================================
Query: {state["query"]}
Intent: {result.get("intent")}
Answer: {result.get("answer")}
Source: {result.get("source")}
Confidence: {result.get("confidence")}
Initial history: []
Final history: {result.get("conversation_history")}
"""

output_file = Path(
    "part3_support_agent/transcripts/06_fresh_conversation.txt"
)

output_file.write_text(
    transcript,
    encoding="utf-8",
)

print(transcript)
print(f"Saved to: {output_file}")