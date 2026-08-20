from pathlib import Path

from part3_support_agent.graph.workflow import support_graph


query = "What product category is this image?"

state = {
    "query": query,
    "conversation_history": [],
    "image_path": "data/sample_images/01_pullover.png",
}


result = support_graph.invoke(state)

transcript = f"""IMAGE CLASSIFICATION TEST
========================================
Query: {query}
Image: data/sample_images/01_pullover.png
Intent: {result.get("intent")}
Answer: {result.get("answer")}
Source: {result.get("source")}
Confidence: {result.get("confidence")}
"""

output_file = Path(
    "part3_support_agent/transcripts/04_image_classification.txt"
)

output_file.write_text(
    transcript,
    encoding="utf-8",
)

print(transcript)
print(f"Saved to: {output_file}")