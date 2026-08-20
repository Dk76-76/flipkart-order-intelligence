from pathlib import Path

from part3_support_agent.graph.workflow import support_graph


query = "What is the return risk for this order?"

state = {
    "query": query,
    "conversation_history": [],
    "order_features": {
        "order_value": 2499,
        "delivery_days": 5,
        "product_category": "Apparel",
        "payment_method": "COD",
        "customer_order_count": 3,
        "customer_return_count": 1,
    },
}


result = support_graph.invoke(state)

transcript = f"""RETURN RISK TEST
========================================
Query: {query}
Intent: {result.get("intent")}
Answer: {result.get("answer")}
Source: {result.get("source")}
Confidence: {result.get("confidence")}
"""

output_file = Path(
    "part3_support_agent/transcripts/03_return_risk.txt"
)

output_file.write_text(
    transcript,
    encoding="utf-8",
)

print(transcript)
print(f"Saved to: {output_file}")