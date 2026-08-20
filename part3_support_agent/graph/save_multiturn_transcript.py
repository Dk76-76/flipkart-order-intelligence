from pathlib import Path

from part3_support_agent.graph.workflow import support_graph


state = {
    "query": "What is the return risk for this order?",
    "conversation_history": [],
    "order_features": {
        "price_inr": 2572.0,
        "discount_pct": 23.8,
        "customer_tenure_days": 17,
        "num_previous_orders": 3,
        "num_previous_returns": 0,
        "delivery_distance_km": 604.6,
        "delivery_days": 1,
        "is_weekend_order": 0,
        "rating_given": 2.0,
        "product_category": "Footwear",
        "payment_method": "Prepaid_Card",
    },
}


transcript_lines = []

transcript_lines.append("MULTI-TURN CONVERSATION TEST")
transcript_lines.append("=" * 40)

state = support_graph.invoke(state)

transcript_lines.append("TURN 1")
transcript_lines.append("-" * 40)
transcript_lines.append(f"Query: {state['query']}")
transcript_lines.append(f"Intent: {state.get('intent')}")
transcript_lines.append(f"Answer: {state.get('answer')}")
transcript_lines.append(f"History: {state.get('conversation_history')}")

state["query"] = "What about that order?"

state = support_graph.invoke(state)

transcript_lines.append("")
transcript_lines.append("TURN 2")
transcript_lines.append("-" * 40)
transcript_lines.append(f"Query: {state['query']}")
transcript_lines.append(f"Intent: {state.get('intent')}")
transcript_lines.append(f"Answer: {state.get('answer')}")
transcript_lines.append(f"History: {state.get('conversation_history')}")

transcript = "\n".join(transcript_lines)

output_file = Path(
    "part3_support_agent/transcripts/05_multiturn_state.txt"
)

output_file.write_text(
    transcript,
    encoding="utf-8",
)

print(transcript)
print(f"\nSaved to: {output_file}")