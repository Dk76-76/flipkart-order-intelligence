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


print("TURN 1")
print("-" * 40)

state = support_graph.invoke(state)

print("Query:", state["query"])
print("Intent:", state["intent"])
print("Answer:", state["answer"])
print("History:", state["conversation_history"])


print()
print("TURN 2")
print("-" * 40)

state["query"] = "What about that order?"

state = support_graph.invoke(state)

print("Query:", state["query"])
print("Intent:", state["intent"])
print("Answer:", state["answer"])
print("History:", state["conversation_history"])