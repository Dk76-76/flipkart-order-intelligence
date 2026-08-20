POLICY_DOCUMENTS = [
    {
        "document_id": "apparel_returns",
        "title": "Apparel Return Policy",
        "text": (
            "Apparel products can be returned within 7 days of delivery. "
            "The product should be unused and have its original tags attached."
        ),
    },
    {
        "document_id": "footwear_returns",
        "title": "Footwear Return Policy",
        "text": (
            "Footwear products can be returned within 7 days of delivery. "
            "The footwear should be unused and returned with its original packaging."
        ),
    },
    {
        "document_id": "electronics_returns",
        "title": "Electronics Return Policy",
        "text": (
            "Eligible electronics products can be returned within 7 days of delivery. "
            "Some products may require troubleshooting or verification before a return is approved."
        ),
    },
    {
        "document_id": "home_returns",
        "title": "Home Product Return Policy",
        "text": (
            "Eligible home products can generally be returned within 7 days of delivery. "
            "The product should be unused and available for inspection during the return process."
        ),
    },
    {
        "document_id": "cod_refunds",
        "title": "COD Refund Policy",
        "text": (
            "For a returned COD order, the refund is processed after the returned product passes the required verification. "
            "The refund is sent to the customer's selected eligible refund method."
        ),
    },
    {
        "document_id": "delivery_sla",
        "title": "Delivery SLA Policy",
        "text": (
            "The estimated delivery date shown for an order is based on the destination and available delivery service. "
            "Delivery can take longer when there are operational or logistics delays."
        ),
    },
    {
        "document_id": "reverse_pickup",
        "title": "Reverse Pickup Eligibility",
        "text": (
            "Reverse pickup is available when the product and order are eligible for return and the service is available at the delivery location. "
            "The product must be packed according to the return instructions before pickup."
        ),
    },
    {
        "document_id": "damaged_product",
        "title": "Damaged Product Policy",
        "text": (
            "Customers should report a product that arrives damaged as soon as possible through the support process. "
            "The product may be inspected before a replacement, return, or refund is approved."
        ),
    },
    {
        "document_id": "wrong_product",
        "title": "Wrong Product Policy",
        "text": (
            "If a customer receives a product different from the ordered item, the issue should be reported through the support process. "
            "The order and product may be verified before the return or replacement is processed."
        ),
    },
    {
        "document_id": "exchange_policy",
        "title": "Exchange Policy",
        "text": (
            "Exchange is available only for products and orders that are eligible for exchange. "
            "Availability can depend on the product, size or variant, and stock at the time of the request."
        ),
    },
    {
        "document_id": "prepaid_refunds",
        "title": "Prepaid Refund Policy",
        "text": (
            "For an eligible prepaid order, the refund is initiated after the return is approved and the required verification is completed. "
            "The refund is processed back through the applicable payment or refund method."
        ),
    },
    {
        "document_id": "order_cancellation",
        "title": "Order Cancellation Policy",
        "text": (
            "An order can be cancelled when the cancellation option is available for that order. "
            "Cancellation availability can change after the order moves further into the fulfilment process."
        ),
    },
]


RETRIEVAL_EVALUATION = [
    {
        "query": "What is the return window for apparel products?",
        "relevant_documents": ["apparel_returns"],
    },
    {
        "query": "How long can I return an electronic product?",
        "relevant_documents": ["electronics_returns"],
    },
    {
        "query": "How does a COD refund work after a return?",
        "relevant_documents": ["cod_refunds"],
    },
    {
        "query": "When is reverse pickup available?",
        "relevant_documents": ["reverse_pickup"],
    },
    {
        "query": "What happens if I receive the wrong product?",
        "relevant_documents": ["wrong_product"],
    },
    {
        "query": "What is the expected delivery time for an order?",
        "relevant_documents": ["delivery_sla"],
    },
]


def get_policy_documents():
    return POLICY_DOCUMENTS


def get_retrieval_evaluation_queries():
    return RETRIEVAL_EVALUATION


if __name__ == "__main__":
    print("Policy documents:", len(POLICY_DOCUMENTS))
    print("Evaluation queries:", len(RETRIEVAL_EVALUATION))

    for document in POLICY_DOCUMENTS:
        print(document["document_id"], "-", document["title"])