import re


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?rules",
    r"ignore\s+the\s+instructions",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you\s+are",
]

GROUNDING_THRESHOLD = 0.50


def check_input(query: str) -> dict:
    """
    Check user input for common prompt-injection patterns.
    """

    normalized_query = query.lower().strip()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized_query):
            return {
                "blocked": True,
                "reason": "Potential prompt injection detected.",
            }

    return {
        "blocked": False,
        "reason": "",
    }


def guard_input(query: str) -> None:
    """
    Raise an error when prompt injection is detected.
    """

    result = check_input(query)

    if result["blocked"]:
        raise ValueError(
            "Request blocked: potential prompt injection detected."
        )


def check_groundedness(results: list) -> dict:
    """
    Check whether the best retrieved policy chunk
    meets the minimum similarity threshold.
    """

    if not results:
        return {
            "grounded": False,
            "score": 0.0,
            "threshold": GROUNDING_THRESHOLD,
        }

    best_score = max(
        float(result["score"])
        for result in results
    )

    return {
        "grounded": best_score >= GROUNDING_THRESHOLD,
        "score": round(best_score, 4),
        "threshold": GROUNDING_THRESHOLD,
    }