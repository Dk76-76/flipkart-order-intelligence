from pathlib import Path

from part3_support_agent.rag.policy_kb import (
    get_retrieval_evaluation_queries
)
from part3_support_agent.rag.retriever import retrieve


def calculate_metrics(
    retrieved_documents,
    relevant_documents,
):
    retrieved_ids = []

    for result in retrieved_documents:
        document_id = result["document_id"]

        if document_id not in retrieved_ids:
            retrieved_ids.append(document_id)

    relevant_retrieved = [
        document_id
        for document_id in retrieved_ids
        if document_id in relevant_documents
    ]

    retrieved_count = len(retrieved_ids)
    relevant_count = len(relevant_documents)
    relevant_retrieved_count = len(relevant_retrieved)

    if retrieved_count == 0:
        precision = 0.0
    else:
        precision = (
            relevant_retrieved_count
            / retrieved_count
        )

    if relevant_count == 0:
        recall = 0.0
    else:
        recall = (
            relevant_retrieved_count
            / relevant_count
        )

    return (
        precision,
        recall,
        retrieved_ids,
        relevant_retrieved,
    )


def evaluate_retrieval():
    evaluation_queries = (
        get_retrieval_evaluation_queries()
    )

    precision_scores = []
    recall_scores = []

    output_lines = []

    output_lines.append(
        "RETRIEVAL EVALUATION"
    )
    output_lines.append("=" * 60)

    for index, item in enumerate(
        evaluation_queries,
        start=1,
    ):
        query = item["query"]
        relevant_documents = item[
            "relevant_documents"
        ]

        results = retrieve(
            query,
            top_k=3,
        )

        (
            precision,
            recall,
            retrieved_ids,
            relevant_retrieved,
        ) = calculate_metrics(
            results,
            relevant_documents,
        )

        precision_scores.append(precision)
        recall_scores.append(recall)

        relevant_count = len(
            relevant_documents
        )

        retrieved_count = len(
            retrieved_ids
        )

        relevant_retrieved_count = len(
            relevant_retrieved
        )

        lines = [
            "",
            f"QUERY {index}",
            "-" * 60,
            f"Query: {query}",
            (
                "Relevant documents: "
                f"{relevant_documents}"
            ),
            (
                "Top-3 retrieved documents: "
                f"{retrieved_ids}"
            ),
            (
                "Relevant retrieved documents: "
                f"{relevant_retrieved}"
            ),
            "",
            "Precision@3 arithmetic:",
            (
                f"{relevant_retrieved_count} "
                f"/ {retrieved_count} "
                f"= {precision:.4f}"
            ),
            "",
            "Recall@3 arithmetic:",
            (
                f"{relevant_retrieved_count} "
                f"/ {relevant_count} "
                f"= {recall:.4f}"
            ),
        ]

        output_lines.extend(lines)

        print("\n".join(lines))

    average_precision = (
        sum(precision_scores)
        / len(precision_scores)
    )

    average_recall = (
        sum(recall_scores)
        / len(recall_scores)
    )

    average_lines = [
        "",
        "=" * 60,
        "AVERAGE RESULTS",
        "=" * 60,
        (
            "Average Precision@3: "
            f"{average_precision:.4f}"
        ),
        (
            "Average Recall@3: "
            f"{average_recall:.4f}"
        ),
    ]

    output_lines.extend(average_lines)

    print("\n".join(average_lines))

    output_file = Path(
        "part3_support_agent/transcripts/"
        "10_retrieval_evaluation.txt"
    )

    output_file.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":
    evaluate_retrieval()