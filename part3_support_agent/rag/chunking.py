from policy_kb import get_policy_documents


def split_into_sentences(text):
    sentences = []

    for sentence in text.split("."):
        sentence = sentence.strip()

        if sentence:
            sentences.append(sentence + ".")

    return sentences


def create_chunks():
    chunks = []

    for document in get_policy_documents():
        sentences = split_into_sentences(document["text"])

        for sentence_number, sentence in enumerate(sentences, start=1):
            chunks.append(
                {
                    "chunk_id": f'{document["document_id"]}_chunk_{sentence_number}',
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "text": sentence,
                }
            )

    return chunks


if __name__ == "__main__":
    chunks = create_chunks()

    print("Total chunks:", len(chunks))

    for chunk in chunks:
        print(
            chunk["chunk_id"],
            "|",
            chunk["document_id"],
            "|",
            chunk["text"],
        )