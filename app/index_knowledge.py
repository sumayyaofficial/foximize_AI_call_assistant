from app.rag import build_database


if __name__ == "__main__":

    print("Refreshing Foximize knowledge base...")

    build_database()

    print("Knowledge base refresh completed.")

    
def build_database():

    print("Refreshing Foximize knowledge base...")

    documents = []
    ids = []
    metadatas = []

    # Read all knowledge-base files
    for file_path in KNOWLEDGE_BASE_DIR.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            documents.append(chunk)

            ids.append(
                f"{file_path.stem}_{index}"
            )

            metadatas.append({
                "source": file_path.name
            })

    if not documents:
        print("No knowledge-base documents found.")
        return

    # Remove old records
    existing = collection.get()

    existing_ids = existing.get("ids", [])

    if existing_ids:
        collection.delete(ids=existing_ids)

    print(
        f"Creating embeddings for {len(documents)} chunks..."
    )

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Knowledge base refreshed successfully.")
    print(f"Total chunks: {len(documents)}")