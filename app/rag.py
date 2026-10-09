from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "data" / "chroma"


# -----------------------------
# Embedding model
# -----------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="foximize_knowledge"
)


# -----------------------------
# Text Chunking
# -----------------------------

def chunk_text(text, chunk_size=800):
    """
    Split text into smaller chunks.
    """

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


# -----------------------------
# Build / Refresh Database
# -----------------------------

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

    # No files found
    if not documents:

        print("No knowledge-base documents found.")

        return


    # -----------------------------
    # Remove old database records
    # -----------------------------
    # Remove all existing records
    existing = collection.get()

    existing_ids = existing.get("ids", [])

    if existing_ids:
        collection.delete(ids=existing_ids)

    # -----------------------------
    # Generate embeddings
    # -----------------------------

    print(
        f"Creating embeddings for {len(documents)} chunks..."
    )

    embeddings = embedding_model.encode(
        documents
    ).tolist()


    # -----------------------------
    # Insert fresh data
    # -----------------------------

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


    print(
        f"Knowledge base refreshed successfully."
    )

    print(
        f"Total chunks: {len(documents)}"
    )


# -----------------------------
# Search Knowledge
# -----------------------------

def search_knowledge(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]


    results_list = []


    for document, metadata in zip(
        documents,
        metadatas
    ):

        source = "unknown"

        if metadata and "source" in metadata:

            source = metadata["source"]


        results_list.append({
            "text": document,
            "source": source
        })


    return results_list


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    build_database()

    results = search_knowledge(
        "What is the fee for Data Science?"
    )

    for result in results:

        print(
            "\nSOURCE:",
            result["source"]
        )

        print(
            result["text"]
        )