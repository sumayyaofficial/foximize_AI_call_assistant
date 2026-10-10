from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "data" / "chroma"


# -----------------------------
# Lazy Singletons for Models and DB
# -----------------------------

_embedding_model = None
_client = None
_collection = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
    return _embedding_model


def get_chroma_client():
    global _client
    if _client is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client_inst = get_chroma_client()
        _collection = client_inst.get_or_create_collection(name="foximize_knowledge")
    return _collection


class _LazyProxy:
    def __init__(self, getter):
        self._getter = getter

    def __getattr__(self, name):
        return getattr(self._getter(), name)


embedding_model = _LazyProxy(get_embedding_model)
client = _LazyProxy(get_chroma_client)
collection = _LazyProxy(get_collection)


# -----------------------------
# Text Chunking
# -----------------------------

def chunk_text(text, chunk_size=600, overlap=100):
    """
    Split text into boundary-aware chunks respecting paragraphs and sentences.
    Prevents splitting words, phone numbers, or fees across chunks.
    """
    text = text.strip()
    if not text:
        return []

    # First split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If single paragraph is larger than chunk_size, split by lines or sentences
        if len(paragraph) > chunk_size:
            lines = [l.strip() for l in paragraph.split("\n") if l.strip()]
            for line in lines:
                if len(current_chunk) + len(line) + 1 <= chunk_size:
                    current_chunk = f"{current_chunk}\n{line}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = line
        else:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{paragraph}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

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
    if not query or not query.strip():
        return []

    coll = get_collection()
    total_docs = coll.count()
    if total_docs == 0:
        return []

    emb_model = get_embedding_model()
    query_embedding = emb_model.encode(
        [query.strip()]
    ).tolist()

    n_results = min(top_k, total_docs)
    results = coll.query(
        query_embeddings=query_embedding,
        n_results=n_results
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