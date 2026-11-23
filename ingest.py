import glob
import os
import sqlite3

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PINECONE_INDEX,
)

from embeddings.sentence_transformer import SentenceEmbedding
from loaders.pdf_loader import PDFLoader
from splitters.text_splitter import TextSplitter
from vectorstores.pinecone_store import PineconeStore


# ============================================================
# ADD DOCUMENT METADATA
# ============================================================

def add_document_metadata(documents, document_id, filename):
    """
    Add document information to every chunk.

    This metadata is stored along with the vector in Pinecone.
    It allows us to later retrieve chunks from only one PDF.
    """

    for document in documents:

        if not hasattr(document, "metadata"):
            document.metadata = {}

        document.metadata["document_id"] = str(document_id)
        document.metadata["filename"] = filename

    return documents


# ============================================================
# INGEST ONE PDF
# ============================================================

def ingest_document(pdf_path, document_id):
    """
    Load, split and store one PDF in Pinecone.

    Every chunk gets:
        document_id
        filename
    as Pinecone metadata.
    """

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    filename = os.path.basename(pdf_path)

    print()
    print("=" * 60)
    print(f"Ingesting: {filename}")
    print(f"Document ID: {document_id}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    pdf_loader = PDFLoader()

    loaded_documents = pdf_loader.load(
        pdf_path
    )

    print(
        f"Loaded {len(loaded_documents)} document section(s)."
    )

    # --------------------------------------------------------
    # Split PDF into chunks
    # --------------------------------------------------------

    text_splitter = TextSplitter(
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    documents = text_splitter.split(
        loaded_documents
    )

    print(
        f"Created {len(documents)} chunks."
    )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    documents = add_document_metadata(
        documents,
        document_id,
        filename
    )

    print("Added Pinecone metadata:")
    print(f"  document_id = {document_id}")
    print(f"  filename    = {filename}")

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embedding = SentenceEmbedding(
        EMBEDDING_MODEL
    ).get_model()

    # --------------------------------------------------------
    # Connect to Pinecone
    # --------------------------------------------------------

    vectorstore = PineconeStore(
        PINECONE_INDEX,
        embedding
    )

    # --------------------------------------------------------
    # Store vectors + metadata
    # --------------------------------------------------------

    vectorstore.create_store(
        documents
    )

    print()
    print(
        f"Successfully ingested {len(documents)} "
        f"chunks from {filename}."
    )

    return len(documents)


# ============================================================
# GET DOCUMENT ID FROM SQLITE
# ============================================================

def get_document_id(filename):
    """
    Find the SQLite document ID using the uploaded filename.
    """

    conn = sqlite3.connect(
        "users.db"
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id
        FROM documents
        WHERE filename = ?
        ORDER BY id DESC
        LIMIT 1
    """, (
        filename,
    ))

    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Look for PDFs in data/raw
    # --------------------------------------------------------

    pdf_paths = sorted(
        glob.glob(
            os.path.join(
                "data",
                "raw",
                "*.pdf"
            )
        )
    )

    if not pdf_paths:

        raise SystemExit(
            "No PDF files found in data/raw."
        )

    total_chunks = 0

    # --------------------------------------------------------
    # Process each PDF
    # --------------------------------------------------------

    for pdf_path in pdf_paths:

        filename = os.path.basename(
            pdf_path
        )

        document_id = get_document_id(
            filename
        )

        if document_id is None:

            print()
            print(
                f"Skipping {filename}: "
                "document not found in users.db."
            )

            continue

        chunks = ingest_document(
            pdf_path,
            document_id
        )

        total_chunks += chunks

    print()
    print("=" * 60)
    print(
        f"Total chunks ingested: {total_chunks}"
    )
    print(
        f"Total PDF files processed: {len(pdf_paths)}"
    )
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()