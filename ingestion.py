import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import hashlib

load_dotenv()

CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def extract_text_from_pdf(pdf_path: str) -> list[dict]:

    """
    Returns list of dicts: {text, page_number, doc_name}
    """
    pages = []
    doc_name = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page_number": page_num,
                    "doc_name": doc_name
                })
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Chunks page's text, preserving metadata.
    Returns list of chunk dicts with chunk_id.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size= CHUNK_SIZE,
        chunk_overlap= CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        split_texts = splitter.split_text(page["text"])
        for idx, chunk_text in enumerate(split_texts):
            chunk_id = hashlib.md5(f"{page['doc_name']}_p{page['page_number']}_c{idx}".encode()).hexdigest()
            chunks.append({
                "text": chunk_text,
                "doc_name": page["doc_name"],
                "page_number": page["page_number"],
                "chunk_id": chunk_id
            })
    return chunks


def embed_and_store(chunks: list[dict], collection) -> int:
    """
    Embeds chunks and stores into ChromaDB.
    Returns count of chunks stored.
    """

    embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "doc_name": c["doc_name"],
            "page_number": c["page_number"],
            "chunk_id": c["chunk_id"]
        }
        for c in chunks
    ]

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        embeddings = embedder.embed_documents(batch_texts)

        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas
        )

    return len(chunks)


def ingest_pdf(pdf_path: str) -> dict:
    collection = get_chroma_collection()
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return {"status": "error", "message": "No text extracted from PDF"}
    
    chunks = chunk_pages(pages)
    count = embed_and_store(chunks, collection)

    return {
        "status": "success",
        "doc_name": os.path.basename(pdf_path),
        "pages_processed": len(pages),
        "chunks_stored": count
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <path_to_pdf>")
        sys.exit(1)
    
    result = ingest_pdf(sys.argv[1])
    print(result)