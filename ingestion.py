import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import hashlib
import re

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


def clean_extracted_text(text: str) -> str:
    text = re.sub(r'([a-zA-Z])\n([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([●○•])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([a-z]{2,})([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'^\s*[o•●○]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_garbage_page(text: str) -> bool:
    # Page is garbage if same sentence repeats more than 3 times
    # or if it contains EOS/pad tokens (attention visualization pages)
    if text.count("<EOS>") > 2:
        return True
    if text.count("<pad>") > 2:
        return True
    # Check if any single line repeats more than 3 times
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if len(line) > 10 and lines.count(line) > 3:
            return True
    return False


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    pages = []
    doc_name = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text or not text.strip():
            continue
        text = clean_extracted_text(text)
        if len(text.strip()) < 100:
            continue
        if is_garbage_page(text):
            print(f"Skipping garbage page {page_num}")
            continue
        pages.append({
            "text": text,
            "page_number": page_num,
            "doc_name": doc_name
        })
    doc.close()
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

    embedder = HuggingFaceEmbeddings()

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