import os
import streamlit as st
import tempfile
from ingestion import ingest_pdf, get_chroma_collection
from retrieval import retrieve_and_rerank
from generation import generate_answer
from guardrail import check_relevance
from memory import SessionMemory

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def load_retrieval_models():
    """
    Pre-loads the cross-encoder and embedding model at startup.
    Without this, models reload on every user interaction — 3-4 second delay per query.
    """
    from retrieval import get_cross_encoder
    from langchain_huggingface import HuggingFaceEmbeddings

    encoder = get_cross_encoder()
    embedder = HuggingFaceEmbeddings()
    return encoder, embedder



def init_session():
    if "memory" not in st.session_state:
        st.session_state.memory = SessionMemory()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ingested_docs" not in st.session_state:
        st.session_state.ingested_docs = []

    
init_session()

def sync_ingested_docs():
    """
    On startup/refresh, check what's actually in ChromaDB
    and restore the ingested_docs list in session state.
    """
    if st.session_state.ingested_docs:
        return  # already populated, no need to sync
    
    try:
        collection = get_chroma_collection()
        if collection.count() > 0:
            # Get unique doc names from metadata
            result = collection.get(include=["metadatas"])
            doc_names = list(set(
                m["doc_name"] for m in result["metadatas"]
            ))
            st.session_state.ingested_docs = sorted(doc_names)
    except Exception:
        pass  # collection doesn't exist yet — fine, first run

sync_ingested_docs()

with st.spinner("Loading models..."):
    load_retrieval_models()



with st.sidebar:
    st.title("📃 Documents")
    st.markdown("Upload pdf to begin chatting.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        if st.button("Ingest documents", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                result = ingest_pdf(tmp_path, original_filename=uploaded_file.name)
                os.unlink(tmp_path)

                if result["status"] == "success":
                    if uploaded_file.name not in st.session_state.ingested_docs:
                        st.session_state.ingested_docs.append(uploaded_file.name)
                    st.success(
                        f"✅ {uploaded_file.name}: "
                        f"{result['chunks_stored']} chunks from "
                        f"{result['pages_processed']} pages"
                    )
                else:
                    st.error(f"❌ {uploaded_file.name}: {result['message']}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.empty()

    
    if st.session_state.ingested_docs:
        st.markdown("---")
        st.markdown("**Loaded documents:**")
        for doc in st.session_state.ingested_docs:
            st.markdown(f"- 📄 `{doc}`")

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.chat_history = []
        st.rerun()

    if os.path.exists("eval_results.txt"):
        st.markdown("---")
        st.markdown("**Pipeline Evaluation (RAGAS)**")
        with open("eval_results.txt", encoding="utf-8") as f:
            st.code(f.read(), language=None)


st.title("📚 RAG Document Intelligence")
st.markdown("Ask questions about your uploaded documents.")

if not st.session_state.ingested_docs:
    st.info("👈 Upload and ingest at least one PDF using the sidebar to get started.")
    st.stop()

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander(f"📎 {len(turn['sources'])} source(s)"):
                for src in turn["sources"]:
                    st.markdown(
                        f"- **{src['doc_name']}** — Page {src['page_number']}"
                    )


if prompt := st.chat_input("Ask a question about your documents..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "sources": None
    })

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            docs = retrieve_and_rerank(prompt)
            guard = check_relevance(docs)

        if not guard["pass"]:
            response_text = guard["message"]
            sources = []
            st.warning(response_text)
        else:
            with st.spinner("Generating answer..."):
                history = st.session_state.memory.get_history()
                result = generate_answer(prompt, docs, history)
                response_text = result["answer"]
                sources = result["sources"]

            st.markdown(response_text)

            if sources:
                with st.expander(f"📎 {len(sources)} source(s)"):
                    for src in sources:
                        st.markdown(
                            f"- **{src['doc_name']}** — Page {src['page_number']}"
                        )

            st.caption(
                f"Model: {result['model']} | "
                f"Tokens: {result['tokens_used']} | "
                f"Reranker score: {guard['top_score']:.3f}"
            )

            st.session_state.memory.add_turn(prompt, response_text)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources
    })