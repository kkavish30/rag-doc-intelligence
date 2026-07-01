from langchain_groq import ChatGroq
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024
)

SYSTEM_PROMPT = """You are a precise document assistant. You answer questions ONLY using the provided context chunks from the user's documents.

Rules:
1. Base every claim on the provided context. Do not use external knowledge.
2. After every piece of information you use, cite it in this exact format: (Source: filename, Page X)
3. If different chunks provide conflicting information, note the conflict and cite both.
4. If the context does not contain enough information to answer, say so clearly.
5. Keep answers concise and well-structured.
"""


def format_context(reranked_docs: list[dict], max_chars_per_chunk: int = 800) -> str:
    context_parts = []
    for i, doc in enumerate(reranked_docs, 1):
        meta = doc["metadata"]
        text = doc["text"][:max_chars_per_chunk]
        context_parts.append(
            f"[Chunk {i}] Source: {meta["doc_name"]}, Page {meta["page_number"]}\n"
            f"{text}"
        )
    return "\n\n---\n\n".join(context_parts)


def build_message(query: str, context: str, conversation_history: list[dict]) -> list:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for turn in conversation_history[-12:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    user_content = f"""Context from documents: {context}
                ---
                Question: {query}
                Answer based only on the context above. Cite each source as (Source: filename, Page X)."""

    messages.append(HumanMessage(content=user_content))
    return messages


def generate_answer(query: str, reranked_docs: list[dict], conversation_history: list[dict]) -> list:
    if conversation_history is None:
        conversation_history = []

    context = format_context(reranked_docs)
    messages = build_message(query, context, conversation_history)

    response = llm.invoke(messages)
    answer = response.content     
    tokens_used = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)

    sources = []
    seen = set()
    for doc in reranked_docs:
        key = (doc["metadata"]["doc_name"], doc["metadata"]["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "doc_name": doc["metadata"]["doc_name"],
                "page_number": doc["metadata"]["page_number"]
            })

    return {
        "answer": answer,
        "sources": sources,
        "model": "llama-3.3-70b-versatile",
        "tokens_used": tokens_used
    }


if __name__ == "__main__":
    from retrieval import retrieve_and_rerank
    from guardrail import check_relevance

    query = "What is the transformer architecture?"

    docs = retrieve_and_rerank(query)
    guard = check_relevance(docs)

    if not guard["pass"]:
        print(f"GUARDRAIL: {guard['message']}")
    else:
        result = generate_answer(query, docs, [])
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used: {result['sources']}")
        print(f"Tokens: {result['tokens_used']}")