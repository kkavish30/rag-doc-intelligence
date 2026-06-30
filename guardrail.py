RELEVANCE_THRESHOLD = 0.3
OUT_OF_SCOPE_MESSAGE = """I couldn't find sufficiently relevant information in the 
uploaded documents to answer your question. Please rephrase your query 
or ensure the relevant document has been uploaded."""


def check_relevance(reranked_docs: list[dict]) -> dict:
    if not reranked_docs:
        return {
            "pass": False,
            "top_score": 0.0,
            "message": OUT_OF_SCOPE_MESSAGE
        }
    
    top_score = reranked_docs[0].get("rerank_score", 0.0)

    if top_score < RELEVANCE_THRESHOLD:
        return {
            "pass": False,
            "top_score": top_score,
            "message": OUT_OF_SCOPE_MESSAGE
        }
    
    return {
        "pass": True,
        "top_score": top_score,
    }


def is_query_relevant(reranked_docs: list[dict]) -> bool:
    return check_relevance(reranked_docs)["pass"]


if __name__ == "__main__":
    mock_irrelevant = [{"rerank_score": 0.12, "text": "Some doc"}]
    result = check_relevance(mock_irrelevant)
    print(f"Irrelevant: pass={result['pass']}, score={result['top_score']}")

    mock_relevant = [{"rerank_score": 8.75, "text": "Very relevant doc"}]
    result = check_relevance(mock_relevant)
    print(f"Relevant: pass={result['pass']}, score={result['top_score']}")