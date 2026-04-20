from langchain_community.vectorstores import FAISS
from importlib.metadata import PackageNotFoundError, version

_db = None
_db_init_error = None


def _torch_is_compatible():
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        return False

    parts = torch_version.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor) >= (2, 4)


def _load_vector_db():
    global _db, _db_init_error

    if _db is not None:
        return _db
    if _db_init_error is not None:
        raise RuntimeError(_db_init_error)

    try:
        if not _torch_is_compatible():
            raise RuntimeError(
                "PyTorch >= 2.4 is required for local HuggingFace embeddings."
            )

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return _db
    except Exception as exc:
        _db_init_error = str(exc)
        raise RuntimeError(_db_init_error) from exc


# 🔍 RETRIEVER
def retriever_node(state):
    query = state["query"]

    try:
        docs = _load_vector_db().similarity_search(query, k=3)
        contents = [d.page_content for d in docs]
        retrieval_error = None
    except Exception as exc:
        contents = []
        retrieval_error = (
            "Retriever unavailable. Check FAISS index and embedding dependencies. "
            f"Details: {exc}"
        )

    return {
        **state,
        "retrieved_docs": contents,
        "retrieval_error": retrieval_error,
    }


# 🧠 ANALYST
from app.llm import llm

def analyst_node(state):
    query = state["query"]
    docs = state["retrieved_docs"]

    context = "\n\n".join(docs)

    prompt = f"""
        You are an expert research assistant.

        Use ONLY the provided context to answer the question.

        Context:
        {context}

        Question:
        {query}

        Give a clear, structured answer.
        """

    response = llm.invoke(prompt)

    return {
        **state,
        "draft_answer": response.content
    }


# 🔎 CRITIC
def critic_node(state):
    docs = state["retrieved_docs"]
    answer = state["draft_answer"]

    context = "\n\n".join(docs)

    prompt = f"""
    You are a strict evaluator.

    Context:
    {context}

    Answer:
    {answer}

    Evaluate:
    1. Is answer supported by context?
    2. Any hallucination?
    3. Missing info?

    Return:
    Score (0-1)
    Feedback
    """

    response = llm.invoke(prompt).content

    try:
        score = float(response.split("Score")[1].split("\n")[0].split(":")[1])
    except:
        score = 0.5

    return {
        **state,
        "critique_score": score,
        "critique_feedback": response,
        "revision_count": state.get("revision_count", 0) + 1
    }
