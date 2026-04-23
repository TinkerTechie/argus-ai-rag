# Argus: Multi-Agent Verified Retrieval-Augmented Generation

Argus is a stateful research assistant designed to provide high-accuracy answers by combining multi-agent orchestration with local LLM inference. Unlike standard RAG systems that follow a linear "retrieve-and-generate" path, Argus utilizes a directed graph workflow (via LangGraph) to reason, evaluate, and self-correct its responses before they reach the user.

## Core Methodology

The system is built on the principle of "verification through collaboration." By separating concerns into specialized agents, the system can handle complex queries that require more than just a simple database lookup.

### The Multi-Agent Pipeline

1. **Router**: Analyzes the incoming query to determine the best processing path. It distinguishes between general greetings (which bypass retrieval) and technical research questions.
2. **Query Expander**: Generates multiple semantic variations of the user's question to ensure broader coverage during the retrieval phase.
3. **Retriever**: Performs semantic search against a Qdrant vector database to fetch relevant document chunks.
4. **Analyst**: Synthesizes the retrieved context and the expanded queries to draft a comprehensive, grounded answer.
5. **Critic**: Acts as an internal quality gate. It evaluates the Analyst's draft for faithfulness to the source text and identifies any potential hallucinations. If the quality score is low, it triggers a revision loop.

## Technical Architecture

The project is designed to run entirely on local hardware, ensuring data privacy and eliminating external API dependencies.

- **Orchestration**: LangGraph manages the state and transitions between agents.
- **Vector Database**: Qdrant is used for high-performance semantic search and metadata filtering.
- **Inference Engine**: Ollama handles local execution of models like Phi3 or Llama3.
- **Embeddings**: SentenceTransformers (specifically all-MiniLM-L6-v2) for generating semantic vectors.
- **Backend**: FastAPI provides an asynchronous interface with thread-offloading for long-running graph tasks.

## Performance and Concurrency

Argus is optimized for responsiveness even during heavy LLM workloads:
- **Asynchronous Handling**: The FastAPI layer is fully async, ensuring the server remains reachable while processing requests.
- **Thread Management**: Since LLM inference is CPU/GPU intensive, the system uses `asyncio.to_thread` to offload graph execution, preventing blocking of the main event loop.
- **Infrastructure Note**: While the API layer scales horizontally, local LLM inference is typically sequential. For high-volume production use, we recommend a distributed inference backend.

## Getting Started

### Prerequisites
- Python 3.10+
- Ollama (running locally)
- Qdrant (or local storage enabled)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/TinkerTechie/argus-ai-rag.git
cd argus-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the vector store:
```bash
python -m app.rag.vectorstore
```

4. Launch the application:
```bash
# Start the API
uvicorn app.api:app --host 127.0.0.1 --port 8000

# In a new terminal, start the UI
streamlit run app/ui.py
```

## Evaluation and Justification

The project includes built-in benchmarking tools to validate architectural choices:
- **Evaluation Framework**: Uses RAGAS to measure faithfulness, relevancy, and recall.
- **Chunking Analysis**: Empirical testing of different chunk sizes (256, 500, 1000) to find the optimal retrieval balance.
- **Critic Calibration**: Quantified comparison of system performance with and without the self-correction loop.