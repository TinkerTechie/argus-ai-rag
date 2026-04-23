# 🧠 Argus: Multi-Agent Verified RAG Research Assistant

Argus is a **stateful multi-agent Retrieval-Augmented Generation (RAG) system** built using LangGraph, Qdrant, and a local LLM (Phi3 via Ollama). It simulates a real-world AI research assistant by combining retrieval, reasoning, and self-evaluation in a structured pipeline.

---

## 🚀 Features

- 🔍 Semantic Retrieval using Qdrant + SentenceTransformers  
- 🧠 Multi-Agent Workflow (LangGraph)
  - Router (query classification)
  - Retriever (document search)
  - Analyst (LLM-based reasoning)
  - Critic (LLM-based evaluation)
- 🔁 Self-Correction Loop to reduce hallucinations  
- 🏠 Fully Local LLM using Ollama (no API cost)  
- ⚡ Modular and scalable architecture  

---

## 🧠 System Architecture


User Query
↓
Router
↓
Retriever (Qdrant)
↓
Analyst (LLM - Phi3)
↓
Critic (LLM Evaluation)
↓
Retry Loop (if needed)
↓
Final Answer


---

## 📦 Tech Stack

- LangGraph – Workflow orchestration  
- LangChain – LLM & retrieval integration  
- Qdrant – Vector database  
- SentenceTransformers – Embeddings  
- Ollama (Phi3) – Local LLM  
- Python  

---

## 📂 Project Structure


argus-ai/
│
├── app/
│ ├── graph/ # LangGraph workflow & nodes
│ ├── rag/ # ingestion & vectorstore
│ ├── state/ # graph state schema
│ ├── llm.py # LLM configuration
│ └── main.py # entry point
│
├── data/
│ └── raw/ # PDF documents
│
├── notebooks/
├── requirements.txt
└── README.md


---

## ⚡ Concurrency & Performance

Argus is designed for efficient local execution:
- **Async FastAPI**: All endpoints are asynchronous to prevent blocking the main event loop.
- **Thread Offloading**: Long-running graph executions (LLM calls) are offloaded to separate threads using `asyncio.to_thread` to ensure the API remains responsive.
- **Current Scaling Limitations**: While the API layer is async, local LLM inference is sequential on single-GPU/CPU setups. For production scaling, we recommend moving LLM inference to a dedicated cluster (e.g., vLLM or TGI).

---

### 1️⃣ Clone repo
```bash
git clone https://github.com/your-username/argus-ai-rag.git
cd argus-ai