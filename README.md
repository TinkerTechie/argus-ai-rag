# 🧠 Argus – Multi-Agent RAG Research Assistant

Argus is a **stateful multi-agent Retrieval-Augmented Generation (RAG) system** built using LangGraph, FAISS, and a local LLM (Llama3 via Ollama). It simulates a real-world AI research assistant by combining retrieval, reasoning, and self-evaluation in a structured pipeline.

---

## 🚀 Features

- 🔍 Semantic Retrieval using FAISS + SentenceTransformers  
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
Retriever (FAISS)
↓
Analyst (LLM - Llama3)
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
- FAISS – Vector database  
- SentenceTransformers – Embeddings  
- Ollama (Llama3) – Local LLM  
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

## ⚙️ Setup Instructions

### 1️⃣ Clone repo
```bash
git clone https://github.com/your-username/argus-ai-rag.git
cd argus-ai