from langchain_ollama import ChatOllama

# Using local Ollama inference (gemma3:1b) — no internet required
llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.2
)