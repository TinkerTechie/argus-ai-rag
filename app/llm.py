from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="phi3",
    temperature=0.2
)