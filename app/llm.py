import os
from langchain_groq import ChatGroq

# Using Groq API for high-speed inference on Streamlit Cloud
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.environ.get("GROQ_API_KEY")
)