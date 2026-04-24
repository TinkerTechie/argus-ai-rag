import os
import time
from langchain_groq import ChatGroq

class SafeChatGroq(ChatGroq):
    def invoke(self, *args, **kwargs):
        for i in range(5):
            try:
                return super().invoke(*args, **kwargs)
            except Exception as e:
                if '429' in str(e) or 'rate_limit' in str(e).lower() or 'Rate limit' in str(e):
                    print(f"[Rate Limit Hit] Sleeping for {2 ** i} seconds before retry...")
                    time.sleep(2 ** i)
                else:
                    raise e
        return super().invoke(*args, **kwargs)

# Using Groq API for high-speed inference on Streamlit Cloud
llm = SafeChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_retries=5,
    api_key=os.environ.get("GROQ_API_KEY")
)