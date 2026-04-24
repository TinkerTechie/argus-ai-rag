import streamlit as st
import requests
import time
import sys
import os

# Ensure the repository root is in sys.path so 'app' modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------- SETTINGS ----------------
st.set_page_config(
    page_title="Argus: Multi-Agent Verified RAG | Premium RAG",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
def local_css():
    st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #050505;
        color: #e0e0e0;
    }

    /* Gradient Header */
    .stApp header {
        background-color: transparent;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 3rem;
    }

    /* Chat Containers */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin-bottom: 1rem !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px);
    }

    /* Confidence Badge */
    .confidence-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
    }

    .confidence-label {
        font-size: 0.8rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Progress Bar Color */
    div[data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7);
    }

    /* Expander Styling */
    .stExpander {
        background-color: transparent !important;
        border: none !important;
        border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #050505;
    }
    ::-webkit-scrollbar-thumb {
        background: #222;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #333;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.title("Argus: Multi-Agent RAG")
    st.caption("v2.0.0 | Production Grade RAG")
    
    st.markdown("---")
    
    with st.container():
        st.subheader("🚀 System Status")
        st.markdown(f"**Model:** `Llama 3.1 (Groq)`")
        st.markdown(f"**Mode:** `Advanced Multi-Agent`")
        st.status("System Active", state="complete")

    st.markdown("---")
    
    st.subheader("⚙️ Configuration")
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.2)
    top_k = st.slider("Retrieval Depth (Top-K)", 1, 10, 5)
    
    st.markdown("---")
    
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs to build knowledge", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Process & Ingest", use_container_width=True):
        if uploaded_files:
            with st.spinner("Reading & ingesting documents..."):
                from app.ingest import ingest_file
                total_chunks = 0
                for f in uploaded_files:
                    total_chunks += ingest_file(f)
                st.success(f"Added {total_chunks} chunks to the database! You can now ask questions about them.")
        else:
            st.warning("Please upload a file first.")
            
    st.markdown("---")
    st.info("Argus uses LangGraph for iterative reasoning and self-criticism to ensure factual accuracy.")

# ---------------- MAIN UI ----------------
st.markdown('<h1 class="main-header">Argus: Multi-Agent Verified RAG</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced research assistant powered by Groq & LangGraph</p>', unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            # Confidence Score
            score = message.get("score", 0.0)
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f'<span class="confidence-label">Confidence</span>', unsafe_allow_html=True)
            with col2:
                st.progress(score)
                st.caption(f"{int(score*100)}% reliability score")
            
            # Critic Feedback
            if "feedback" in message:
                with st.expander("🛠️ Internal Reasoning & Critic Feedback"):
                    st.markdown(message["feedback"])

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Ask Argus anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Backend / Graph Directly
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing data and generating response..."):
            try:
                # Direct graph invocation instead of requests.post
                from app.graph.workflow import build_graph
                
                if "graph_app" not in st.session_state:
                    st.session_state.graph_app = build_graph()
                
                result = st.session_state.graph_app.invoke({
                    "query": prompt,
                    "revision_count": 0
                })
                
                answer = result.get("draft_answer", "No answer generated.")
                score = result.get("critique_score", 0.0)
                feedback = result.get("critique_feedback", "No feedback available.")
                
                # Simulate streaming for premium feel
                import re
                # Split while preserving spaces and newlines
                tokens = re.split(r'(\s+)', answer)
                for token in tokens:
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
                
                message_placeholder.markdown(full_response)
                
                # Score and Feedback
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f'<span class="confidence-label">Confidence</span>', unsafe_allow_html=True)
                with col2:
                    st.progress(score)
                    st.caption(f"{int(score*100)}% reliability score")
                
                with st.expander("🛠️ Internal Reasoning & Critic Feedback"):
                    st.markdown(feedback)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "score": score,
                    "feedback": feedback
                })
                    
            except Exception as e:
                st.error(f"Error processing query: {e}")