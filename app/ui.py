import streamlit as st
import requests
import time

# ---------------- SETTINGS ----------------
st.set_page_config(
    page_title="Argus AI | Premium RAG",
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
    st.title("Argus AI")
    st.caption("v2.0.0 | Production Grade RAG")
    
    st.markdown("---")
    
    with st.container():
        st.subheader("🚀 System Status")
        st.markdown(f"**Model:** `Phi3` (Local)")
        st.markdown(f"**Mode:** `Advanced Multi-Agent`")
        st.status("System Active", state="complete")

    st.markdown("---")
    
    st.subheader("⚙️ Configuration")
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.2)
    top_k = st.slider("Retrieval Depth (Top-K)", 1, 10, 5)
    
    st.markdown("---")
    st.info("Argus uses LangGraph for iterative reasoning and self-criticism to ensure factual accuracy.")

# ---------------- MAIN UI ----------------
st.markdown('<h1 class="main-header">Argus AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent local research assistant powered by Phi3 & LangGraph</p>', unsafe_allow_html=True)

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

    # Call Backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing data and generating response..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"query": prompt},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received.")
                    score = data.get("score", 0.0)
                    feedback = data.get("feedback", "No feedback available.")
                    
                    # Simulate streaming for premium feel
                    for chunk in answer.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)
                    
                    message_placeholder.markdown(full_response)
                    
                    # Score and Feedback (rendered after streaming)
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
                    
                else:
                    st.error(f"Error: Backend returned {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection failed: {e}")