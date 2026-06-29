import streamlit as st

# -------------------- Page Config -------------------- #
st.set_page_config(
    page_title="YouTube RAG Chat",
    page_icon="🎥",
    layout="wide"
)

# -------------------- Sidebar -------------------- #
with st.sidebar:
    st.title("⚙ Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=1500,
        value=1000,
        step=100
    )

    top_k = st.slider(
        "Top K Chunks",
        min_value=1,
        max_value=10,
        value=4
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

    st.divider()

    show_chunks = st.checkbox(
        "Show Retrieved Chunks",
        value=False
    )

# -------------------- Main Page -------------------- #

st.title("🎥 Chat with YouTube Videos")

st.markdown(
    """
Ask questions about any YouTube video using **Retrieval-Augmented Generation (RAG)**.
"""
)

st.divider()

video_url = st.text_input(
    "Paste a YouTube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

process = st.button(
    "🚀 Process Video",
    use_container_width=True
)

st.divider()

st.subheader("💬 Chat")

st.info(
    "Process a YouTube video to begin chatting."
)