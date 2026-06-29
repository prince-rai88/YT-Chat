import streamlit as st
import os
from dotenv import load_dotenv

from utils import extract_video_id, fetch_youtube_metadata, get_transcript
from rag import create_vector_store, get_rag_chain

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chat", 
    page_icon="🎥", 
    layout="wide"
)

def main():
    st.title("🎥 YouTube RAG Chat Web App")
    st.markdown("Chat with any YouTube video transcript using Mistral AI and LangChain! Extract knowledge quickly and interactively.")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Chunking Settings")
        chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        
        st.subheader("Retrieval & LLM Settings")
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=4)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        st.subheader("Display Settings")
        show_chunks = st.checkbox("Show Retrieved Chunks", value=False)
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This internship-level project uses Retrieval-Augmented Generation (RAG) to let you converse with content from YouTube videos.")
        st.markdown("**Tech Stack:**\n- Streamlit\n- LangChain\n- FAISS\n- Mistral AI Models")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        
    if "video_meta" not in st.session_state:
        st.session_state.video_meta = None

    url_input = st.text_input("🔗 Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("▶️ Process Video"):
        if not url_input:
            st.warning("Please enter a YouTube video URL first.")
            return

        video_id = extract_video_id(url_input)
        if not video_id:
            st.error("Could not extract a valid Video ID from the provided URL.")
            return

        st.session_state.video_meta = fetch_youtube_metadata(video_id)
            
        with st.spinner("Fetching video transcript..."):
            transcript, error = get_transcript(video_id)
            
        if error:
            st.error(error)
            return
            
        with st.spinner("Splitting text..."):
            pass
            
        with st.spinner("Creating Mistral embeddings & building FAISS vector store..."):
            try:
                vs, chunks = create_vector_store(transcript, chunk_size, chunk_overlap)
                st.session_state.vector_store = vs
                st.session_state.transcript_length = len(transcript)
                st.session_state.num_chunks = len(chunks)
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                st.info("Make sure you have set the MISTRAL_API_KEY in your .env file.")
                return
            
        st.success("Video processed successfully! You can now start chatting.")
        
        st.session_state.messages = []

    # Display Video Metadata if available
    if st.session_state.video_meta:
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                thumb = st.session_state.video_meta.get("thumbnail_url")
                if thumb:
                    st.image(thumb, use_container_width=True)
            with col2:
                st.subheader(st.session_state.video_meta.get("title", "Unknown Title"))
                st.write(f"**Channel:** {st.session_state.video_meta.get('author_name', 'Unknown')}")
                
                if "transcript_length" in st.session_state:
                    col_met1, col_met2 = st.columns(2)
                    col_met1.metric("Transcript Length", f"{st.session_state.transcript_length} chars")
                    col_met2.metric("Total Chunks Created", st.session_state.num_chunks)
            st.markdown("---")

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the video..."):
        if st.session_state.vector_store is None:
            st.warning("Please process a video first before chatting.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag_chain, retriever = get_rag_chain(
                    st.session_state.vector_store, 
                    top_k, 
                    temperature
                )
                
                try:
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    if show_chunks:
                        docs = retriever.invoke(prompt)
                        with st.expander("Show Retrieved Chunks from Transcript"):
                            for i, doc in enumerate(docs):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.info(doc.page_content)
                except Exception as e:
                    st.error(f"Error during response generation: {e}")
                    
    # Footer
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px; color: grey; font-size: 0.8em;'>
        Built with Streamlit • LangChain • FAISS • Mistral AI • YouTube Transcript API
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()