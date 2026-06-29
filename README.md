# 🎥 YouTube RAG Chat Web App

A modern, high-performance Retrieval-Augmented Generation (RAG) web application that brings conversational AI to YouTube videos. Extract knowledge quickly and interactively by chatting directly with the content of any YouTube video. 

This application seamlessly extracts video transcripts (including multi-lingual support), splits them into context-aware chunks, generates vector embeddings, and leverages a Large Language Model to answer your questions based *strictly* on the video's context to prevent hallucinations.

---

## ✨ Key Features

- **Universal URL Support**: Seamlessly processes standard links, shortened links (`youtu.be`), and YouTube Shorts.
- **Multi-Lingual Capabilities**: Automatically detects and processes video transcripts regardless of the origin language.
- **Dynamic RAG Pipeline**: Adjust chunk size, chunk overlap, top-K retrieval mechanisms, and LLM temperature via an intuitive sidebar.
- **Transparent Sourcing**: Toggle the ability to view the exact retrieved transcript chunks used to generate any given answer.
- **State-of-the-Art AI**: Powered by Mistral AI's embedding and highly capable conversational models.
- **Optimized UI Experience**: Built entirely in Streamlit with modern layout aesthetics, active metrics display, and dynamic chat elements.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Frontend Framework** | [Streamlit](https://streamlit.io/) | Interactive web UI and application orchestration |
| **Pipeline Setup** | [LangChain](https://python.langchain.com/) | Powerful framework for LLM-based applications |
| **Vector Database** | [FAISS](https://github.com/facebookresearch/faiss) | Lightning-fast similarity search via Facebook AI |
| **Embeddings & LLM** | [Mistral AI](https://mistral.ai/) | Semantic processing and core conversational logic |
| **Transcript API** | `youtube-transcript-api` | Pythonic, headless video subtitle extraction |

---

## 📐 System Architecture

1. **Input & Extraction**: The user provides a YouTube URL. The system validates the ID, fetches rich metadata (thumbnails, channel information) via the YouTube oEmbed API, and downloads the raw transcript (falling back dynamically on available languages).
2. **Text Processing**: The transcript is parsed into smaller overlapping text segments using `RecursiveCharacterTextSplitter`.
3. **Embedding Generation**: Text chunks are passed to `MistralAIEmbeddings` to generate robust vector representations.
4. **Vector Storage**: Resulting vectors are indexed into an in-memory `FAISS` vector database for high-speed retrieval.
5. **Similarity Search**: User questions are embedded and matched against the FAISS store to discover the Top-K most relevant chunks.
6. **Augmented Generation**: The retrieved context is formatted into a strict prompt template and fed to `ChatMistralAI` to guarantee grounded, accurate, and non-hallucinated answers.

---

## ⚙️ How To Run Locally

Follow these instructions to get development up and running on your local machine.

### 1. Clone the repository
```bash
git clone https://github.com/prince-rai88/YT-Chat.git
cd YT-Chat
```

### 2. Set up a Python Virtual Environment
Keep your dependencies isolated by creating a virtual environment:
```bash
python3 -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the provided environment template to a production `.env` file:
```bash
cp .env.example .env
```
Open `.env` and add your Mistral API key (get one from the [Mistral AI Platform](https://console.mistral.ai/)):
```env
MISTRAL_API_KEY=your_actual_api_key_here
```

### 5. Launch the Application
Start the Streamlit development server:
```bash
streamlit run app.py
```
The web app will automatically launch in your default browser at `http://localhost:8501`.

---

## 🚧 Roadmap & Future Directions

- **Persistent Indexing**: Implement disk-based caching (saving indexes in `faiss_cache/`) to avoid reprocessing previously queried videos.
- **Extended Scraping**: Broaden transcript extraction to gracefully handle auto-generated captions when a native subtitle track isn't deployed.
- **Provider Agnostic**: Introduce additional LLM providers (e.g., OpenAI, Anthropic, Google Gemini) accessible via a user-facing dropdown.
- **Channel-Level RAG**: Support indexing multiple videos to answer questions holistically across an entire creator's channel.
