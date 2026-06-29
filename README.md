# YouTube RAG Chat Web App

An internship-level Retrieval-Augmented Generation (RAG) web application that allows you to chat with any YouTube video. The application extracts the transcript of the video, splits it into chunks, creates embeddings, and uses a Large Language Model to answer your questions based *only* on the video's context.

## 🚀 Features

- **Any YouTube Video:** Supports various YouTube URL formats (standard, shortened, shorts).
- **RAG Architecture:** Leverages LangChain for setting up a robust semantic search pipeline.
- **Adjustable Settings:** Configure chunk size, chunk overlap, top-K retrieval, and LLM temperature dynamically via the sidebar.
- **Transparency:** Option to view the exact retrieved transcript chunks used to generate the answer.
- **Mistral AI:** Uses Mistral's state-of-the-art embedding and chat models.
- **Clean UI:** Built entirely in Streamlit with modern layout, chat elements, and metrics display.

## 🛠️ Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://python.langchain.com/)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (cpu)
- **Embeddings & LLM:** [Mistral AI](https://mistral.ai/)
- **Transcript Extraction:** `youtube-transcript-api`
- **Language:** Python 3.9+

## 📐 Architecture

1. **Input:** User provides a YouTube Video URL.
2. **Extraction:** Transcript is downloaded via API, and video metadata is fetched via oEmbed.
3. **Processing:** Transcript is split into smaller overlapping segments (RecursiveCharacterTextSplitter).
4. **Embedding:** Text chunks are passed to `MistralAIEmbeddings` to generate vectors.
5. **Storage:** Vectors are indexed in an in-memory `FAISS` vector database.
6. **Query:** User asks a question, which is embedded and matched against the FAISS store (Similarity Search).
7. **Generation:** Top-K formatted chunks are provided to `ChatMistralAI` along with a strict prompt template to generate an accurate, grounded answer.

## 📁 Folder Structure

```
YT-Chat/
│
├── app.py               # Main Streamlit application and UI
├── rag.py               # LangChain logic, Vector Store, LLM initialization
├── utils.py             # Helper methods (Transcript fetching, ID extraction, metadata)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── .env.example         # Template for environment variables
├── .gitignore           # Git ignore rules
└── assets/              # Demo assets (screenshots, gifs)
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd YT-Chat
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   - Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Get a Mistral API key from [Mistral AI Platform](https://console.mistral.ai/).
   - Add your Mistral API key to the `.env` file:
     ```env
     MISTRAL_API_KEY=your_actual_api_key_here
     ```

## 🚀 Running the Project

Run the Streamlit application using the following command:

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

## 📸 Screenshots & Demo

*(Add your screenshot `assets/screenshot.png` and demo `assets/demo.gif` here)*

## 🚧 Future Improvements

- Implementing persistent FAISS storage and disk-based caching (saving indexes in `faiss_cache/`).
- Expanding transcript scraping capabilities for videos with auto-generated captions or disabled native transcripts.
- Adding additional LLM providers (OpenAI, Anthropic, Gemini) via an interactive dropdown.
- Supporting multi-video RAG across an entire YouTube channel.
