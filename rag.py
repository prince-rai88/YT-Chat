import os
from typing import Tuple, List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def create_vector_store(text: str, chunk_size: int, chunk_overlap: int) -> Tuple[FAISS, List[str]]:
    """Chunks the text, creates embeddings, and builds a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = MistralAIEmbeddings(mistral_api_key=os.getenv("MISTRAL_API_KEY"))
    
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store, chunks

def get_rag_chain(vector_store: FAISS, top_k: int, temperature: float) -> Tuple[Any, Any]:
    """Constructs the RAG chain for answering questions."""
    
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=temperature, 
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    template = """You are a helpful assistant that answers questions based on the provided YouTube video transcript.
Answer ONLY from the context provided.
Never hallucinate or make up information.
If the answer isn't present in the context, say exactly: "I don't know."
Keep answers concise and clear.

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever