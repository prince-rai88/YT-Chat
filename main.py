from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from dotenv import load_dotenv
load_dotenv()

video_id = "LPZh9BOjkQs"
# video_id = "J5_-l7WIO_w"  # Hindi

try:
    # STEP 1: Fetch the transcript from YouTube
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=["en"])

    transcript = " ".join(chunk.text for chunk in transcript_list)

    # STEP 2: Split the transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.create_documents([transcript])

    # STEP 3: Create embeddings
    embeddings = MistralAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # STEP 4: Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # STEP 5: Create LLM
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2,
        max_tokens=512
    )

    # STEP 6: Prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer ONLY from the provided context. "
            "If the answer is not present in the context, say 'I don't know.'"
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}"
        )
    ])

    # STEP 7: Chat loop
    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        # Retrieve relevant documents
        docs = retriever.invoke(question)

        # Combine retrieved chunks
        context = "\n\n".join(doc.page_content for doc in docs)

        # Create chain
        chain = prompt | llm

        # Get response
        response = chain.invoke({
            "context": context,
            "question": question
        })

        print("\nAnswer:")
        print(response.content)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")

except Exception as e:
    print(f"Error: {e}")