# RAG Q&A Conversational Chatbot
import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


# SETTING UP STREAMLIT

st.title("Conversational RAG With PDF uploads with chat history")
st.write("Upload PDF's and chat with their constants!")

# INPUT THE GROQ API KEY
api_key = st.text_input("Enter Your Groq Api KEY : ", type="password")


# Check if the api key is there

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # CHAT SESSION
    session_id = st.text_input("Session ID", value="default_session")

    # state fully manage chat history
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Choose a PDF file ", type="pdf", accept_multiple_files=False)

    # Process uploaded files (single file when accept_multiple_files=False)
    if uploaded_files:
        documents = []
        temppdf = "./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_files.getvalue())
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

        # Split and create Embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in chat history"
            "formulate a standalone question which can be understood"
            "without the chat history. Do Not Answer the question,"
            "just reformulate it if needed and otherwise return it as is"
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt)

        # ANSWER THE QUESTIOn

        system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer "
            "the question. If you dont know the answer, say that you "
            "dont know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        user_input = st.text_input("Your Questions : ")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            st.write("Assistant : ", response["answer"])
            st.write("Chat History : ", session_history.messages)
    else:
        st.warning("Please Enter GROQ API KEY")