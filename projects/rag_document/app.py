import streamlit as st
import os
import time
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Load .env from project root (parent of this file) or current directory
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)
load_dotenv()

## LOAD THE API KEYS
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

st.title("RAG Document Q&A With Groq and Llama 3")

# Initialize the LLM - Corrected model name for Groq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        # 1. Initialize Embeddings
        st.session_state.embeddings = OpenAIEmbeddings()

        # 2. Data Ingestion - use folder next to this script
        papers_dir = Path(__file__).resolve().parent / "research_papers"
        st.session_state.loader = PyPDFDirectoryLoader(str(papers_dir))

        # 3. Document Loading
        st.session_state.docs = st.session_state.loader.load()
        if not st.session_state.docs:
            st.error("No PDFs found. Add PDF files in the 'research_papers' folder and try again.")
            return

        # 4. Text Splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # 5. Creating Final Documents (limit to first 50 docs for performance)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        if not st.session_state.final_documents:
            st.error("No text could be extracted from the PDFs.")
            return

        # 6. Vector Embeddings
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

user_prompt = st.text_input("Enter Your Query from the research papers")

if st.button("Document Embedding"):
    with st.spinner("Loading PDFs and building vector store..."):
        create_vector_embedding()
    if "vectors" in st.session_state:
        st.success("Vector database created successfully!")

# Logic to handle the query
if user_prompt:
    # Important: Check if vectors have been initialized first
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.perf_counter()
        response = retrieval_chain.invoke({"input": user_prompt})
        response_time = time.perf_counter() - start
        
        st.write(f"Response Time: {response_time:.2f} seconds")
        st.write(response['answer'])

        # With a streamlit expander for transparency
        with st.expander("Document Similarity Search (Context Used)"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('--------------------------------')
    else:
        st.error("Please click 'Document Embedding' first to initialize the database.")