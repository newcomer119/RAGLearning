import streamlit as st
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# Load .env from project root (parent of this file's dir)
load_dotenv()
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(_env_path)
nvidia_api_key = os.getenv("NVIDIA_VIM_API_KEY") or os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key=nvidia_api_key)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings(nvidia_api_key=nvidia_api_key)
        pdf_dir = "pdfs"
        if not os.path.isdir(pdf_dir):
            os.makedirs(pdf_dir, exist_ok=True)
            st.error(f"Created empty folder: '{pdf_dir}'. Add PDF files there and click 'Document Embedding' again.")
            return
        st.session_state.loader = PyPDFDirectoryLoader(pdf_dir)
        st.session_state.docs = st.session_state.loader.load()
        if not st.session_state.docs:
            st.error(f"No PDFs found in '{pdf_dir}' or no text could be extracted. Add PDF files and try again.")
            return
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.final_documents = [d for d in st.session_state.final_documents if (d.page_content or "").strip()]
        if not st.session_state.final_documents:
            st.error("No text chunks from PDFs. Check that the PDFs contain extractable text.")
            return
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("NVIDIA NIM DEMO")

prompt = ChatPromptTemplate.from_template(""" 
Answer the questions based on the provided context only.
Please provide the most accurate response
<context>
{context}
</context>
Questions: {input}
""")


prompt1 = st.text_input("Enter Your Question from documents")

if st.button("Document Embedding"):
    vector_embedding()
    if "vectors" in st.session_state:
        st.success("FAISS Vector Store DB is Ready Using NvidiaEmbeddings")


if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' first to build the vector store.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt=prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response Time : ", time.process_time() - start)
        st.write(response['answer'])

        ## With a Streamlit expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("--------------------------------")