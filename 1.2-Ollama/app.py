import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

## LANGSMITH TRACKING

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")


## Prompt Template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the question asked "),
        ("user","Question:{question}")
    ]
)

## STREAM LIT FRAMEWORK
st.title("LANGCHAIN Demo With LLAMA2")
input_text = st.text_input("What Question do you have in your mind")

## Ollama LLama model
llm = Ollama(model="llama3")
output_parser =StrOutputParser()
chain = prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({"question":input_text}))