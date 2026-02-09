from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()
from langserve import add_routes


groq_api = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api)


## 1.Create Prompt Templates

system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

parser = StrOutputParser()

#Create chain
chain = prompt_template|model|parser

## APP definition

app = FastAPI(title="Langchain Server", version="1.0",description="A Simple AI server using Langchain runnable interfaces")


add_routes(
    app,
    chain,
    path="/chain"
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)