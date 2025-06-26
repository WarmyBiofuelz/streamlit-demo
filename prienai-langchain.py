from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from langchain import hub

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, TextLoader
import bs4  # BeautifulSoup for parsing HTML

load_dotenv()  # take environment variables

# from .env file
# Load environment variables from .env file

token = os.getenv("SECRET2")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# Load Prienai Wikipedia page
wiki_loader = WebBaseLoader(
    web_paths=("https://lt.wikipedia.org/wiki/Prienai",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("mw-parser-output", "mw-content-ltr")
        )
    ),
)
wiki_docs = wiki_loader.load()

# Load local Prienai text file
try:
    local_loader = TextLoader("prienai.txt", encoding="utf-8")
    local_docs = local_loader.load()
    st.success("✅ Sėkmingai įkeltas prienai.txt failas")
except FileNotFoundError:
    st.warning("⚠️ prienai.txt failas nerastas. Naudojama tik Vikipedijos informacija.")
    local_docs = []

# Combine documents from both sources
all_docs = wiki_docs + local_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_splitter.split_documents(all_docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token, # type: ignore
))

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

st.title("Prienų Informacijos Botas")
st.markdown("Užduokite klausimą apie Prienus ir gaukite atsakymą iš Vikipedijos ir vietinės informacijos!")

# Show data sources
st.sidebar.markdown("### Informacijos šaltiniai:")
st.sidebar.markdown("✅ Vikipedija (lt.wikipedia.org)")
if local_docs:
    st.sidebar.markdown("✅ Lokalus failas (prienai.txt)")
else:
    st.sidebar.markdown("❌ Lokalus failas (prienai.txt) - nerastas")

def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    
    
    st.info(rag_chain.invoke(input_text))

with st.form("my_form"):
    text = st.text_area(
        "Užduokite klausimą apie Prienus:",
        "Kokie yra Prienų istoriniai faktai?",
    )
    submitted = st.form_submit_button("Siųsti")
    if submitted:
        generate_response(text)

# Add some example questions
st.markdown("### Pavyzdiniai klausimai:")
example_questions = [
    "Kokie yra Prienų istoriniai faktai?",
    "Kiek gyventojų gyvena Prienuose?",
    "Kokie yra Prienų geografiniai duomenys?",
    "Kokios pramonės šakos veikia Prienuose?",
    "Kokie yra Prienų švietimo įstaigos?",
    "Kokie sporto klubai veikia Prienuose?"
]

for question in example_questions:
    if st.button(question, key=question):
        generate_response(question)
