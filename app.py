import streamlit as st
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# LangChain imports (correct)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain.chains.question_answering import load_qa_chain


# ---------------- UI ----------------
st.set_page_config(page_title="Law & Policy Decoder")

st.title("🇮🇳 Law & Policy Decoder")
st.write("Ask questions about Indian government schemes")


# ---------------- LOAD PDF ----------------
loader = PyPDFLoader("data/pm_kisan.pdf")
documents = loader.load()


# ---------------- SPLIT TEXT ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)


# ---------------- EMBEDDINGS ----------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)


# ---------------- VECTOR DB ----------------
db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=api_key
)

chain = load_qa_chain(llm, chain_type="stuff")


# ---------------- INPUT ----------------
question = st.text_input("Enter your question")

if question:

    with st.spinner("Thinking..."):

        relevant_docs = retriever.get_relevant_documents(question)

        prompt = f"""
        You are a helpful AI assistant for Indian citizens.

        Answer in simple language.

        Use only the provided government scheme documents.

        Question:
        {question}
        """

        response = chain.run(
            input_documents=relevant_docs,
            question=prompt
        )

    st.subheader("Answer")
    st.write(response)

    st.subheader("Sources")

    for doc in relevant_docs:
        st.write(doc.page_content[:500])
        st.write("------")
