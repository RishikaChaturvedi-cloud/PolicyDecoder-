import streamlit as st
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.getenv("GOOGLE_API_KEY")

# ---- LangChain Imports (FULLY UPDATED FOR 2026/0.2+) ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Naye chains ke liye sahi imports
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---------------- UI ----------------
st.set_page_config(page_title="Law & Policy Decoder")
st.title("🇮🇳 Law & Policy Decoder")
st.write("Ask questions about Indian government schemes")

# ---------------- LOAD PDF ----------------
loader = PyPDFLoader("data/pm_kisan.pdf")
documents = loader.load()

# ---------------- SPLIT TEXT ----------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# ---------------- EMBEDDINGS ----------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# ---------------- VECTOR DB ----------------
db = Chroma.from_documents(docs, embeddings, persist_directory="db")
retriever = db.as_retriever(search_kwargs={"k": 3})

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=api_key
)

# ---------------- NEW CHAIN SETUP ----------------
# Naye LangChain mein prompt aur chain aise banti hai
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant helping Indian citizens understand government schemes.
Answer in simple and clear language.
Use ONLY the given context to answer the question. If you don't know, just say you don't know.

Context:
{context}

Question:
{input}
""")

# Yeh purane load_qa_chain ka naya aur sahi replacement hai
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# ---------------- USER INPUT ----------------
question = st.text_input("Enter your question")

if question:
    with st.spinner("Thinking..."):
        # Relevant documents fetch karein
        relevant_docs = retriever.invoke(question)
        
        # Chain ko run karein (Naya tareeqa)
        response = question_answer_chain.invoke({
            "context": relevant_docs,
            "input": question
        })

    st.subheader("Answer")
    # Naye chain ka output seedhe string hota hai
    st.write(response)

    st.subheader("Sources")
    for doc in relevant_docs:
        st.write(doc.page_content[:500])
        st.write("------")
