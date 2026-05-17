import streamlit as st
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.getenv("GOOGLE_API_KEY")

# ---- LangChain Imports (Stable 0.2.x Compatible) ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# ---------------- UI ----------------
st.set_page_config(page_title="Law & Policy Decoder")
st.title("🇮🇳 Law & Policy Decoder")
st.write("Ask questions about Indian government schemes")

# ---------------- LOAD PDF ----------------
# Make sure your GitHub repository has 'data/pm_kisan.pdf'
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

chain = load_qa_chain(llm, chain_type="stuff")

# ---------------- USER INPUT ----------------
question = st.text_input("Enter your question")

if question:
    with st.spinner("Thinking..."):
        # invoke use kar rahe hain jo stable hai
        relevant_docs = retriever.invoke(question)

        prompt = f"""
        You are an AI assistant helping Indian citizens understand government schemes.
        Answer in simple and clear language.
        Use ONLY the given documents.

        Question:
        {question}
        """

        # Stable invoke method for QA Chain
        response = chain.invoke({"input_documents": relevant_docs, "question": prompt})

    st.subheader("Answer")
    st.write(response["output_text"])

    st.subheader("Sources")
    for doc in relevant_docs:
        st.write(doc.page_content[:500])
        st.write("------")
