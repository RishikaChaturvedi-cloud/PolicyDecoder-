import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

# Load API key
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Law & Policy Decoder")

st.title("🇮🇳 Law & Policy Decoder")
st.write("Ask questions about Indian government schemes")

# Load PDF
loader = PyPDFLoader("data/pm_kisan.pdf")
documents = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Create vector database
db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

chain = load_qa_chain(llm, chain_type="stuff")

# User question
question = st.text_input("Enter your question")

if question:

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
