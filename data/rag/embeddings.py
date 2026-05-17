from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def get_embeddings():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    return embeddings
