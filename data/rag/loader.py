import os
from langchain_community.document_loaders import PyPDFLoader

def load_documents():

    documents = []

    folder_path = "data"

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(pdf_path)

            documents.extend(loader.load())

    return documents
