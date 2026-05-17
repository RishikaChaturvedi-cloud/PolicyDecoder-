from langchain_community.vectorstores import Chroma

def create_vector_db(docs, embeddings):

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="db"
    )

    return db
