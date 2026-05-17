from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.question_answering import load_qa_chain

load_dotenv()

def get_answer(question, db):

    retriever = db.as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.get_relevant_documents(question)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    prompt = f"""
    You are an AI assistant for Indian citizens.

    Answer the question in simple language.

    Use only the provided government scheme documents.

    Question:
    {question}
    """

    response = chain.run(
        input_documents=relevant_docs,
        question=prompt
    )

    return response, relevant_docs
