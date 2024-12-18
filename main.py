import uuid

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import streamlit as st

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

persistent_db_path = "cybercrimeDB"

db_client = Chroma(
    persist_directory=persistent_db_path,
    embedding_function=embeddings_model,
    collection_name="cyberbullying_docs"
)


# %% LLM Setup
# available models: https://console.groq.com/docs/models
# MODEL = "llama-3.1-8b-instant"
# llm = ChatOpenAI(model=MODEL, temperature=0)


MODEL_NAME = 'gpt-3.5-turbo'
llm = ChatOpenAI(model_name=MODEL_NAME,
                   temperature=0.5, # controls creativity
                   api_key=os.getenv('OPENAI_API_KEY'))


# %% RAG Chat with style
def rag_chat_add_style_language(query: str, k: int = 5, style: str = "informal", language: str = "english"):
    retrieved_docs = db_client.similarity_search(query=query, k=k)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    retrieved_docs_text_str = "\n".join(retrieved_docs_text)

    query_and_context = (
        "These docs can help you with your questions. If you have no answer, simply say 'I have no answer'."
        f"Question: {query}\n"
        f"Relevant docs: {retrieved_docs_text_str}"
    )

    messages = [
        ("system", f"You are an expert assistant providing information strictly based on the context provided to you. Your task is to answer questions or provide information only using the details given in the current context. Do not reference any external knowledge or information not explicitly mentioned in the context. If the context does not contain sufficient information to answer a question, clearly state that the information is not available in the provided context. Always keep your answers to not more than 100 words. You should answer in a {style} style and in {language} language."),
        ("human", query_and_context)
    ]
    
    res = llm.invoke(messages)
    return res.content


class MyChatMessage:
    def __init__(self, text: str, role: str = "user"):
        self.message = text
        self.role = role


def main():
    st.session_state.setdefault(
        "messages", []
    )

    st.title("Cyberbullying 101 Q&A")

    for chat in st.session_state.messages:
        with st.chat_message(chat.role):
            st.markdown(chat.message)

    if prompt := st.chat_input("Ask a question"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append(
            MyChatMessage(prompt)
        )
        response = rag_chat_add_style_language(prompt, k=3, language="english")

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append(
            MyChatMessage(response, role="assistant")
        )


if __name__ == "__main__":
    main()
