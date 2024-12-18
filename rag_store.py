from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

doc_file_path = 'data/cyberbullying.pdf'


def process_data(file_path: str, title: str) -> list[Document]:
    # load text
    loader = PyPDFLoader(file_path)
    data = loader.load()
    # split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(data)
    # add metadata book_title
    for chunk in chunks:
        # print(chunk.page_content)
        chunk.metadata["title"] = title
    return chunks


# Embeddings Model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# %% connect to the database
persistent_db_path = "cybercrimeDB"
db_client = Chroma(
    collection_name="cyberbullying_docs",
    persist_directory=persistent_db_path,
    embedding_function=embeddings_model
)

data_chunks = process_data(doc_file_path, "Cyber Bullying")

# add the documents to the database
db_client.add_documents(documents=data_chunks)

len(db_client.get()['ids'])
