import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader #as we are loading different files so we didnt used any dict loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_db" 
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") 

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_directory_documents(folder_path: str):
    documents = []

    for root, _, files in os.walk(folder_path):  #root is folder that has files example: ragdocs (we used 2 loops for iterating in any subfolder then root=ragdocs/invoice  )
        for file in files:
            full_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            ext = ext.lower()

            if ext == ".csv":
                loader = CSVLoader(file_path=full_path)
                
            elif ext == ".pdf":
                loader = PyPDFLoader(file_path=full_path)

            elif ext in [".txt", ".md"]:
                loader = TextLoader(file_path=full_path, encoding="utf-8", autodetect_encoding=True)

            else:
                continue
            try:
                loaded_docs=loader.load()

                for doc in loaded_docs:
                        doc.metadata["file_name"] = file

                documents.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,  
        add_start_index=True
    )
    return splitter.split_documents(documents)

def build_vectorstore(chunks):
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    return db

def ingest(folder_path: str):
    documents = load_directory_documents(folder_path)
    chunks = split_documents(documents)

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        # DB exists — load it and ADD new chunks on top
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        db.add_documents(chunks)  # adds without wiping
    else:
        # DB doesn't exist — build fresh
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=CHROMA_PATH
        )
    return db

def ingest_temporary(folder_path: str, existing_db=None):
    documents = load_directory_documents(folder_path)
    chunks = split_documents(documents)

    if existing_db:
        # copy default DB chunks into a NEW in-memory DB
        existing_data = existing_db.get()
        existing_texts = existing_data["documents"]
        existing_metadatas = existing_data["metadatas"]

        # create fresh in-memory DB with default docs
        temp_db = Chroma.from_texts(
            texts=existing_texts,
            embedding=embeddings,
            metadatas=existing_metadatas
            # no persist_directory = stays in memory
        )
        # now add uploaded chunks on top
        temp_db.add_documents(chunks)
        return temp_db
    else:
        db = Chroma.from_documents(chunks, embeddings)
        return db

if __name__ == "__main__":
    ingest(DATA_PATH)