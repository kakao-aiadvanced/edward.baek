from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Any

def load_and_create_vector_store() -> Chroma:
    """Load documents from URLs and create a vector store."""
    
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    
    # Load documents
    docs = WebBaseLoader(urls).load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag_chroma",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="./chroma_db"
    )
    
    return vectorstore

if __name__ == "__main__":
    # Test the document loading
    vectorstore = load_and_create_vector_store()
    print(f"Vector store created with {vectorstore._collection.count()} documents")