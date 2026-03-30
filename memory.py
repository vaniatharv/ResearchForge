import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json

load_dotenv()

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize vector store
vector_store = Chroma(
    collection_name="research_data",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


def store_in_vectordb(content: str, metadata: dict = None):
    """
    Store content as embedding vectors in ChromaDB.
    Optimized for research documents - stores full content for better retrieval.
    
    Args:
        content: The content to store
        metadata: Additional metadata to attach to the document
    
    Returns:
        List of document IDs that were added
    """
    try:
        # For research documents, store as single unified document for better context
        if metadata and metadata.get("type") == "research_result":
            doc = Document(
                page_content=content,
                metadata=metadata or {}
            )
            documents = [doc]
        else:
            # For other content, parse JSON to extract structured data
            try:
                data = json.loads(content)
                documents = []
                
                if isinstance(data, dict):
                    # Create documents for each section
                    for key, value in data.items():
                        doc_metadata = metadata.copy() if metadata else {}
                        doc_metadata["section"] = key
                        
                        doc = Document(
                            page_content=str(value),
                            metadata=doc_metadata
                        )
                        documents.append(doc)
                else:
                    # If not a dict, store as single document
                    doc = Document(
                        page_content=content,
                        metadata=metadata or {}
                    )
                    documents.append(doc)
                    
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, store as plain text
                doc = Document(
                    page_content=content,
                    metadata=metadata or {}
                )
                documents = [doc]
        
        # Add documents to vector store
        ids = vector_store.add_documents(documents)
        print(f"✓ Stored {len(documents)} document(s) in vector database")
        return ids
        
    except Exception as e:
        print(f"Error: Failed to store in vector database: {e}")
        return []


def retrieve_from_vectordb(query: str, k: int = 5, filter_metadata: dict = None):
    """
    Retrieve top 5 most similar documents from vector store based on query.
    
    Args:
        query: Search query
        k: Number of results to return (default: 5, max: 5)
        filter_metadata: Optional metadata filter
    
    Returns:
        List of up to 5 most relevant documents
    """
    try:
        # Ensure we don't exceed 5 documents
        k = min(k, 5)
        
        if filter_metadata:
            results = vector_store.similarity_search(
                query, 
                k=k,
                filter=filter_metadata
            )
        else:
            results = vector_store.similarity_search(query, k=k)
        
        print(f"✓ Retrieved {len(results)} relevant document(s)")
        return results
    except Exception as e:
        print(f"Error: Failed to retrieve from vector database: {e}")
        return []


def retrieve_with_scores(query: str, k: int = 5, score_threshold: float = None):
    """
    Retrieve top 5 most similar documents with similarity scores.
    
    Args:
        query: Search query
        k: Number of results to return (default: 5, max: 5)
        score_threshold: Optional minimum similarity score (lower is better, typically 0.0-2.0)
    
    Returns:
        List of up to 5 tuples (document, score) sorted by relevance
    """
    try:
        # Ensure we don't exceed 5 documents
        k = min(k, 5)
        
        results = vector_store.similarity_search_with_score(query, k=k)
        
        # Filter by score threshold if provided (lower score = more similar)
        if score_threshold is not None:
            results = [(doc, score) for doc, score in results if score <= score_threshold]
        
        print(f"✓ Retrieved {len(results)} document(s) with scores")
        return results
    except Exception as e:
        print(f"Error: Failed to retrieve from vector database: {e}")
        return []


def delete_collection():
    """Delete the entire collection from vector store."""
    try:
        vector_store.delete_collection()
        print("✓ Deleted vector store collection")
    except Exception as e:
        print(f"Error: Failed to delete collection: {e}")


