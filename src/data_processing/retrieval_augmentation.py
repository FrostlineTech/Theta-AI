"""
Retrieval Augmented Generation (RAG) System for Theta AI

This module enhances Theta's knowledge retrieval capabilities using
vector embeddings and semantic search.
"""

import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import random
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalAugmentation:
    """
    Implements Retrieval Augmented Generation for Theta AI.
    Creates vector embeddings for knowledge chunks and performs semantic search.
    """
    
    def __init__(self, datasets_dir: Path, vector_db_path: Optional[Path] = None):
        """
        Initialize the RAG system.
        
        Args:
            datasets_dir: Path to the datasets directory
            vector_db_path: Path to store vector database files (if None, uses datasets_dir/vector_db)
        """
        self.datasets_dir = datasets_dir
        
        # Set up vector db path
        if vector_db_path is None:
            self.vector_db_path = datasets_dir / "vector_db"
        else:
            self.vector_db_path = vector_db_path
            
        # Create vector db directory if it doesn't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Initialize in-memory vector store (in production, use a proper vector DB)
        self.document_store = []
        self.embedding_vectors = None
        
        # Track if the system has been initialized
        self.initialized = False
        
        try:
            # Try to import necessary dependencies
            # In a real system, these would be installed via pip
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Initialize vectorizer (placeholder for better embedding model)
            self.vectorizer = TfidfVectorizer(stop_words='english')
            
            logger.info("Successfully initialized RetrievalAugmentation system with TF-IDF")
        except ImportError as e:
            logger.warning(f"Could not initialize vectorizer: {e}")
            logger.warning("Will use fallback keyword matching instead of vector embeddings")
    
    def process_knowledge_base(self):
        """
        Process all available knowledge files and build the vector database.
        """
        logger.info("Processing knowledge base for RAG...")
        
        # Find all knowledge files
        knowledge_files = []
        
        # Standard QA datasets
        for json_file in self.datasets_dir.glob("*.json"):
            knowledge_files.append(json_file)
        
        # Curated QA datasets
        curated_qa_dir = self.datasets_dir / "curated_qa"
        if curated_qa_dir.exists():
            for json_file in curated_qa_dir.glob("*.json"):
                knowledge_files.append(json_file)
        
        # Case studies
        case_studies_dir = self.datasets_dir / "case_studies"
        if case_studies_dir.exists():
            for json_file in case_studies_dir.glob("*.json"):
                knowledge_files.append(json_file)
                
        logger.info(f"Found {len(knowledge_files)} knowledge files to process")
        
        # Process all knowledge files
        for knowledge_file in knowledge_files:
            self._process_file(knowledge_file)
        
        # Create embeddings
        self._create_embeddings()
        
        logger.info(f"Processed {len(self.document_store)} documents for RAG")
        self.initialized = True
    
    def _process_file(self, file_path: Path):
        """
        Process a knowledge file and add to the document store.
        
        Args:
            file_path: Path to the knowledge file
        """
        try:
            # First try with UTF-8 encoding and error handling
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # If that fails, try a more aggressive approach with binary reading
                logger.info(f"JSON decode error with utf-8 for {file_path}, trying alternative approach")
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # Replace or remove non-UTF-8 bytes
                cleaned_content = b''
                for i in range(0, len(content)):
                    byte = content[i:i+1]
                    try:
                        byte.decode('utf-8')
                        cleaned_content += byte
                    except UnicodeDecodeError:
                        # Replace problematic bytes
                        cleaned_content += b'\xEF\xBF\xBD'  # UTF-8 replacement character
                
                # Try parsing the cleaned content
                data = json.loads(cleaned_content.decode('utf-8'))
            
            # Process different file formats
            if isinstance(data, list):
                # Standard QA format
                for item in data:
                    if isinstance(item, dict) and "question" in item and "answer" in item:
                        document = {
                            "id": f"{file_path.stem}_{len(self.document_store)}",
                            "content": f"Question: {item['question']}\nAnswer: {item['answer']}",
                            "question": item["question"],
                            "answer": item["answer"],
                            "source": file_path.name,
                            "source_type": "qa_pair"
                        }
                        
                        # Add domain if available
                        if "domain" in item:
                            document["domain"] = item["domain"]
                        
                        self.document_store.append(document)
            
            elif isinstance(data, dict):
                # Handle case studies
                if "title" in data and "steps" in data:
                    # Extract case study content
                    content = f"Case Study: {data['title']}\nBackground: {data.get('background', '')}\n\n"
                    for step in data.get("steps", []):
                        content += f"{step.get('title', 'Step')}: {step.get('content', '')}\n\n"
                    
                    document = {
                        "id": f"{file_path.stem}_case_study",
                        "content": content,
                        "title": data.get("title", ""),
                        "domain": data.get("domain", ""),
                        "source": file_path.name,
                        "source_type": "case_study"
                    }
                    self.document_store.append(document)
                    
                    # Also add QA pairs if available
                    for i, qa_pair in enumerate(data.get("qa_pairs", [])):
                        if "question" in qa_pair and "answer" in qa_pair:
                            document = {
                                "id": f"{file_path.stem}_qa_{i}",
                                "content": f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}",
                                "question": qa_pair["question"],
                                "answer": qa_pair["answer"],
                                "domain": data.get("domain", ""),
                                "source": file_path.name,
                                "source_type": "case_study_qa"
                            }
                            self.document_store.append(document)
                
                # Handle knowledge graphs
                elif "entities" in data and "relations" in data:
                    # Create a textual representation of the graph
                    content = f"Knowledge Graph: {data.get('domain', '')}\n\nEntities:\n"
                    for entity_id, entity in data.get("entities", {}).items():
                        content += f"- {entity.get('name', '')}: {entity.get('type', '')}\n"
                    
                    content += "\nRelations:\n"
                    for relation in data.get("relations", []):
                        source_id = relation.get("source", "")
                        target_id = relation.get("target", "")
                        relation_type = relation.get("type", "")
                        
                        # Get entity names if available
                        source_name = data["entities"].get(source_id, {}).get("name", source_id)
                        target_name = data["entities"].get(target_id, {}).get("name", target_id)
                        
                        content += f"- {source_name} {relation_type} {target_name}\n"
                    
                    document = {
                        "id": f"{file_path.stem}_knowledge_graph",
                        "content": content,
                        "domain": data.get("domain", ""),
                        "source": file_path.name,
                        "source_type": "knowledge_graph"
                    }
                    self.document_store.append(document)
            
            logger.debug(f"Processed {file_path.name}, added {len(self.document_store)} documents")
                
        except Exception as e:
            logger.error(f"Error processing knowledge file {file_path}: {e}")
    
    def _create_embeddings(self):
        """
        Create embeddings for all documents in the document store.
        Uses TF-IDF vectorization as a placeholder for better embedding models.
        """
        if not hasattr(self, 'vectorizer') or self.vectorizer is None:
            logger.warning("Vectorizer not available, skipping embedding creation")
            return
            
        logger.info("Creating embeddings for document store...")
        
        if not self.document_store:
            logger.warning("No documents to create embeddings for")
            return
            
        # Extract content from documents
        texts = [doc["content"] for doc in self.document_store]
        
        try:
            # Create the vectors
            self.embedding_vectors = self.vectorizer.fit_transform(texts)
            logger.info(f"Created embeddings with shape {self.embedding_vectors.shape}")
            
            # Save the vector index (in a real system, use a proper vector DB)
            vectors_path = self.vector_db_path / "vector_index.npz"
            self._save_sparse_matrix(self.embedding_vectors, vectors_path)
            
            # Save document info
            docs_path = self.vector_db_path / "documents.json"
            with open(docs_path, 'w') as f:
                json.dump([{k: v for k, v in doc.items() if k != 'content'} 
                          for doc in self.document_store], f, indent=2)
                
            logger.info(f"Saved vector database to {self.vector_db_path}")
        
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
    
    def _save_sparse_matrix(self, matrix, path):
        """Save sparse matrix to a file."""
        import scipy.sparse
        if scipy.sparse.issparse(matrix):
            scipy.sparse.save_npz(path, matrix)
        else:
            np.save(path, matrix)
    
    def _load_sparse_matrix(self, path):
        """Load sparse matrix from a file."""
        import scipy.sparse
        if str(path).endswith('.npz'):
            return scipy.sparse.load_npz(path)
        else:
            return np.load(path)
    
    def load_vector_database(self):
        """
        Load the vector database from disk.
        """
        vectors_path = self.vector_db_path / "vector_index.npz"
        docs_path = self.vector_db_path / "documents.json"
        
        if not vectors_path.exists() or not docs_path.exists():
            logger.warning("Vector database not found. Please run process_knowledge_base() first.")
            return False
            
        try:
            # Load vectors
            self.embedding_vectors = self._load_sparse_matrix(vectors_path)
            
            # Load document info
            try:
                with open(docs_path, 'r', encoding='utf-8', errors='replace') as f:
                    doc_infos = json.load(f)
            except json.JSONDecodeError:
                # If that fails, try a more aggressive approach with binary reading
                logger.info(f"JSON decode error with utf-8 for {docs_path}, trying alternative approach")
                with open(docs_path, 'rb') as f:
                    content = f.read()
                
                # Replace or remove non-UTF-8 bytes
                cleaned_content = b''
                for i in range(0, len(content)):
                    byte = content[i:i+1]
                    try:
                        byte.decode('utf-8')
                        cleaned_content += byte
                    except UnicodeDecodeError:
                        # Replace problematic bytes
                        cleaned_content += b'\xEF\xBF\xBD'  # UTF-8 replacement character
                
                # Try parsing the cleaned content
                doc_infos = json.loads(cleaned_content.decode('utf-8'))
                
            # Load full documents
            self.document_store = []
            for doc_info in doc_infos:
                source = doc_info.get("source", "")
                source_type = doc_info.get("source_type", "")
                
                # Try to load the original content
                content = ""
                if source_type == "qa_pair" or source_type == "case_study_qa":
                    content = f"Question: {doc_info.get('question', '')}\nAnswer: {doc_info.get('answer', '')}"
                else:
                    # For other types, reconstruct from source files (ideally, store content in vector DB)
                    pass
                
                # Create document with content
                document = {**doc_info, "content": content}
                self.document_store.append(document)
                
            logger.info(f"Loaded vector database with {len(self.document_store)} documents")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant documents for a query using semantic search.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents
        """
        if not self.initialized:
            # Try to load database first
            success = self.load_vector_database()
            if not success:
                logger.warning("RAG system not initialized and failed to load vector database")
                return self._fallback_retrieval(query, top_k)
        
        if not hasattr(self, 'vectorizer') or self.vectorizer is None or self.embedding_vectors is None:
            logger.warning("Vector search not available, using fallback retrieval")
            return self._fallback_retrieval(query, top_k)
            
        try:
            # Create query vector
            query_vector = self.vectorizer.transform([query])
            
            # Compute similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, self.embedding_vectors).flatten()
            
            # Get top_k indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Get documents
            results = []
            for idx in top_indices:
                # Only include results with non-zero similarity
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    doc = self.document_store[idx]
                    results.append({
                        "document": doc,
                        "score": float(similarities[idx]),
                        "content": doc.get("content", ""),
                        "source": doc.get("source", ""),
                        "domain": doc.get("domain", "")
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during vector retrieval: {e}")
            return self._fallback_retrieval(query, top_k)
    
    def _fallback_retrieval(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Fallback keyword-based retrieval when vector search is not available.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents
        """
        logger.info("Using fallback keyword retrieval")
        
        # Simple keyword matching
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        results = []
        
        for doc in self.document_store:
            content = doc.get("content", "").lower()
            
            # Count matching terms
            matching_terms = sum(1 for term in query_terms if term in content)
            
            # Add to results if at least one term matches
            if matching_terms > 0:
                results.append({
                    "document": doc,
                    "score": matching_terms / len(query_terms) if query_terms else 0,
                    "content": doc.get("content", ""),
                    "source": doc.get("source", ""),
                    "domain": doc.get("domain", "")
                })
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def retrieve_and_generate(self, query: str, top_k: int = 3) -> Dict:
        """
        Retrieve relevant documents and format them for generation.
        
        Args:
            query: The user's query
            top_k: Number of top results to include
            
        Returns:
            Dict with retrieval results and formatted context
        """
        # Retrieve relevant documents
        results = self.retrieve(query, top_k)
        
        # Format context for model input
        formatted_context = ""
        if results:
            formatted_context = "Based on the following information:\n\n"
            for i, result in enumerate(results, 1):
                content = result["content"]
                source = result["source"]
                formatted_context += f"[Document {i} from {source}]\n{content}\n\n"
        
        return {
            "results": results,
            "formatted_context": formatted_context,
            "query": query
        }

def main():
    """Main function to process knowledge base and create vector database."""
    # Get project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = project_root / "Datasets"
    
    # Create retrieval augmentation system
    rag = RetrievalAugmentation(datasets_dir)
    
    # Process knowledge base
    rag.process_knowledge_base()
    
    # Test retrieval
    test_queries = [
        "What is zero trust security?",
        "How do I implement binary search in Python?",
        "What is the difference between TCP and UDP?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag.retrieve(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"Result {i} (Score: {result['score']:.3f}):")
            print(f"Source: {result['source']}")
            print(f"Content snippet: {result['content'][:100]}...")

if __name__ == "__main__":
    main()
