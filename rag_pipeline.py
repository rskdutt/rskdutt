"""
title: RAG Pipeline with Pinecone (Compatible with LlamaIndex 0.12.x)
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for Retrieval-Augmented Generation (RAG) using LlamaIndex 0.12.x with Pinecone vector store , llama-index==0.12.52, pinecone
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, pinecone 
"""

from typing import List, Union, Generator, Iterator, Optional
import os
import tempfile
import base64
from pydantic import BaseModel

try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext
    from llama_index.core.node_parser import SimpleNodeParser
    #from llama_index.core.embeddings import OllamaEmbedding
    from llama_index.embeddings.ollama import OllamaEmbedding
    #from llama_index.llms import Ollama
    from llama_index.llms.ollama import Ollama
    from llama_index.vector_stores import PineconeVectorStore
    import pinecone
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False


class Pipeline:
    class Valves(BaseModel):
        PINECONE_API_KEY: str = "pcsk_4Xta5A_S93wFTEmRwmXeorserq5pvpeKmhAhXJW9W8hKX3DZKxxcipXsbSHw2MM6HJdqRn"
        PINECONE_INDEX_NAME: str = "open-webui-rag"
        PINECONE_ENVIRONMENT: str = "us-east-1"  # or your environment
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        EMBEDDING_MODEL: str = "nomic-embed-text"
        LLM_MODEL: str = "llama2"
        CHUNK_SIZE: int = 500
        CHUNK_OVERLAP: int = 200
        SIMILARITY_TOP_K: int = 5

    def __init__(self):
        self.type = "manifold"
        self.name = "RAG Pipeline with Pinecone"
        self.valves = self.Valves()
        self.index = None
        self.query_engine = None
        self.service_context = None
        
        if not LLAMA_INDEX_AVAILABLE:
            print("LlamaIndex not available. Please install: pip install llama-index==0.12.52 pinecone-client")
            return
            
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Initialize the RAG pipeline with Pinecone"""
        try:
            if not self.valves.PINECONE_API_KEY:
                print("Pinecone API key not set. Please configure PINECONE_API_KEY in valves.")
                return

            # Initialize Pinecone (v2.x compatible)
            pinecone.init(
                api_key=self.valves.PINECONE_API_KEY,
                environment=self.valves.PINECONE_ENVIRONMENT
            )
            
            # Check if index exists, create if not
            if self.valves.PINECONE_INDEX_NAME not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.valves.PINECONE_INDEX_NAME,
                    dimension=384,  # nomic-embed-text dimension
                    metric="cosine"
                )
                print(f"Created new Pinecone index: {self.valves.PINECONE_INDEX_NAME}")

            # Initialize embedding model
            embed_model = OllamaEmbedding(
                model_name=self.valves.EMBEDDING_MODEL,
                base_url=self.valves.OLLAMA_BASE_URL
            )

            # Initialize LLM
            llm = Ollama(
                model=self.valves.LLM_MODEL,
                base_url=self.valves.OLLAMA_BASE_URL,
                temperature=0.1
            )

            # Create node parser
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.valves.CHUNK_SIZE,
                chunk_overlap=self.valves.CHUNK_OVERLAP
            )

            # Create service context
            self.service_context = ServiceContext.from_defaults(
                embed_model=embed_model,
                llm=llm,
                node_parser=node_parser
            )

            # Initialize Pinecone vector store
            vector_store = PineconeVectorStore(
                pinecone_index=pinecone.Index(self.valves.PINECONE_INDEX_NAME)
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create or load index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                service_context=self.service_context
            )
            
            # Create query engine
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.valves.SIMILARITY_TOP_K,
                service_context=self.service_context
            )
            
            print("RAG pipeline with Pinecone initialized successfully")
            
        except Exception as e:
            print(f"Error setting up RAG pipeline: {e}")
            self.index = None
            self.query_engine = None

    def _process_files(self, files: List[dict]) -> bool:
        """Process uploaded files and add to vector store"""
        try:
            if not self.index or not self.service_context:
                print("Index not initialized")
                return False

            temp_files = []
            
            for file in files:
                if file.get("type") == "file" and file.get("file"):
                    file_data = file["file"]
                    filename = file_data.get("name", "unknown")
                    content = file_data.get("content", "")
                    
                    if not content:
                        continue
                    
                    # Decode base64 content
                    try:
                        file_bytes = base64.b64decode(content)
                    except Exception as e:
                        print(f"Error decoding file {filename}: {e}")
                        continue
                    
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(filename)[1],
                        prefix="rag_"
                    )
                    temp_file.write(file_bytes)
                    temp_file.close()
                    temp_files.append((temp_file.name, filename))

            if not temp_files:
                return False

            # Load and process documents
            file_paths = [temp_file[0] for temp_file in temp_files]
            documents = SimpleDirectoryReader(input_files=file_paths).load_data()
            
            if documents:
                # Add documents to index
                for doc in documents:
                    self.index.insert(doc)
                
                # Refresh query engine
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=self.valves.SIMILARITY_TOP_K,
                    service_context=self.service_context
                )
                
                print(f"Successfully processed {len(documents)} documents")
                success = True
            else:
                success = False

            # Clean up temporary files
            for temp_file, _ in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            return success

        except Exception as e:
            print(f"Error processing files: {e}")
            return False

    def pipes(self) -> List[dict]:
        return [{"id": "rag", "name": "RAG with Pinecone"}]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        
        if not LLAMA_INDEX_AVAILABLE:
            return "LlamaIndex libraries not installed. Please install: pip install llama-index==0.12.52 pinecone-client"
        
        if not self.valves.PINECONE_API_KEY:
            return "Please configure your Pinecone API key in the pipeline settings."
        
        if not self.query_engine:
            return "RAG pipeline not initialized. Please check your configuration."

        # This is where you can add your custom RAG pipeline.
        # Enhanced implementation with Pinecone vector store for document retrieval and synthesis.
        
        try:
            # Handle file uploads
            if body.get("files"):
                success = self._process_files(body["files"])
                if success:
                    return "✅ Documents successfully processed and added to Pinecone vector store. You can now ask questions about the content."
                else:
                    return "❌ Failed to process uploaded documents. Please check the files and try again."

            # Extract user question
            user_question = user_message
            if messages and len(messages) > 0:
                user_question = messages[-1].get("content", user_message)

            print(f"Processing query: {user_question}")

            # Query the RAG system
            response = self.query_engine.query(user_question)
            
            # Extract response text
            if hasattr(response, 'response'):
                answer = response.response
            else:
                answer = str(response)

            # Add source information
            source_info = ""
            if hasattr(response, 'source_nodes') and response.source_nodes:
                sources = []
                for i, node in enumerate(response.source_nodes[:3], 1):
                    if hasattr(node, 'extra_info'):
                        extra_info = node.extra_info
                        source = f"📄 Source {i}"
                        if 'file_name' in extra_info:
                            source += f": {extra_info['file_name']}"
                        if 'page_label' in extra_info:
                            source += f" (Page {extra_info['page_label']})"
                        sources.append(source)
                
                if sources:
                    source_info = f"\n\n**📚 Sources:**\n" + "\n".join(sources)

            final_response = answer + source_info
            print(f"Generated response with sources")
            
            return final_response

        except Exception as e:
            error_msg = f"Error in RAG pipeline: {str(e)}"
            print(error_msg)
            return "Sorry, I encountered an error while processing your request. Please try again."
