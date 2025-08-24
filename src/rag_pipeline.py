import os
import shutil
import logging
import json
from typing import List
from sqlalchemy.orm import Session
from models import Docstore
from embed import DenseEmbedder, SparseEmbedder
from qdrant_utils import QdrantStore
from chunking import parent_child_splitter
from rag_utils import retrieve_parent_chunks_from_docstore, load_files, retrieve_parent_neighbors
from rerank import Rerank
from llm import LLM
from config import (
    COLLECTION,
    PARENT_CHUNKS_FILE, 
    PARENT_CHUNKS_DIR, 
    PARENT_CHUNK_SIZE, 
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)


# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
# Create a module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RagPipeline:
    def __init__(
        self,
        collection_name: str,
        llm_api_key: str,
        llm_model: str = "gpt-3.5-turbo",
        system_prompt_path: str = "../template/rag.txt",
        rewrite_query_prompt_path = None,
        dense_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_embedding_model_name: str = "Qdrant/bm25",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2", 
        upsert_batch_size: int = 500,  
        top_k: int = 50,
        rerank_top_k: int = 3,
        parent_chunks_dir: str = PARENT_CHUNKS_DIR,
        parent_chunks_file: str = PARENT_CHUNKS_FILE,
        max_seq_len_embedding: int = 512,
        parent_chunk_size: int = PARENT_CHUNK_SIZE,
        parent_chunk_overlap: int = PARENT_CHUNK_OVERLAP,
        child_chunk_size: int = CHILD_CHUNK_SIZE,
        child_chunk_overlap: int = CHILD_CHUNK_OVERLAP,
        docstore_session = None,
        get_neighbors: bool = True,
        retriever_eval: bool = False,
        collection_resources: str = ""
    ):
        self.collection_name = collection_name
        self.llm_api_key = llm_api_key
        self.model = llm_model
        self.top_k = top_k
        self.upsert_batch_size = upsert_batch_size
        self.rerank_top_k = rerank_top_k
        self.parent_chunks_dir = parent_chunks_dir
        self.parent_chunks_file = parent_chunks_file
        self.dense_embedder = DenseEmbedder(dense_embedding_model_name)
        self.sparse_embedder = SparseEmbedder(sparse_embedding_model_name)
        self.qstore = QdrantStore(vector_size=self.dense_embedder.embedding_dim, collection_name=self.collection_name)
        self.reranker = Rerank(cross_encoder_model_name)
        self.retriever_eval = retriever_eval
        self.llm = LLM(model=self.model, api_key=self.llm_api_key)
        self.parent_file_path = os.path.join(self.parent_chunks_dir, self.parent_chunks_file)
        self.max_seq_len_embedding = max_seq_len_embedding
        self.get_neighbors = get_neighbors
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_overlap = child_chunk_overlap
        self.docstore_session = docstore_session
        self.collection_resources = collection_resources
        if not child_chunk_size:
            self.child_chunk_size = self.max_seq_len_embedding - 30
        else:
            self.child_chunk_size = child_chunk_size

        logger.debug(f"Child chunk size set to : {self.child_chunk_size}")

        with open(system_prompt_path) as f:
            self.system_prompt = f.read()

        if rewrite_query_prompt_path:
            with open(rewrite_query_prompt_path) as f:
                self.rewrite_query_prompt = f.read()
            if self.collection_resources:
                formatted_sources_string = str([self.collection_resources])
                self.rewrite_query_prompt = self.rewrite_query_prompt.format(sources=formatted_sources_string)
        else:
            self.rewrite_query_prompt = None

    def process_file(
            self,
            # file_path_dir: str,
            session, 
            save_file_path:str = "/tmp/uploaded_docs",
            file_id_map = {},
    
    ):
        # 1. Load file(s) and save to a temp dir
        # This code can be replaced to download the file from an object storage like Minio

        texts = load_files(save_file_path)

        # 2. Create chunks
        parent_chunks, child_chunks = parent_child_splitter(
                                                        parent_separators = ["\n\n", "\n", "."],
                                                        texts=texts,
                                                        metadata=None,                                               
                                                        parent_chunk_size=self.parent_chunk_size,
                                                        parent_chunk_overlap=self.parent_chunk_overlap,
                                                        child_chunk_size=self.child_chunk_size,
                                                        child_chunk_overlap=self.child_chunk_overlap,                                                       
                                                        )
        logger.debug(f"Length of parent chunks: {len(parent_chunks)} \nLength of child chunks: {len(child_chunks)}")

        # save parent_chunks so can be used during retrieval
        # def _append_to_parents_json(new_entries: list, file_path: str = None):
        #     """
        #     Efficiently appends multiple new entries to parents.json, skipping duplicates.
        #     - Checks for duplicates using (source, id) composite key
        #     - Only reads/writes the file once
        #     - Early exit if no existing data
        #     """
        #     file_path = file_path or self.parent_file_path
        #     os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        #     try:
        #         # Load existing data
        #         existing_data = []
        #         if os.path.exists(file_path):
        #             with open(file_path, "r") as f:
        #                 existing_data = json.load(f)
                
        #         # Early exit if no existing data (all new entries are unique)
        #         if not existing_data:
        #             with open(file_path, "w") as f:
        #                 json.dump(new_entries, f, indent=2)
        #                 logger.debug(f"Saved {len(new_entries)} new parent chunks. Total number of parent chunks is {len(new_entries)}")
        #             return len(new_entries)
                
        #         # Create set of existing keys for fast lookup
        #         existing_keys = {
        #             (entry["metadata"]["source"], entry["metadata"]["id"])
        #             for entry in existing_data
        #         }

        #          # Filter out duplicates from new entries
        #         unique_new_entries = []
        #         duplicates = []

        #         for entry in new_entries:
        #             key = (entry["metadata"]["source"], entry["metadata"]["id"])
        #             if key in existing_keys:
        #                 duplicates.append(entry)
        #             else:
        #                 unique_new_entries.append(entry)
        #                 existing_keys.add(key)  # Update set to catch intra-batch duplicates
                
        #         # Append if we found any unique entries
        #         if unique_new_entries:
        #             existing_data.extend(unique_new_entries)
        #             with open(file_path, "w") as f:
        #                 json.dump(existing_data, f, indent=2)
        #         logger.debug(f"Saved {len(unique_new_entries)} new parent chunks. Found {len(duplicates)} duplicates. Total number of parent chunks is {len(existing_data)}")
        #         return len(unique_new_entries)
        
        #     except Exception as e:
        #         logger.error(f"Error updating {file_path}: {str(e)}")
        #         raise

        def upsert_parents_to_docstore(new_entries: list, session = session) -> int:
            """
            Inserts parent chunks into the docstore table, skipping duplicates.
            new_entries: list of dicts with keys: source, id, text, metadata
            Input:
                new_entries: list of dicts of parent_chunks with keys source, id, text, metadata
                session: SQLALchemy session
            """
            saved = 0
            if session:
                for entry in new_entries:
                    # Check if this (source, parent_id) already exists
                    exists = session.query(Docstore).filter_by(
                        source=entry["metadata"]["source"], parent_id=entry["metadata"]["id"]
                    ).first()

                    if not exists:
                        doc = Docstore(
                            file_id=file_id_map[entry["metadata"]["source"]],
                            source=entry["metadata"]["source"],
                            parent_id=entry["metadata"]["id"],
                            text=entry["text"],
                            chunk_metadata=entry.get("metadata", {})
                        )
                        session.add(doc)
                        saved += 1
                # session.commit()
                return saved
            else:
                logger.debug("No doctore session found")
            
        def _save_parents(new_entries: list, session):
            try:      
                # Create a new session for this operation
                # with self.docstore_session() as session:  # docstore_session is actually SessionLocal
                saved_count = upsert_parents_to_docstore(new_entries, session=session)
                return saved_count
            except Exception as e:
                logger.exception(f"Error saving parent chunks: {e}")
                raise
        
        # num_saved_parents = _append_to_parents_json(parent_chunks)
        num_saved_parents = _save_parents(parent_chunks, session)
        # session.commit()
        logger.debug(f"{num_saved_parents} parent chunks saved in docstore")

        # Get text and metadata to be encoded
        chunks = [c["text"] for c in child_chunks]
        metadata = [c["metadata"] for c in child_chunks]

        # 3 Dense and Sparse Embed documents
        dense_embeds = self.dense_embedder.embed(text=chunks, doc_type="documents")
        sparse_embeds = self.sparse_embedder.embed(chunks)
        logger.debug("Successfully created Dense and Sparse embeddings")

        # 4. Build payload (text + metadata) and store embeddings in vector db
        payload = []
        assert len(metadata) == len(chunks), "Length of metadata and text chunks should be equal"
        for i in range(len(chunks)):
            payload.append({"text":chunks[i], "metadata": metadata[i]})

        self.qstore.upsert(dense_embeds, sparse_embeds, payload, self.upsert_batch_size)
        logger.debug(f"Data indexed in vector DB in collection {self.collection_name}")

    def chat(
            self,
            query: str
    ):
        # 0. rewrite query
        if self.rewrite_query_prompt:
            logger.debug("Rewriting Query")
            user_query = query
            messages = []
            messages.append({"role": "system", "content": self.rewrite_query_prompt})
            messages.append({"role": "user", "content": user_query})
            rewritten_res = json.loads(self.llm.invoke(messages))
            query = rewritten_res["query"]
            sources = rewritten_res.get("sources")

            logger.debug(f"Actual User Query: {user_query}\nRewritten Query: {query}\nFilter: {sources}")
        else:
            logger.debug(f"Did not rewrite query as path to rewrite query prompt is {self.rewrite_query_prompt} ")

        # 1. embed query
        dense_query_embed = self.dense_embedder.embed(query, doc_type="query")[0]
        sparse_query_embed = self.sparse_embedder.embed(query)

        # 2. Similarity search
        hits = self.qstore.search(dense_query_vector=dense_query_embed, sparse_query_vector=sparse_query_embed, hybrid=True, top_k=50, sources=sources)
        logger.debug(f"Similarity search completed!. Found {len(hits.points)} child chunks")

        # 3. retrieve parent chunks

        # this block used parents,json
        # with open(self.parent_file_path, "r") as f:
        #     parent_chunks = json.load(f)
        # parents = retrieve_parent_chunks(hits, parent_chunks)

        # loading parents from docstore
        with self.docstore_session() as session:  # SessionLocal
            parents = retrieve_parent_chunks_from_docstore(hits, session)

        if parents:
            logger.debug(f"Number of retrieved parents: {len(parents)}")
        else:
            logger.debug("No valid context found for the user query.")
            return "No valid context found for the user query.", ""

        # 4. rerank parent chunks
        ranks = self.reranker.rerank(query, parents, get_all=False, top_k=self.rerank_top_k)
        logger.debug(f"Reranked parent chunks successfully. Number of parent chunks: {len(ranks)}")

        # 5. build context and prompt
        contexts = []

        if self.get_neighbors:
            with self.docstore_session() as session:
                neighbors = retrieve_parent_neighbors(ranks, parents, session)
            for item in neighbors:
                text = item["text"]
                source = item.get("metadata", {}).get("source", "Unknown Source")
                id_ = item.get("metadata", {}).get("id", "Unknown parent id")
                formatted = f"[Source: {source}, ID: {id_}]\nText: {text}\n"
                contexts.append(formatted)
            logger.debug(f"Retrieved neighboring parent chunks. Total number of parent chunks {len(neighbors)}")

        else:
            for rank in ranks:
                data = parents[rank["corpus_id"]]
                text = data.get("text", "")
                source = data.get("metadata", {}).get("source", "Unknown Source")
                id_ = data.get("metadata", {}).get("id", "Unknown parent id")
                formatted = f"[Source: {source}, ID: {id_}]\nText: {text}\n]"
                contexts.append(formatted)

        if not self.retriever_eval:
            # 6. generate response
            messages = []
            messages.append({"role": "system", "content": self.system_prompt})
            # sending context under the USER tag. For experiment, adjust the 
            # system prompt to have the context under the system tag, at the very end
            # of the system prompt to encourage efficient KV cache
            messages.append({"role": "user", "content": f"User question:\n{query}\n\nContext snippets:\n{contexts}"})
            
            def stream():      
                for chunk in self.llm.stream(messages):
                    # print(chunk, end="", flush=True)
                    yield chunk
            
            # response = ""
            # for chunk in self.llm.stream(messages):
            #     print(chunk, end="", flush=True)
            #     response += chunk

            # return response, contexts
            return stream(), contexts
        else:
            logger.debug(f"retriever_eval: {self.retriever_eval}. In Retriever Eval mode")
            return None, contexts

if __name__ == "__main__":
    from config import API_KEY

    collection_name = "legal_law"
    source_data_dir = "../data/articles"
    system_prompt_path = "../template/rag.txt"
    query = "What are the different kinds of errors in a program?"
    query = "What is the 'math' module in python and why is it used?"

    pipeline = RagPipeline(
        llm_api_key=API_KEY,
        collection_name=collection_name,
        system_prompt_path=system_prompt_path,
        rerank_top_k=20,
        )
    
    pipeline.upload_file(
        file_path_dir=source_data_dir
        )
    
    response, contexts = pipeline.chat(query=query)
    # print(f"-----CONTEXTS----")
    # print(contexts)
    print(f"-----RESPONSE----")
    print(response)
    















        
        

