import logging, os, asyncio
from typing import Optional, List, Union, Tuple, Dict, Any, Awaitable, Generator
from sqlalchemy.ext.asyncio import AsyncSession
from src.file_loader import FileLoader
from src.chunker import TextChunker

from src.embed import DenseEmbedder, SparseEmbedder
from src.rerank import Rerank
from src.qdrant_utils import QdrantStore
from src.llm import LLM 
from src.config import *
from src.docstore.session import AsyncSessionLocal
from src.rag_utils import check_context_relevance
from src.builder import *
from src.docstore.docstore_crud import (async_retrieve_parent_chunks_from_docstore,
                                    async_retrieve_parent_neighbors)
from src.docstore.files_crud import async_ensure_unique_filenames
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class RagPipeline:
    def __init__(
            self,
            session: AsyncSession,
            loader: FileLoader,
            chunker: TextChunker,
            dense_embedder: DenseEmbedder,
            sparse_embedder: SparseEmbedder,
            vectorstore: Optional[QdrantStore],
            reranker: Rerank,
            llm: LLM,
            system_prompt: str,
            context_relevance_prompt: str,
            context_relevance: bool = True

    ):
        self.loader = loader
        self.chunker = chunker
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.llm = llm
        self.system_prompt = system_prompt
        self.context_relevance_prompt = context_relevance_prompt
        self.context_relevance = context_relevance
        self.session = session

    def load(self, path: Union[str, os.PathLike]) -> List[Tuple[str, str]]:
        """
        Loads text from a file or directory
        Args:
            path: path of a single file or a directory
        returns: List of tuples [(source, text)]
        """
        text = self.loader.load_files(path=path)
        return text
    
    def split(self, texts: List[Tuple[str, str]],
                    base_metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Takes List[Tuple[str, str]]: [(source, text)] and some optional base metadata
        Returns parent and child chunks 
        parent_chunk and child_chunk: {"text", "metadata"}
        """
        try:
            parent_chunks, child_chunks = self.chunker.parent_child_splitter(texts=texts, base_metadata=base_metadata)
            logger.debug(f"Text splitted into {len(parent_chunks)} parent and {len(child_chunks)} child chunks")
            return parent_chunks, child_chunks
        except Exception as e:
            logger.exception(f"Splitting failed: {e}")
            raise RuntimeError(f"Splitting failed: {e}")

    async def embed_and_index(self, texts: List[Dict], doc_type: str) -> None:
        """Embeds and indexes documents into the vector store.

        The function extracts text from the provided documents, generates both dense
        and sparse embeddings, and upserts them into the vector store along with
        their metadata.

        Args:
            texts (List[Dict]): 
                A list of document dictionaries. Each dictionary must include a
                "text" field and may include additional metadata fields.
                Example:
                    {
                        "text": "Document content here",
                        "source": "file1.pdf",
                        "id": "123"
                    }
            doc_type (str): 
                A string identifier describing the type of document (e.g., "query",
                "paragraph", "sentence"). This may be used by the embedder to select
                the appropriate embedding configuration.

        Returns:
            None

        Raises:
            RuntimeError: If no texts are provided or if embedding/indexing fails.
        """
        if not texts:
            logger.warning(f"No text provided to embed")
            raise RuntimeError(f"No text provided to embed. Len of texts {len(texts)}")
        
        chunks = [c["text"] for c in texts]
        # Dense and Sparse Embed documents
        try:
            dense_embeds = self.dense_embedder.embed(text=chunks, doc_type=doc_type)
            sparse_embeds = self.sparse_embedder.embed(text=chunks)
            await self.vectorstore.async_upsert(
                                    dense_embeddings=dense_embeds,
                                    sparse_embeddings=sparse_embeds,
                                    metadatas=texts
                                )
            logger.debug(f"Data indexed in vector DB in collection {self.vectorstore.collection_name}")
        except Exception as e:
            logger.exception(f"Unable to index embeddings: {e}")
            raise RuntimeError(f"Unable to index embeddings: {e}")
        
    async def retrieve(self, 
                        query: str,
                        top_k: int = 50,
                        rerank_top_k: int = 5,                        
                        sources: list = None,
                        rerank: bool = True,
                        retrieve_neighbors: bool = True                     
            ) -> List[Dict[str, str]]:
        """Retrieves documents relevant to a query from the vector store.

    The function performs hybrid retrieval using dense and sparse embeddings,
    fetches parent document chunks, and optionally applies reranking and
    neighbor retrieval for improved results.

    Args:
        query (str): 
            The input query string for which documents will be retrieved.
        top_k (int, optional): 
            Maximum number of candidate documents to retrieve before reranking. 
            Defaults to 50.
        rerank_top_k (int, optional): 
            Number of top documents to keep after reranking. 
            Defaults to 5.
        sources (List[str], optional): 
            A list of strings representing specific sources to restrict the search to. 
            Defaults to None (search all sources).
        rerank (bool, optional): 
            Whether to rerank the retrieved documents. 
            Defaults to True.
        retrieve_neighbors (bool, optional): 
            Whether to retrieve neighboring documents of the retrieved parents. 
            Defaults to True.

    Returns:
        Awaitable[List[Dict[str, Any]]]: 
            A list of retrieved document dictionaries. Each dictionary typically
            contains metadata and text for a parent chunk.

    Raises:
        RuntimeError: If the retrieval process fails at any stage.
    """
        # embed the query, similarity search, retrieve parent chunks, rerank, get neighbors
        try:
            dense_embeds = self.dense_embedder.embed(text=query, doc_type="query")
            sparse_embeds = self.sparse_embedder.embed(text=query)
            hits = await self.vectorstore.async_search(
                dense_query_vector=dense_embeds,
                sparse_query_vector=sparse_embeds,
                hybrid=True,
                top_k = top_k or self.top_k,
                sources=sources
            )
            async with self.session() as session:
                retrieved_parents = await async_retrieve_parent_chunks_from_docstore(hits=hits, session=session)

            if not rerank and not retrieve_neighbors:
                logger.debug(f"Retrieved {len(retrieved_parents)} documents, without reranking and neighbors")
                return retrieved_parents
            
            if rerank:
                scores = self.reranker.rerank(query=query,
                                   docs = retrieved_parents, 
                                   top_k = rerank_top_k
                                   )
                retrieved_parents = self.reranker.get_ranked_docs(docs=retrieved_parents, scores=scores)

            if retrieve_neighbors:
                async with self.session() as session:
                    retrieved_parents = await async_retrieve_parent_neighbors(docs=retrieved_parents, session=session)

            logger.debug(f"Retrieved {len(retrieved_parents)} with rerank={rerank} and neighbors={retrieve_neighbors}")
            return retrieved_parents
        
        except Exception as e:
            logger.exception(f"Failed to retrieve parents: {e}")
            raise RuntimeError(f"Failed to retrieve parents: {e}")
        
    def build_context(self, documents: List[Dict[str, str]]) -> List[str]:
        """Formats a list of document dictionaries into readable context strings.

        Each document is expected to contain a "text" field and an optional
        "metadata" dictionary with "source" and "id" keys. The function creates
        a formatted string for each document with its source, ID, and text.

        Args:
            documents (List[Dict[str, str]]): 
                A list of dictionaries where each dictionary represents a document.
                Example:
                    {
                        "text": "Some content",
                        "metadata": {
                            "source": "Database A",
                            "id": "123"
                        }
                    }

        Returns:
            List[str]: 
                A list of formatted strings, each including the source, ID,
                and text content of the corresponding document.
        """
        contexts = []
        for doc in documents:
            text = doc.get("text")
            source = doc.get("metadata", {}).get("source", "Unknown Source")
            id_ = doc.get("metadata", {}).get("id", "Unknown parent id")
            formatted = f"[Source: {source}, ID: {id_}]\nText: {text}\n]"
            contexts.append(formatted)
        return contexts
        
        
    async def generate_response(self,
                                query: str,
                                context: list,
                                ) -> Union[Generator[str, None, None], str]:
        """Generates a response from the language model using a query and context.

        The function streams a response from the LLM by providing it with a system
        prompt and user input that includes the query and context snippets. If
        context relevance checking is enabled, the function first evaluates how
        relevant the provided context is to the query. If the relevance score is
        low, it returns a fallback instructional message instead of an LLM response.

        Args:
            query (str):
                The user-provided question to be answered by the LLM.
            context (list):
                A list of context snippets (strings or structured content) to
                provide additional background information for the query.

        Returns:
            Union[Generator[str, None, None], str]:
                - A generator that streams response chunks from the LLM if the
                context is considered relevant or if context relevance is disabled.
                - A fallback instructional string if the relevance score indicates
                that the retrieved context is insufficient.

        Raises:
            RuntimeError: If an unexpected error occurs while generating the response.
        """  
        messages = []

        def stream_response(messages):      
                for chunk in self.llm.stream(messages):
                    yield chunk
  
        # sending context under the USER tag for efficient KV cache
        # For experiment, adjust the system prompt to have the context under the system tag, but at the very end
        # of the system prompt to encourage efficient KV cache
        messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content":f"User question:\n{query}\n\nContext snippets:\n{context}"})
    
        if not self.context_relevance:
            return stream_response(messages)
        try:
            logger.debug(f"Checking relevance score")
            relevance = await check_context_relevance(user_query=query,
                                                context=context,
                                                context_relevance_prompt=self.context_relevance_prompt,
                                                llm=self.llm                                               
                                            )
            logger.debug(f"Relevance score = {relevance['rating']}")

            rating = relevance.get("rating")
            if rating is None:
                raise KeyError("Missing 'rating' in relevance response.")

            if 2 <= rating <= 5:
                return stream_response(messages)
            
            else:
                    stream = f"""I couldn't find specific information to answer your question in the {COLLECTION} collection. 
                            To get the best results:
                            • Ensure your question relates to the documents available
                            • Use the "Get Files" option to see what documents are in this collection
                            • If the desired file is not present in the list of files, use "Upload File" to upload a new file.
                            • Try rephrasing your question with more context
                            The system did retrieve some content, but it didn't directly address your query."""
                    
                    return stream
        except Exception as e:
                logger.exception(f"Unable to generate response: {e}")
                raise RuntimeError(f"Unable to generate response: {e}")
        
if __name__ == "__main__":
    import asyncio
    from fastapi import UploadFile
    from tempfile import SpooledTemporaryFile
    from src.docstore.docstore_crud import async_upsert_parents_to_docstore
    from src.docstore.files_crud import async_stage_file_rows, async_commit_session
    from builder import (
    loader,
    chunker,
    dense_embedder,
    sparse_embedder,
    qdrant_store,
    reranker,
    llm,
    system_prompt,
    rewrite_query_prompt,
    context_relevance_prompt
    )
    from docstore.session import AsyncSessionLocal
    from rag_utils import rewrite_query

    def path_to_uploadfile(path: str) -> UploadFile:
        # Open the file in binary mode
        spooled_file = SpooledTemporaryFile()
        with open(path, "rb") as f:
            spooled_file.write(f.read())
        spooled_file.seek(0)  # reset pointer so UploadFile can read from start
        # Wrap in UploadFile
        return UploadFile(file=spooled_file, filename=path.split("/")[-1])

    pipeline = RagPipeline(
        session=AsyncSessionLocal,
        loader=loader,
        chunker=chunker,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        vectorstore=qdrant_store,
        reranker=reranker,
        llm=llm,
        system_prompt=system_prompt,
        context_relevance_prompt=context_relevance_prompt,
        context_relevance=True,
 
    )

    async def process_file(file_path, file, session):
        await async_ensure_unique_filenames(session=session, file=file)
        texts = pipeline.load(path=file_path)
        parent_chunks, child_chunks = pipeline.split(texts=texts, base_metadata=None)
        await pipeline.embed_and_index(texts=child_chunks, doc_type="documents")
        staged_file = await async_stage_file_rows(session=session, minio_metadata={"path": file_path, "object_name": file_path.split("/")[-1]})
        file_id_map = {staged_file.name: staged_file.id}
        saved = await async_upsert_parents_to_docstore(parent_chunks, session, file_id_map=file_id_map)
        return saved, staged_file

    async def generate(user_query, 
                 db_session,
                 system_prompt=system_prompt,
                 rewrite_query_prompt=rewrite_query_prompt, 
                 is_rewrite_query=True, 
                 llm=llm,                
                 top_k=50,
                 rerank=True,
                 rerank_top_k=5,
                 retrieve_neighbors=True,
                 context_relevance=True,
                 context_relevance_prompt=context_relevance_prompt
                 ):
        if is_rewrite_query:
            query, sources = await rewrite_query(user_query=user_query, rewrite_query_prompt=rewrite_query_prompt, llm=llm)
            logger.debug(f"Actual query: {user_query}\nRewritten query: {query}\n Sources: {sources}")
        retrieved_docs = await pipeline.retrieve(query=query, 
                                           top_k=top_k, 
                                           rerank=rerank, 
                                           rerank_top_k=rerank_top_k,
                                           retrieve_neighbors=retrieve_neighbors,
                                           session=db_session
                                           )
        logger.debug(f"Retrieved {len(retrieved_docs)} docs")
        context = pipeline.build_context(retrieved_docs)
        stream, contexts = await pipeline.generate_response(query, context, system_prompt, context_relevance=True, context_relevance_prompt=context_relevance_prompt)
        return stream, contexts

    async def main(query, process_file=True):
        file_path = "/Users/mac/Desktop/machine-learning/hybrid-rag/data/standard_labors_act.pdf"
        upload_file = path_to_uploadfile(file_path)

        if process_file:
            async with AsyncSessionLocal() as session:
                saved, staged_file = await process_file(file_path=file_path, session=session, file=upload_file)
                file = await async_commit_session(session, staged_file)

        async with AsyncSessionLocal() as session:
            stream, contexts = await generate(query, db_session=session)
            session.commit()
        return stream, contexts

    query = "What does the law say about annual paid leaves?"
    stream, contexts = asyncio.run(main(query, False))
    for chunk in stream:
        print(chunk, end="", flush=True)









                          
    




    
    

                

        


        










                






            



        
    


    
    



        
