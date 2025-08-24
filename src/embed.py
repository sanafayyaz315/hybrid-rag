import numpy as np
import logging
from typing import List, Optional, Union
from fastembed import TextEmbedding, SparseTextEmbedding
from config import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class DenseEmbedder:
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 batch_size: int = 32
                 ):
        """
        Parameters:
        - model_name: name of sentence-transformers model
        - batch_size: batch size to use during encoding
        """
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.model = TextEmbedding(embedding_model_name)

        # embedding dimension
        self.embedding_dim = self.model.embedding_size
        logger.debug(f"Dense Embedder initialized with embedding_dim={self.embedding_dim}")

    def normalize_embed(self, vector: List):
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        return  vector / norm if norm != 0 else vector

    def embed(self, text: Union[str, List[str]], doc_type: str = "query", is_normalize: bool = False) -> np.ndarray:
        if not text:
            logger.info("Empty text provided. Nothing to embed (Dense).")
            return np.array([])   
                 
        try: 
            if doc_type not in ("query", "documents"):
                error_msg = "Invalid document type for embedding. Should be one of: ['query', 'documents']"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if doc_type == "documents":
                if not isinstance(text, list):
                    text = [text]
                embeddings = list(self.model.passage_embed(text))
            
            else: # embed query
                embeddings = list(self.model.query_embed(text))
            
            logger.info("Dense Embeddings created successfully")
            
            if is_normalize:
                logger.debug("Normalizing embeddings")
                embeddings = self.normalize_embed(embeddings)
                return embeddings
            
            logger.debug("NOT Normalizing embeddings")
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to create embeddings due to: {e}")
            raise

class SparseEmbedder:
    def __init__(
            self, 
            embedding_model_name: str = "Qdrant/bm25"
    ):
        """
        Parameters:
        - model_name: name of sentence-transformers model
        """
        self.model = SparseTextEmbedding(embedding_model_name)
        logger.debug(f"Sparse Embedder initialized")

    def embed(self, text: Union[str, List[str]]) -> list:
        if not text:
            logger.info("Empty text provided. Nothing to embed.")
            return np.array([])

        try:
            embeddings = list(self.model.embed(text))
            logger.info("Sparse embeddings created successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Unable to create dense embeddings due to : {e}")
        
if __name__ == "__main__":
    from config import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL
    embedder = DenseEmbedder(DENSE_EMBEDDING_MODEL)
    emb = embedder.embed("Hi my name is sana", doc_type="query")
    print(f"emebdding shape for single string: {emb.shape}")
    emb = embedder.embed(["Hi my name is sana", "How are you?"], doc_type="documents")
    print(f"emebdding shape for a list of strings: {emb.shape}")

    sparse_embedder = SparseEmbedder(SPARSE_EMBEDDING_MODEL)
    sp_emb = sparse_embedder.embed("Hi my name is sana")
    print(f"emebdding shape for single string: {emb.shape}")
    sp_emb = embedder.embed(["Hi my name is sana", "How are you?"])
    print(f"emebdding shape for a list of strings: {emb.shape}")

        

        


