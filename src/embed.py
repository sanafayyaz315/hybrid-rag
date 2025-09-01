import numpy as np
import logging
from typing import List, Optional, Union
from fastembed import TextEmbedding, SparseTextEmbedding
from src.config import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL

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
                 ):
        """
        Initialize the DenseEmbedder with a FastEmbed TextEmbedding model.

        Parameters:
            embedding_model_name (str): Name of the embedding model. By default, 
                a sentence-transformers model is used. FastEmbed's TextEmbedding 
                is loaded with this model name.

        Notes:
            - FastEmbed automatically normalizes embeddings. 
        """
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.model = TextEmbedding(embedding_model_name)

        # embedding dimension
        self.embedding_dim = self.model.embedding_size
        logger.debug(f"Dense Embedder initialized with embedding_dim={self.embedding_dim}")

    def normalize_embed(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector or array of vectors to unit length.

        Parameters:
            vector (np.ndarray): The input vector or array of vectors to normalize.

        Returns:
            np.ndarray: Normalized vector(s) with unit L2 norm. 
                - If the input vector has zero norm, it is returned unchanged.

        Notes:
            Normalization is useful when embeddings need to be compared using 
            cosine similarity, ensuring that all vectors lie on the unit hypersphere.
        """
        norm = np.linalg.norm(vector)
        return  vector / norm if norm != 0 else vector

    def embed(self, text: Union[str, List[str]], doc_type: str, is_normalize: bool = False) -> np.ndarray:
        """
        Embed text using the dense model.

        Parameters:
            text (str or List[str]): Text or list of texts to embed.
            doc_type (str): Either "query" or "documents". Determines which embedding method to use.
            is_normalize (bool): Whether to normalize the resulting embedding vectors. 
                - The DenseEmbedder creates a fastembed TextEmbedding object, for which the embeddings are already normalized, so this can typically be left as False.

        Returns:
            np.ndarray: Embedding vector(s) as a numpy array.
        """

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
            
            embeddings = np.array(embeddings)
            
            if is_normalize:
                logger.debug("Normalizing embeddings")
                embeddings = self.normalize_embed(embeddings)

            else:
                logger.debug("NOT Normalizing embeddings")

            logger.info(f"Dense Embeddings successfully created with shape {embeddings.shape}")
            if doc_type == "documents":
                return embeddings
            else:
                return embeddings[0]
            
        except Exception as e:
            logger.error(f"Failed to create dense embeddings.")
            raise RuntimeError(f"Dense embeddings not created due to : {e}")

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
        """
        Create sparse embeddings for input text(s).

        Args:
            text (str | List[str]): A single string or list of strings to embed.

        Returns:
            np.ndarray: Array of embeddings. Shape (n_texts, embedding_dim) if list, (embedding_dim,) if single string.
        """
        if not text:
            logger.info("Empty text provided. Nothing to embed.")
            # return np.array([])
            return []

        try:
            embeddings = list(self.model.embed(text))
            logger.info("Sparse embeddings created successfully")
            # return np.array(embeddings)
            return embeddings
        
        except Exception as e:
            logger.error(f"Unable to create sparse embeddings")
            raise RuntimeError(f"Sparse embeddings not created due to : {e}")

        
if __name__ == "__main__":
    from config import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL
    dense_embedder = DenseEmbedder(DENSE_EMBEDDING_MODEL)
    dense_emb = dense_embedder.embed("Hi my name is sana", doc_type="query")
    print(f"emebdding shape for single string: {dense_emb.shape}")
    emb = dense_embedder.embed(["Hi my name is sana", "How are you?"], doc_type="documents")
    print(f"emebdding shape for a list of strings: {emb.shape}")

    sparse_embedder = SparseEmbedder(SPARSE_EMBEDDING_MODEL)
    sp_emb = sparse_embedder.embed("Hi my name is sana")
    print(f"sparse emebdding shape for single string: {sp_emb}")
    print("dense_embedding:", type(dense_emb))



        

        


