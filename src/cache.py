from typing import Optional
from redis import Redis
from redisvl.extensions.cache.llm  import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

class RedisClient:
    """
    Simple wrapper for a Redis client with configurable connection parameters.
    Args:
        host (str): Redis server hostname. Defaults to "localhost".
        port (int): Redis server port. Defaults to 6379.

    Attributes:
        client (Redis): The underlying Redis client instance.
    """

    def __init__(self, host="localhost", port=6379):
        """
        Initialize the Redis client.

        Args:
            host (str): Redis server hostname. Defaults to "localhost".
            port (int): Redis server port. Defaults to 6379.
        """
        self.client = Redis(
            host=host,
            port=port,
            decode_responses=False,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30
        )

class RagSemanticCache:
    """
    Wrapper around RedisVL SemanticCache for use in RAG pipelines.

    Provides synchronous and asynchronous methods for storing and retrieving
    semantically similar query-response pairs in Redis.

    Attributes:
        client (Redis): Redis client used by the cache.
        vectorizer (HFTextVectorizer): Embedding model for queries.
        cache (SemanticCache): RedisVL SemanticCache instance.
    """
    def __init__(
            self,
            index_name: Optional[str] = "llmcache",
            embedding_model_name: str ="sentence-transformers/all-MiniLM-L6-v2",
            redis_client: Optional[Redis] = None,
            redis_url: str = "redis://localhost:6379",
            ttl: Optional[int] = None,
            prefix: Optional[str] = "llmcache",
            distance_threshold: int = 0.2
            ):
        """
        Initialize the RAG semantic cache.

        Args:
            index_name (str, optional): Name of the Redis index. Defaults to "llmcache".
            embedding_model_name (str): Name of the HuggingFace embedding model. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            redis_client (Redis, optional): Optional pre-initialized Redis client. If not provided, redis_url is used.
            redis_url (str): Redis connection URL, used if redis_client is None.
            ttl (int, optional): Default time-to-live (seconds) for cache entries. None = no expiration.
            prefix (str, optional): Prefix for Redis keys. Defaults to "llmcache".
            distance_threshold (float): Cosine distance threshold for semantic similarity. Defaults to 0.2.
        
            Attributes:
                client (Redis): The underlying Redis client instance.
                vectorizer (HFTextVectorizer): The embedding model for queries.
                cache (SemanticCache): The RedisVL SemanticCache instance.
        
        """
        self.client = redis_client or Redis.from_url(redis_url)
        self.vectorizer = HFTextVectorizer(model=embedding_model_name)
        self.cache = SemanticCache(
            name=index_name,
            vectorizer=self.vectorizer,
            redis_client=self.client,
            ttl=ttl,
            distance_threshold=distance_threshold,
            prefix=prefix
        )
    
    def lookup(self,
               prompt: str,
               top_k: int = 1
               
    ):
        """
        Retrieve semantically similar cached responses for a query.

        Args:
            prompt (str): User query or prompt.
            top_k (int): Number of nearest neighbors to return. Defaults to 1.

        Returns:
            list or None: List of cached results if found, otherwise None.
        """
        results = self.cache.check(prompt, num_results=top_k)
        return results if results else None
    
    def store(self,
              prompt: str,
              response: str = None,
              metadata: dict = None      
    ):
        """
        Store a query-response pair in the cache.

        Args:
            prompt (str): The query or prompt string.
            response (str, optional): The response to cache.
            metadata (dict, optional): Optional metadata to store with the entry.
        """
        self.cache.store(
            prompt=prompt,
            response=response,
            metadata=metadata or {}
        )

    async def alookup(self, prompt: str, top_k: int = 1):
        """
        Asynchronously retrieve semantically similar cached responses.

        Args:
            prompt (str): User query or prompt.
            top_k (int): Number of nearest neighbors to return. Defaults to 1.

        Returns:
            list or None: List of cached results if found, otherwise None.
        """
        return await self.cache.acheck(prompt, num_results=top_k)

    async def astore(self, prompt: str, response: str, metadata: dict = None):
        """
        Asynchronously store a query-response pair in the cache.

        Args:
            prompt (str): The query or prompt string.
            response (str): The response to cache.
            metadata (dict, optional): Optional metadata to store with the entry.
        """
        await self.cache.astore(prompt=prompt, response=response, metadata=metadata or {})

    def clear(self):
        """Clear all entries in the cache while keeping the index structure intact."""
        self.cache.clear()
    
    def delete(self):
        """Completely delete the cache index and all stored data."""
        self.cache.delete()

    def set_threshold(self, threshold: float):
        """
        Update the semantic similarity threshold for cache lookups.

        Args:
            threshold (float): New cosine distance threshold.
        """
        self.cache.set_threshold(threshold)
    
    def set_ttl(self, ttl: int):
        """
        Update the default TTL for new cache entries.

        Args:
            ttl (int): Time-to-live in seconds.
        """
        self.cache.set_ttl(ttl)


# --- IGNORE ---

if __name__ == "__main__":
    # Example usage
    redis_client = RedisClient(host="localhost", port=6379).client
    rag_cache = RagSemanticCache(redis_client=redis_client)
    metadata = {"source": "AI (Artificial Intelligence) is a field of study that intersects with statistics."}
    rag_cache.store("What is AI?", "AI stands for Artificial Intelligence.", metadata=metadata)
    result = rag_cache.lookup("Define AI", top_k=1)
   
    prompt = "Explain the process of photosynthesis"
    result = rag_cache.lookup(prompt, top_k=1)
    rag_cache.store("What is photosynthesis?", "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.", metadata={"source": "Photosynthesis occurs in the chloroplasts of plant cells."})
    print(result)

    



