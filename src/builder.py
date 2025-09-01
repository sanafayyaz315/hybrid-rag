from src.file_loader import FileLoader
from src.chunker import TextChunker
from qdrant_client import QdrantClient, AsyncQdrantClient
from minio import Minio
from src.embed import DenseEmbedder, SparseEmbedder
from src.rerank import Rerank
from src.qdrant_utils import QdrantStore
from src.rag import RagPipeline
from src.llm import LLM
from src.docstore.setup import init_db
from src.docstore.session import AsyncSessionLocal
from src.config import *
import logging


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# initialize loader
loader = FileLoader()

# initialize chunker
CHILD_CHUNK_SIZE = CHILD_CHUNK_SIZE or MAX_SEQ_LENGTH_EMBEDDING - 30   
chunker = TextChunker(
    parent_chunk_size=PARENT_CHUNK_SIZE,
    parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
    child_chunk_size=CHILD_CHUNK_SIZE,
    child_chunk_overlap=CHILD_CHUNK_OVERLAP
)

# Initialize Dense and Sparse embedders
dense_embedder = DenseEmbedder(embedding_model_name=DENSE_EMBEDDING_MODEL)
sparse_embedder = SparseEmbedder(embedding_model_name=SPARSE_EMBEDDING_MODEL)

# Initialize Reranker
reranker = Rerank(reranking_model=CROSS_ENCODER_MODEL)

# initialize Qdrant clients
qdrant_client = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    # prefer_grpc=True,
)

async_qdrant_client = AsyncQdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    # prefer_grpc=True,
)

# Initialize Vector store
qdrant_store = QdrantStore(
                        client=qdrant_client,
                        async_client=async_qdrant_client,
                        host=QDRANT_HOST,
                        port=QDRANT_PORT,
                        collection_name=COLLECTION,
                        vector_size=dense_embedder.embedding_dim,
                        distance=DISTANCE,
                        sparse_modifier=SPARSE_MODIFIER,
                        dense_vector_name=DENSE_VECTOR_NAME,
                        sparse_vector_name=SPARSE_VECTOR_NAME
                        )

# initialize minio client and bucket
def build_minio_client(minio_bucket:str = MINIO_BUCKET) -> Minio:
    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    # Ensure bucket exists
    if not client.bucket_exists(minio_bucket):
        logger.info(f"Creating MinIO bucket {minio_bucket}")
        client.make_bucket(minio_bucket)
    return client
minio_client = build_minio_client()

# Initilaize docstore database
init_db()

# initialize LLM
llm = LLM(
    model=MODEL,
    api_key=API_KEY
)

# Initialize prompts
with open(SYSTEM_PROMPT_PATH) as f:
    system_prompt = f.read()

with open(REWRITE_QUERY_PROMPT_PATH) as f:
    rewrite_query_prompt = f.read()

with open(CONTEXT_RELEVANCE_PROMPT_PATH) as f:
    context_relevance_prompt = f.read()

# Initialize Rag pipeline
pipeline = RagPipeline(
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
        session=AsyncSessionLocal
    )

if __name__ == "__main__":
    minio_client = build_minio_client()
    print(minio_client)
    