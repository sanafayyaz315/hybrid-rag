from dotenv import load_dotenv, find_dotenv
import os
import sys
sys.path.append(os.path.abspath("../src"))
load_dotenv(find_dotenv(), override=True)


API_KEY = os.environ.get("API_KEY")
MODEL = os.environ.get("MODEL")

TEMP_FILE_DOWNLOAD_DIR = os.environ.get("TEMP_FILE_DOWNLOAD_DIR")

DENSE_EMBEDDING_MODEL = os.environ.get("DENSE_EMBEDDING_MODEL")
SPARSE_EMBEDDING_MODEL = os.environ.get("SPARSE_EMBEDDING_MODEL")
MAX_SEQ_LENGTH_EMBEDDING = int(os.environ.get("MAX_SEQ_LENGTH_EMBEDDING"))
CROSS_ENCODER_MODEL= os.environ.get("CROSS_ENCODER_MODEL")

QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_PORT = os.environ.get("QDRANT_PORT")
COLLECTION = os.environ.get("COLLECTION")
resources_string = os.environ.get("COLLECTION_RESOURCES")
COLLECTION_RESOURCES = [item.strip() for item in resources_string.split(",") if item.strip()]
DISTANCE = os.environ.get("DISTANCE")
SPARSE_MODIFIER=os.environ.get("SPARSE_MODIFIER")
DENSE_VECTOR_NAME=os.environ.get("DENSE_VECTOR_NAME")
SPARSE_VECTOR_NAME=os.environ.get("SPARSE_VECTOR_NAME")
UPSERT_BATCH_SIZE=int(os.environ.get("UPSERT_BATCH_SIZE"))

DOCSTORE_USER = os.getenv("DOCSTORE_USER")
DOCSTORE_PASSWORD = os.getenv("DOCSTORE_PASSWORD")
DOCSTORE_HOST = os.getenv("DOCSTORE_HOST")
DOCSTORE_PORT = os.getenv("DOCSTORE_PORT")
DOCSTORE_NAME = os.getenv("DOCSTORE_NAME")

PARENT_CHUNK_SIZE = int(os.environ.get("PARENT_CHUNK_SIZE"))
PARENT_CHUNK_OVERLAP = int(os.environ.get("PARENT_CHUNK_OVERLAP"))
CHILD_CHUNK_SIZE = int(os.environ.get("CHILD_CHUNK_SIZE"))
CHILD_CHUNK_OVERLAP = int(os.environ.get("CHILD_CHUNK_OVERLAP"))
GET_NEIGHBORS = bool(os.environ.get("GET_NEIGHBORS"))

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT")      # S3 API port
MINIO_ACCESS_KEY = os.environ.get("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET")

SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH")
REWRITE_QUERY_PROMPT_PATH = os.environ.get("REWRITE_QUERY_PROMPT_PATH")
CONTEXT_RELEVANCE_PROMPT_PATH = os.environ.get("CONTEXT_RELEVANCE_PROMPT_PATH")


if __name__ == "__main__":
    import os
    print(os.curdir)
    print(os.listdir())
    print(GET_NEIGHBORS)
    if GET_NEIGHBORS:
        print("yes")
    print(COLLECTION_RESOURCES)




