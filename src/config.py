from dotenv import load_dotenv, find_dotenv
import os
import sys
sys.path.append(os.path.abspath("../src"))
load_dotenv(find_dotenv(), override=True)


API_KEY = os.environ.get("API_KEY")
MODEL = os.environ.get("MODEL")

DENSE_EMBEDDING_MODEL = os.environ.get("DENSE_EMBEDDING_MODEL")
SPARSE_EMBEDDING_MODEL = os.environ.get("SPARSE_EMBEDDING_MODEL")
MAX_SEQ_LENGTH_EMBEDDING = int(os.environ.get("MAX_SEQ_LENGTH_EMBEDDING"))
CROSS_ENCODER_MODEL= os.environ.get("CROSS_ENCODER_MODEL")

QDRANT_HOST = os.environ.get("QDRANKT_HOST")
QDRANT_PORT = os.environ.get("QDRANT_POST")
COLLECTION = os.environ.get("COLLECTION")
DISTANCE = os.environ.get("DISTANCE")
SPARSE_MODIFIER=os.environ.get("SPARSE_MODIFIER")
UPSERT_BATCH_SIZE=int(os.environ.get("UPSERT_BATCH_SIZE"))
SYSTEM_PROMPT_DIR = os.environ.get("SYSTEM_PROMPT_DIR")
REWRITE_QUERY_PROMPT_PATH = os.environ.get("REWRITE_QUERY_PROMPT_PATH")
PARENT_CHUNKS_DIR = os.environ.get("PARENT_CHUNKS_DIR")
PARENT_CHUNKS_FILE = os.environ.get("PARENT_CHUNKS_FILE")

PARENT_CHUNK_SIZE = int(os.environ.get("PARENT_CHUNK_SIZE"))
PARENT_CHUNK_OVERLAP = int(os.environ.get("PARENT_CHUNK_OVERLAP"))
CHILD_CHUNK_SIZE = int(os.environ.get("CHILD_CHUNK_SIZE"))
CHILD_CHUNK_OVERLAP = int(os.environ.get("CHILD_CHUNK_OVERLAP"))

GET_NEIGHBORS = bool(os.environ.get("GET_NEIGHBORS"))

with open(SYSTEM_PROMPT_DIR) as f:
    SYSTEM_PROMPT = f.read()



if __name__ == "__main__":
    import os
    print(os.curdir)
    print(os.listdir())
    print(GET_NEIGHBORS)
    if GET_NEIGHBORS:
        print("yes")




