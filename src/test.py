import os
import logging
from embed import DenseEmbedder, SparseEmbedder
from qdrant_utils import QdrantStore
from chunking import parent_child_splitter
from rag_utils import retrieve_parent_chunks, load_files, retrieve_parent_neighbors
from config import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL, MAX_SEQ_LENGTH_EMBEDDING
from rerank import Rerank

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    path = "/Users/mac/Desktop/machine-learning/RAG/data/standard_labors_act.txt"
    texts = load_files(path)
    if os.path.isdir(path):
        logger.debug(f"Loaded files {os.listdir(path)}")
    else:
        logger.debug(f"Loaded file {path}")

    # 1) Chunk documents → child_chunks
    parent_chunks, child_chunks = parent_child_splitter(texts=texts,
                                                        metadata=None,
                                                        parent_chunk_size=2000,
                                                        parent_chunk_overlap=250,
                                                        child_chunk_size=400,
                                                        child_chunk_overlap=100,
                                                        )

    print(f"Length of parent chunks: {len(parent_chunks)} \nLength of child chunks: {len(child_chunks)}")

    # Get text and metadata to be encoded
    chunks = [c["text"] for c in child_chunks]
    metadata = [c["metadata"] for c in child_chunks]

    # 2) Dense and Sparse Embed documents
    dense_embedder = DenseEmbedder(DENSE_EMBEDDING_MODEL)
    dense_embeds = dense_embedder.embed(text=chunks, doc_type="documents")
  
    sparse_embedder = SparseEmbedder(SPARSE_EMBEDDING_MODEL)
    sparse_embeds = sparse_embedder.embed(chunks)
    print("Created embeddings")

    # 3) Build payload (text + metadata)
    payload = []
    assert len(metadata) == len(chunks), "Length of metadata and text chunks should be equal"
    for i in range(len(chunks)):
        payload.append({"text":chunks[i], "metadata": metadata[i]})

    # 4) Store embeddings
    qstore = QdrantStore(vector_size=dense_embedder.embedding_dim, collection_name="labor_law")
    qstore.upsert(dense_embeds, sparse_embeds, payload)

    # 5) Search
    print(f"Running similarity search")
    query = "What does Article 12 say?"
    dense_query_embed = dense_embedder.embed(query, doc_type="query")[0]
    sparse_query_embed = sparse_embedder.embed(query)

    hits = qstore.search(dense_query_vector=dense_query_embed, sparse_query_vector=sparse_query_embed, hybrid=True, top_k=50)

    for point in hits.points:
        print(point.id)
        print(point.score)
        print(point.payload["text"])
        print(point.payload["metadata"]["parent_id"])
        print(point.payload["metadata"]["child_id"])

    parents = retrieve_parent_chunks(hits, parent_chunks)
    print(f"Number of retrieved parents: {len(parents)}")
    print(parents[0])

    reranker = Rerank()
    ranks = reranker.rerank(query, parents, top_k = 10, get_all=False)

    neighbors_and_parents = retrieve_parent_neighbors(ranks, parents, parent_chunks)
    print(neighbors_and_parents)

    



        


    # print(len(ranks))
    # # Print the scores
    # print("Query:", query)
    # for rank in ranks:
    #     print(f"{rank['score']:.2f}\t{parents[rank['corpus_id']]}")





    
