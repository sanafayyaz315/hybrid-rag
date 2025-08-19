import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, SparseVectorParams, Distance, PointStruct, VectorStruct, SparseVector, FusionQuery, Fusion, Prefetch
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QdrantStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "test_vectors",
        vector_size: int = 768,
        distance: str = "Cosine",
        sparse_modifier: str = "idf",
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
):
        self.client = QdrantClient(url=f"http://{host}:{port}")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.sparse_modifier = sparse_modifier
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name

        self._create_collection_if_needed() # calling it here so a collection cab be created at the time of initialization

    # internal function    
    def _create_collection_if_needed(self):
        """
        Create the collection only if it does not already exist.
        """
        collection_exists = self.client.collection_exists(self.collection_name)

        if not collection_exists:
                logger.info(f"Creating new collection named {self.collection_name}")
                try:
                    self.client.create_collection(
                             collection_name=self.collection_name,
                             vectors_config={
                                  self.dense_vector_name: VectorParams(size=self.vector_size, distance=self.distance)
                             },
                             sparse_vectors_config={
                                  self.sparse_vector_name: SparseVectorParams(modifier=self.sparse_modifier)
                             })
                    logger.info(f"Successfully created collection {self.collection_name}")
                        
                except Exception as e:
                        logger.error(f"Unable to create collection {self.collection_name} due to exception {e}")
        else:
              logger.info(f"Using existing collection {self.collection_name}")

    def upsert(
        self,
        dense_embeddings: Optional[np.ndarray],
        sparse_embeddings: Optional[List[SparseVectorParams]],
        metadatas: List[Dict[str, Any]],
        upsert_batch_size: int
          # metadatas = [{"text": "chunktext", metadata:{id:0,...}]
    ):
        """
        embeddings: numpy array (n_points * dim)
        metadatas: list of dicts, same length as embeds
        """
            # Validate lengths
        n_points = len(metadatas)
        if dense_embeddings is not None and dense_embeddings.shape[0] != n_points:
            raise ValueError("Dense embeddings and metadata length must match!")
        if sparse_embeddings is not None and len(sparse_embeddings) != n_points:
            raise ValueError("Sparse embeddings and metadata length must match!")

        # create a list of data points in accordance with Qdrant format (PointStruct objects)
        points = []
        for idx in range(n_points):
            vector = {}
              
            if dense_embeddings is not None:
                vector[self.dense_vector_name] = dense_embeddings[idx].astype(np.float32)

            if sparse_embeddings is not None:
                indices = sparse_embeddings[idx].indices
                values = sparse_embeddings[idx].values
                vector[self.sparse_vector_name] = SparseVector(indices=indices, values=values)
                # vector[self.sparse_vector_name] = sparse_embeddings[idx]
            
            point = PointStruct(
                id = idx + 1,
                vector=vector,
                payload=metadatas[idx]
            )

            points.append(point)
        logger.info(f"Upserting {len(points)} points to Qdrant collection {self.collection_name}")

        try:
            
            for i in range(0, len(points), upsert_batch_size):
                 batch_points = points[i: i+upsert_batch_size]

                 self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                    # points=points
        )
            logger.info(f"Upserted {len(points)} points to {self.collection_name}")
        except Exception as e:
            logger.error(f"Upsert to Qdrant failed: {e}")
            raise 

    def search_depcr(self,
                 query_vector: np.ndarray,
                 top_k: int = 5,
                 filters: Optional[Dict] = None
                 ):
        """
        Returns top_k nearest matches as Qdrant scored points
        """
        try:
            search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filters,
            with_payload=True,
            with_vectors=True,
            )
            logger.debug(f"Found {len(search_results)} matches")

            self.client.query_points
            return search_results

        except Exception as e:
             logger.debug(f"An exception occured while running similariy search, Exception: {e}")
    
    def search(self, dense_query_vector, sparse_query_vector, hybrid=False, query_filter=None, top_k=50):
        if dense_query_vector is None and sparse_query_vector is None:
            raise ValueError("At least one of dense_query_vector or sparse_embed must be provided")
        else:
             sparse_query_vector = SparseVector(indices=sparse_query_vector[0].indices, 
                                                values=sparse_query_vector[0].values )
        
        # Base search params
        common_params = {
            "collection_name": self.collection_name,
            "query_filter": query_filter,
            "limit": top_k,
            "with_payload": True,
            "with_vectors": True
        }

        if dense_query_vector is not None and sparse_query_vector is not None:
            if hybrid:
                try:
                    search_result = self.client.query_points(
                        **common_params,              
                        prefetch=[
                            Prefetch(
                                query=dense_query_vector,
                                using="dense",
                                limit=top_k
                            ),
                            Prefetch(
                                query=sparse_query_vector,
                                using="sparse",
                                limit=top_k
                            )
                        ],
                        query=FusionQuery(fusion=Fusion.RRF),
                    )
                    return search_result
                except Exception as e:
                    logger.debug(f"Unable to perform hybrid search due to the following exception: {e}")
                    raise
                
            else:
                try:
                    dense_search_result = self.client.query_points(
                    **common_params,
                    query=dense_query_vector,
                    using="dense"
                    )

                    sparse_search_result = self.client.query_points(
                    **common_params,
                    query=sparse_query_vector,
                    using="sparse",
                    )

                    return {
                        "dense_search": dense_search_result,
                        "sparse_search": sparse_search_result
                    }
                except Exception as e:
                    logger.debug(f"Tried performing dense and sparse search independently. Failed due to: {e}")
                    raise

        elif dense_query_vector is not None:
            try:
                dense_search_result = self.client.query_points(
                    **common_params,
                    query=dense_query_vector,
                    using="dense"
                    )
                return dense_search_result
            except Exception as e:
                    logger.debug(f"Tried performing dense search. Failed due to: {e}")
                    raise
            
        else:
            try:
                sparse_search_result = self.client.query_points(
                    **common_params,
                    query=sparse_query_vector,
                    using="sparse"
                    )
                return sparse_search_result
            except Exception as e:
                    logger.debug(f"Tried performing sparse search. Failed due to: {e}")
                    raise

    def clear(self):
        """
        Delete the entire collection (dangerous).
        """
        logger.warning(f"Dropping collection {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.warning(f"{self.collection_name} Dropped")
            

                
if __name__ == "__main__":
    import numpy as np
    import logging
    from qdrant_client.http.models import SparseVectorParams  # or your sparse vector import

    logging.basicConfig(level=logging.DEBUG)

    # Initialize QdrantStore
    store = QdrantStore(
        host="localhost",
        port=6333,
        collection_name="test_hybrid_collection_10",
        vector_size=2,
        distance="Cosine",
        sparse_modifier="idf",
        dense_vector_name="dense",
        sparse_vector_name="sparse"
    )

    # Create dummy data for testing
    n_points = 3
    dense_embs = np.random.rand(n_points, 2).astype(np.float32)

    # For sparse embeddings, create dummy SparseVectorParams objects
    # Here just creating empty SparseVectorParams for example, replace with real sparse embeddings
    sparse_embs = [
        SparseVector(indices=[0, 1], values=[1.0, 0.5]),
        SparseVector(indices=[2, 3], values=[0.7, 0.2]),
        SparseVector(indices=[4, 5], values=[0.9, 0.1]),
    ]

    metadatas = [
        {"text": "Document 1", "metadata": {"id": 1}},
        {"text": "Document 2", "metadata": {"id": 2}},
        {"text": "Document 3", "metadata": {"id": 3}},
    ]

    # Upsert embeddings + metadata
    store.upsert(dense_embeddings=dense_embs, sparse_embeddings=sparse_embs, metadatas=metadatas)

    # Prepare query embeddings (dummy again)
    query_dense = np.random.rand(2).astype(np.float32)
    query_sparse = SparseVector(indices=[1, 2], values=[0.6, 0.3])

    # # Run hybrid search
    # print("Hybrid Search Results:")
    # results = store.search(query_dense=query_dense, query_sparse=query_sparse, top_k=2, mode="hybrid")
    # for r in results:
    #     print(r)

    # Dense only search
    print("\nDense Search Results:")
    results = store.search(query_dense=query_dense, top_k=2, mode="dense")
    for r in results:
        print(r)

    # Sparse only search
    print("\nSparse Search Results:")
    results = store.search(query_sparse=query_sparse, top_k=2, mode="sparse")
    for r in results:
        print(r)

    # Both (individual) searches merged
    print("\nBoth Mode (Separate) Search Results:")
    results = store.search(query_dense=query_dense, query_sparse=query_sparse, top_k=2, mode="both")
    for r in results:
        print(r)

    # Uncomment to clear collection after test
    # store.clear()

            
    

            


                    
