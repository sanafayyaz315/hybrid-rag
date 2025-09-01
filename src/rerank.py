from sentence_transformers.cross_encoder import CrossEncoder
from typing import List
import logging
# # Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    handlers=[logging.StreamHandler()]
)

# Create a module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Rerank:
    def __init__(
            self,
            reranking_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
            late_interaction: bool = False
    ):
        """
        Parameters:
        - reranking_model: name of the reranking model - by default: cross-encoder/ms-marco-MiniLM-L6-v2
        - late_interaction: Whether to use late interaction or not. If False, reranking will be done using a cross-encoder
        """
        if not late_interaction:
            self.rerank_model = CrossEncoder(reranking_model)
        
    def rerank(
            self,
            query: str,
            docs: List,
            top_k: int = 3,
            get_all: bool = False
    ):
        """
        Rerank documents using the cross-encoder model.

        Args:
            query (str): The user query to rerank against.
            docs (List): A list of documents (dicts) to rerank. Each dict should contain a "text" field.
            top_k (int, optional): Number of top results to return. Defaults to 3.
            get_all (bool, optional): If True, return scores for all hits. If False, return only the top_k. Defaults to False.

        Returns:
            List[dict]: A list of score dictionaries sorted by relevance, where each dict contains:
                        - "corpus_id": Index of the document in the original hits list
                        - "score": Relevance score assigned by the reranker
        """
        if not docs:
            logger.warning("No documents provided to retrieve neighbors.")
            return []
        
        rerank_corpus = [doc["text"] for doc in docs]
        try:
            scores = self.rerank_model.rank(query, rerank_corpus)
        except Exception as e:
            logger.debug(f"Reranking failed due to: {e}")
            raise RuntimeError(f"Reranking failed due to: {e}")

        return scores if get_all else  scores[:top_k]
        
    def get_ranked_docs(
        self,
        docs: List,
        scores: List[dict]
    ):
        """
        Map rerank scores back to the original documents.

        Args:
            docs (List): The original list of documents (dicts) that were passed to rerank().
            scores (List[dict]): The scores returned by rerank(), each containing:
                                 - "corpus_id": Index of the document in the hits list
                                 - "score": Relevance score

        Returns:
            List[dict]: Ranked documents, where each document dict includes:
                        - All original document fields
                        - "rerank_score": The score assigned by the reranker
        """
        if not docs:
            logger.warning("No documents provided to retrieve neighbors.")
            return []
        
        id_to_doc = {i: doc for i, doc in enumerate(docs)}

        ranked_docs = []
        for s in scores:
            doc = id_to_doc[s["corpus_id"]]
            doc_with_score = {**doc, "rerank_score": s["score"]}
            ranked_docs.append(doc_with_score)

        return ranked_docs
    
if __name__ == "__main__":
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]

    dataset = []
    for c in corpus:
        dataset.append({"text": c})
    
    reranker = Rerank()

    ranks = reranker.rerank(query, dataset)

    # Print the scores
    print("Query:", query)
    for rank in ranks:
        print(f"{rank['score']:.2f}\t{rank['corpus_id']}\t{dataset[rank['corpus_id']]}")

    print(ranks)
    ranked_docs = reranker.get_ranked_docs(dataset, scores=ranks)
    print(ranked_docs)

