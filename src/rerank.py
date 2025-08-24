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
            hits: List,
            top_k: int = 3,
            get_all: bool = False
    ):

        rerank_corpus = [hit["text"] for hit in hits]
        try:
            scores = self.rerank_model.rank(query, rerank_corpus)
        except Exception as e:
            logger.debug(f"Reranking failed due to: {e}")
            raise

        if get_all:
            return scores
        else:
            return scores[:top_k]
    
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




