from datasets import load_dataset
import pandas as pd
import json
import os
from config import (
                    DENSE_EMBEDDING_MODEL,
                    SPARSE_EMBEDDING_MODEL,
                    CROSS_ENCODER_MODEL
)
from rag_utils import load_files
from chunking import parent_child_splitter
from embed import DenseEmbedder, SparseEmbedder
from qdrant_utils import QdrantStore
from rerank import Rerank

ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
train, dev = ds["train"], ds["validation"]

collection_name = "eval"
data_dir = "../data/hotpot_paragraphs"
os.makedirs(data_dir, exist_ok=True)

max_chunks = 10
chunks_written = 0

question_answer = []  # store Q/A/supporting facts

for sample in dev:
    titles = sample["context"]["title"]
    paragraphs = sample["context"]["sentences"]

    # extract supporting fact titles for this sample
    supporting_titles = sample["supporting_facts"]["title"]
    # add entry to question_answer dict
    question_answer.append({
        "question": sample["question"],
        "answer": sample["answer"],
        "supporting_facts": supporting_titles,
        "type": sample["type"]
            })

    for i in range(len(paragraphs)):
        if chunks_written >= max_chunks:
            break
        try:
            text = " ".join(paragraphs[i]).strip()
            title = titles[i]

            # save paragraph to file using exact title
            file_path = os.path.join(data_dir, f"{title}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            chunks_written += 1  # increment only if save was successful

        except Exception as e:
            print(f"Skipping paragraph due to error: {e}")
            continue

    if chunks_written >= max_chunks:
        break

print(f"Saved {chunks_written} paragraph files successfully.")
print(f"Created {len(question_answer)} Q/A entries.")


texts = load_files(data_dir)
parent_chunks, child_chunks = parent_child_splitter(
                                                    parent_separators = ["\n\n", "\n", "."],
                                                    texts=texts,
                                                    metadata=None,                                               
                                                    parent_chunk_size=300,
                                                    parent_chunk_overlap=100,
                                                    child_chunk_size=150,
                                                    child_chunk_overlap=50,                                                       

                                             )
parent_map = {}
for p in parent_chunks:
    meta = p.get("metadata", {})
    src = meta.get("source", "")
    pid = meta.get("id", None)
    if src and pid is not None:
        parent_map[(src, int(pid))] = p

os.makedirs("../eval", exist_ok=True)  
with open("../eval/parents.json", "w") as f:
    json.dump(parent_chunks, f, indent=4)  # indent makes it pretty

# Get text and metadata to be encoded
chunks = [c["text"] for c in child_chunks]
metadata = [c["metadata"] for c in child_chunks]

dense_embedder = DenseEmbedder(DENSE_EMBEDDING_MODEL)
sparse_embedder = SparseEmbedder(SPARSE_EMBEDDING_MODEL)
qstore = QdrantStore(vector_size=dense_embedder.embedding_dim, collection_name=collection_name)
reranker = Rerank(CROSS_ENCODER_MODEL)

# 1. create embeddings of text
dense_embeds = dense_embedder.embed(text=chunks, doc_type="documents")
sparse_embeds = sparse_embedder.embed(chunks)

payload = []
for i in range(len(chunks)):
    payload.append({"text":chunks[i], "metadata": metadata[i]})

qstore.upsert(dense_embeds, sparse_embeds, payload, upsert_batch_size=500)

recall_list = []
for n, sample in enumerate(question_answer):
    query = sample["question"]
    answer = sample["answer"]
    relevant_docs = sample["supporting_facts"]

    dense_query_embed = dense_embedder.embed(query, doc_type="query")[0]
    sparse_query_embed = sparse_embedder.embed(query)

    hits = qstore.search(dense_query_vector=dense_query_embed, sparse_query_vector=sparse_query_embed, hybrid=True, top_k=50)
    retrieved_parents = []
    seen_keys = set()

    for hit in hits.points:
        m = hit.payload.get("metadata", {})
        key = (m.get("source", "")), int(m.get("parent_id", -1))
        if key in seen_keys:
            continue  # avoid duplicates if multiple child-chunks from same parent are returned
        parent = parent_map.get(key)
        if parent is not None:
            retrieved_parents.append(parent)
            seen_keys.add(key)

    print(f"retrieved_parents: {retrieved_parents[0]}")
    ranks = reranker.rerank(query, retrieved_parents, get_all=False, top_k=20)
    print(ranks)
    ranked_parents = []
    for rank in ranks:
        ranked_parents.append(retrieved_parents[rank["corpus_id"]])
    context = ranked_parents
    print(f"query: {query}")
    print(f"context:{context}")

    retrieved_top_k_sources = []
    for entry in context:
        retrieved_top_k_sources.append(entry["metadata"]["source"].replace(".txt",""))
    
    retrieved_top_k_sources = set(retrieved_top_k_sources)
    relevant_docs = set(relevant_docs)
    print(f"retrieved_top_k_sources {retrieved_top_k_sources}")
    print(f"relevant_docs: {relevant_docs}")
    relevant_retrieved = retrieved_top_k_sources.intersection(relevant_docs)
    print(relevant_retrieved)
    recall_k = len(relevant_retrieved)/len(relevant_docs)
    recall_list.append({"item_id": n, "recall_k": recall_k})
    
total_recall = 0.0
for item in recall_list:
    total_recall += item["recall_k"]
    
avg_recall = total_recall / len(recall_list)
print(recall_list)
print(avg_recall)
  









