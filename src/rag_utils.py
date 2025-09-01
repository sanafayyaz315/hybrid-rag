from typing import List, Union, Tuple, Dict
import os, json
from sqlalchemy import select
from sqlalchemy.orm import Session
from fastapi import UploadFile
from src.llm import LLM


import os
import fitz  # PyMuPDF

def load_files(path: Union[str, os.PathLike]) -> List[Tuple[str, str]]:
    """
    Loads text from .txt and .pdf files in a given directory or from a single file.

    Args:
        path (str or os.PathLike): Path to a directory or a single file (.txt or .pdf).

    Returns:
        list[tuple[str, str]]: A list of tuples (filename, text_content).
        Returns an empty list if the path is invalid or no supported files are found.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If the provided path is neither a file nor a directory.
    """
     
    texts = []
    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    # Handle single file
    if os.path.isfile(path):
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext in [".txt", ""]:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    texts.append((filename, text))
            elif ext == ".pdf":
                pdf_text = []
                with fitz.open(path) as pdf_doc:
                    for page in pdf_doc:
                        pdf_text.append(page.get_text("text"))
                texts.append((filename, "\n".join(pdf_text)))
            else:
                print(f"Skipping unsupported file type: {filename}")
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    # Handle directory
    elif os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            ext = os.path.splitext(file)[1].lower()

            try:
                if ext in [".txt", ""]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        texts.append((file, text))
                elif ext == ".pdf":
                    pdf_text = []
                    with fitz.open(file_path) as pdf_doc:
                        for page in pdf_doc:
                            pdf_text.append(page.get_text("text"))
                    texts.append((file, "\n".join(pdf_text)))
                else:
                    print(f"Skipping unsupported file type: {file}")
            except Exception as e:
                print(f"Error reading file {file}: {e}")

    else:
        raise ValueError(f"Path is neither a file nor a directory: {path}")

    return texts


def retrieve_parent_chunks(hits: List, parent_chunks: List):
    """
    hits: search results for child chunks
    parent_chunks: list of parent chunks
    """
    # First, build a lookup dictionary for fast retrieval
    parent_lookup = {
        (p["metadata"]["source"], p["metadata"]["id"]): p
        for p in parent_chunks
    }

    seen = set()
    unique_parents = []

    for hit in hits.points:
        key = (hit.payload["metadata"]["source"], hit.payload["metadata"]["parent_id"])
        if key not in seen:
            parent_chunk = parent_lookup.get(key)
            if parent_chunk:
                unique_parents.append(parent_chunk)
                seen.add(key)
    
    return unique_parents

def retrieve_parent_chunks_from_docstore(hits: list, session: Session):
    """
    Fetch unique parent chunks from the docstore table based on child hits.
    """
    seen_keys = set()
    parent_chunks = []

    for hit in hits.points:
        source = hit.payload["metadata"]["source"]
        parent_id = hit.payload["metadata"]["parent_id"]

        key = (source, parent_id)
        if key in seen_keys:
            continue

        # Fetch parent chunk from DB
        parent = session.query(Docstore).filter_by(
            source=source,
            parent_id=parent_id
        ).first()

        if parent:
            parent_chunks.append({
                "text": parent.text,
                "metadata": {
                    "source": parent.source,
                    "id": parent.parent_id,
                    **(parent.chunk_metadata or {})
                }
            })
            seen_keys.add(key)

    return parent_chunks

def retrieve_parent_neighbors_json(ranks, ranked_parents, parent_chunks):
    """
    Takes ranks of parent chunk, and finds its previous and next parent chunks
    Input: 
        ranks: output of crossencoder
        ranked_parents: parents retrieved from ranks
        parent_chunks: list of parent chunks 
    Output:
        Returns a list of parents chunks that include the actualed ranked parents and their naighbors
    """
    parent_lookup = {
        (p["metadata"]["source"], p["metadata"]["id"]): p
        for p in parent_chunks
    }
    neighbors = []

    for rank in ranks:
        data = ranked_parents[rank["corpus_id"]]
        id_ = data.get("metadata", {}).get("id", "Unknown parent id")
        source = data.get("metadata", {}).get("source", "Unknown Source")

        if (id_ >= 0):
            concatenated_neighbors = ""
            prev_id = id_ - 1
            next_id = id_ + 1
            for neighbor_id in [prev_id, id_, next_id]:
                parent_chunk = parent_lookup.get((source, neighbor_id))
                if parent_chunk:
                    concatenated_neighbors += parent_chunk["text"] + " "

            neighbors.append({"text": concatenated_neighbors, "metadata": {"source": source}})

    return neighbors


def retrieve_parent_neighbors(ranks, retrived_parents, session: Session):
    """
    Fetch ranked parents + their previous and next neighbors directly from DB.
    """
    neighbors = []

    for rank in ranks:
        data = retrived_parents[rank["corpus_id"]]
        parent_id = data.get("metadata", {}).get("id")
        source = data.get("metadata", {}).get("source")

        if parent_id is None or source is None:
            continue

        # Query only the needed neighbors (prev, current, next)
        neighbor_ids = [parent_id - 1, parent_id, parent_id + 1]

        rows = (
            session.execute(
                select(Docstore)
                .where(
                    Docstore.source == source,
                    Docstore.parent_id.in_(neighbor_ids),
                )
                .order_by(Docstore.parent_id)  # ensures correct sequence
            )
            .scalars()
            .all()
        )

        concatenated_neighbors = " ".join([row.text for row in rows])
        neighbors.append({
            "text": concatenated_neighbors,
            "metadata": {"source": source}
        })

    return neighbors

def save_files_locally(local_dir: str, file: UploadFile):
    file_path = os.path.join(local_dir, file.filename)
    with open(file_path, "wb") as f:
            content = file.read()
            f.write(content)
            return file_path


# save files to local path. Turn this to a function
                # file_path = os.path.join(upload_dir, file.filename)
                # with open(file_path, "wb") as f:
                #     content = await file.read()
                #     f.write(content)
                #     saved_files.append(file_path)

async def check_context_relevance(user_query: str, context: List[Dict], llm: LLM, context_relevance_prompt: str) -> Dict:
    """ 
    Assess the relevance of a given context with respect to a user query using an LLM.

    Parameters:
        user_query (str): The query or question provided by the user.
        context (List[Dict]): A list of context items (each as a dictionary) to be evaluated for relevance.
        llm (LLM): An asynchronous language model instance capable of generating responses.
        context_relevance_prompt (str): A template prompt for the LLM that defines how relevance should be evaluated.
                                         The template should include placeholders for the user query and context.

    Returns:
        Dict: A dictionary containing the relevance assessment, structured as:
            {
                "remarks": "<brief rationale for the rating, as a text>",
                "rating": "<your rating, as a number between 1 and 5>"
            }
    """
    messages = []
    formatted_context_relevance_prompt = context_relevance_prompt.format(message=user_query, context=context)
    messages.append({"role": "system", "content": formatted_context_relevance_prompt})
    relevance_response = await llm.async_invoke(messages)
    relevance = json.loads(relevance_response)
    return relevance

async def rewrite_query(user_query: str, llm: LLM, rewrite_query_prompt: str) -> Tuple[str, List]:
    """
    Rewrite a user query for improved retrieval or execution and optionally extract relevant sources.

    Parameters:
        user_query (str): The original query input by the user.
        llm (LLM): An asynchronous language model instance capable of generating responses.
        rewrite_query_prompt (str): A template prompt for the LLM instructing how to rewrite the query and
                                    optionally extract sources.

    Returns:
        Tuple[str, List]: A tuple containing:
            - query (str): The rewritten or refined query text.
            - sources (List): An optional list of relevant sources extracted by the LLM (can be None).

    Example:
        rewritten_query, sources = await rewrite_query(
            user_query="Find details about machine learning",
            llm=my_llm,
            rewrite_query_prompt="Rewrite the query to be more precise and suggest sources if applicable"
        )
    """
    messages = []
    messages.append({"role": "system", "content": rewrite_query_prompt})
    messages.append({"role": "user", "content": user_query})
    rewritten_res = json.loads(await llm.async_invoke(messages))
    query = rewritten_res["query"]
    sources = rewritten_res.get("sources")
    return query, sources

if __name__ == "__main__":
    texts = load_files("/Users/mac/Desktop/machine-learning/RAG/data/chapters")
    for source, text in texts[0]:
        print(source)
        print(text)
    