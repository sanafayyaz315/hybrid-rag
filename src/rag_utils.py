from typing import List, Union, Tuple
import os
from sentence_transformers.cross_encoder import CrossEncoder

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

def retrieve_parent_neighbors(ranks, ranked_parents, parent_chunks):
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

def build_context():
    pass

if __name__ == "__main__":
    texts = load_files("/Users/mac/Desktop/machine-learning/RAG/data/chapters")
    for source, text in texts[0]:
        print(source)
        print(text)
    