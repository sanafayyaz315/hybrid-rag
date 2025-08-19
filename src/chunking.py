from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional, Tuple
import uuid
from config import MAX_SEQ_LENGTH_EMBEDDING

def split_text(
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 5000,
        chunk_overlap: int = 200,
        separators: List[str] =  ["\n\n", "\n", "."],
        return_lc_document: bool = False, 
            
):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    chunks = text_splitter.split_text(text)

    docs = []
    for idx, chunk in enumerate(chunks):
        # remaining metadata will be the same as the metadata for the unsplitted chunk
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata["id"] = idx

        if return_lc_document:
            # LangChain styles document
            lc_doc = text_splitter.create_documents([chunk], [chunk_metadata])
            docs.extend(lc_doc)
        
        else:
            docs.append({
                "text": chunk, 
                "metadata": chunk_metadata
                })
        
    return docs

def child_splitter(
        parent_chunks: List[Dict[str, Any]],
        child_chunk_size: int = MAX_SEQ_LENGTH_EMBEDDING - 30,
        child_chunk_overlap: int = 50,
        separators: List[str] = ["\n\n", "\n", "."],
        return_lc_documents: bool = True
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=separators
    )
    child_chunks = []
    child_id = 0

    for parent in parent_chunks:
        chunks = text_splitter.split_text(parent["text"])

        for chunk in chunks:
            # keep parent info + parent_id
            metadata = parent["metadata"].copy()
            metadata["parent_id"] = metadata.pop("id", None) # rename id as parent_id
            metadata["child_id"] = child_id
            child_id += 1

            if return_lc_documents:
                lc_doc = text_splitter.create_documents([chunk], [metadata])
                child_chunks.extend(lc_doc)

            else:
                child_chunks.append({
                    "text": chunk,
                    "metadata": metadata
                })
        child_id += 1


    return child_chunks
    
def parent_child_splitter(
        texts: List[Tuple[str]],
        metadata: Optional[Dict[str, Any]] = None,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = MAX_SEQ_LENGTH_EMBEDDING - 30,
        child_chunk_overlap: int = 80,
        parent_separators: List[str] = ["\n\n", "\n", "."],
        child_separators: List[str] = ["\n\n", "\n", "."],
        return_parent_lc_documents: bool = False,
        return_child_lc_documents: bool = False,
):  
    parents = []
    children = []
    for text in texts:
        if not metadata:
            metadata = {}   
        metadata["source"] = text[0]
        data = text[1]

        # 1. split into parent chunks
        parent_chunks = split_text(
            text=data,
            metadata=metadata,
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=parent_separators,
            return_lc_document=return_parent_lc_documents,
        )
        parents.extend(parent_chunks)

        # 2. split each parent chunk into child chunk
        child_chunks = child_splitter(
            parent_chunks=parent_chunks,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            separators=child_separators,
            return_lc_documents=return_child_lc_documents
        )
        children.extend(child_chunks)
   
    return parents, children 

if __name__ == "__main__":
    from config import MAX_SEQ_LENGTH_EMBEDDING
    from rag_utils import load_files
    
    source_file = "/Users/mac/Desktop/machine-learning/RAG/data/articles"
    texts = load_files(source_file)
    parent_chunks, child_chunks = parent_child_splitter(texts=texts,
                                                        metadata=None,
                                                        parent_chunk_size=3000,
                                                        parent_chunk_overlap=500,
                                                        child_chunk_size=MAX_SEQ_LENGTH_EMBEDDING-30,
                                                        child_chunk_overlap=150,
                                                        )
    print(f"")
    print(f"len of parent chunks: {len(parent_chunks)}")
    print(f"parent chunks: {parent_chunks[0:6]}")

    print(f"len of children: {len(child_chunks)}")
    print(f"first child chunk: {child_chunks[50]}")





    


