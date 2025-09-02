from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional, Tuple
import uuid
from src.config import MAX_SEQ_LENGTH_EMBEDDING

class TextChunker:
    def __init__(
            self,
            parent_chunk_size: int = 2000,
            parent_chunk_overlap: int = 200,
            child_chunk_size: int = 512,
            child_chunk_overlap: int = 80,
            parent_separators: Optional[List[str]] = None,
            child_separators: Optional[List[str]] = None
    ):

        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.parent_separators = parent_separators or ["\n\n", "\n", "."]
        self.child_separators = child_separators or ["\n\n", "\n", "."]
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Split raw text into parent chunks.
        Args:
            text (str): The raw text to be split.
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with each chunk.
        Returns:
            List[Dict[str, Any]]:  Each chunk is represented as a dictionary with 'text' and 'metadata' keys.
        If metadata is provided, it is included in each chunk's metadata.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=self.parent_separators
        )
        chunks = splitter.split_text(text)
        docs = []
        for idx, chunk in enumerate(chunks):
            md = metadata.copy() if metadata else {}
            md["id"] = idx
            docs.append({"text": chunk, "metadata": md})
        return docs

    def split_children(self, parent_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split parent chunks into child chunks
        Args:
            parent_chunks (List[Dict[str, Any]]): List of parent chunks, each represented as a dictionary with 'text' and 'metadata' keys.
        Returns:
            List[Dict[str, Any]]: List of child chunks, each represented as a dictionary with 'text' and 'metadata' keys.
        Each child chunk's metadata includes a reference to its parent chunk's ID.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=self.child_separators
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
                child_chunks.append({
                    "text": chunk,
                    "metadata": metadata
                })
            # child_id += 1
        return child_chunks
        
    def parent_child_splitter(self,
                            texts: List[Tuple[str, str]], 
                            base_metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict], List[Dict]]:
        """Orchestrates splitting of multiple texts into parent and child chunks.
        Args:
            texts (List[Tuple[str, str]]): List of tuples where each tuple contains (source, text).
            base_metadata (Optional[Dict[str, Any]]): Optional base metadata to include with each chunk.
        Returns:
            Tuple[List[Dict], List[Dict]]: A tuple containing two lists - the first is the list of parent chunks,
            and the second is the list of child chunks. Each chunk is represented as a dictionary with 'text' and 'metadata' keys.  
        """
        parents = []
        children = []
        for source, text in texts:
            metadata = base_metadata.copy() if base_metadata else {}
            metadata["source"] = source
            parent_chunks = self.split_text(text, metadata)
            child_chunks = self.split_children(parent_chunks)
            parents.extend(parent_chunks)
            children.extend(child_chunks)
    
        return parents, children 

if __name__ == "__main__":
    from config import MAX_SEQ_LENGTH_EMBEDDING
    from rag_utils import load_files
    
    source_file = "/Users/mac/Desktop/machine-learning/RAG/data/articles"
    texts = load_files(source_file)
    chunker = TextChunker()

    parent_chunks, child_chunks = chunker.parent_child_splitter(texts=texts,
                                                        base_metadata=None,          
                                                        )
    print(f"")
    print(f"len of parent chunks: {len(parent_chunks)}")
    print(f"parent chunks: {parent_chunks[0:6]}")

    print(f"len of children: {len(child_chunks)}")
    print(f"first child chunk: {child_chunks[50]}")
    print(child_chunks[0])
    print(parent_chunks[0])





    


