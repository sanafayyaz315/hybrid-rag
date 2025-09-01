from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select
from typing import List, Dict
import logging
from src.docstore.models import Docstore, File

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def upsert_parents_to_docstore(
        new_entries: list,
        session: Session,
        file_id_map: dict
):
    """
    Inserts parent chunks into the docstore table, skipping duplicates.
    Does not commit the transaction.

    Args:
        new_entries (list): List of dicts with keys: source, id, text, metadata
        session (Session): SQLAlchemy session
        file_id_map (dict): Mapping from source -> file_id

    Returns:
        int: Number of new parent chunks saved
    """
    saved = 0
    if not session:
        logger.debug(f"No database session for docstore is provided.")
        return saved
    
    try:
        for entry in new_entries:
            source = entry["metadata"]["source"]
            parent_id = entry["metadata"]["parent_id"]

            # check if the same parent chunk already exists
            exists = session.query(Docstore).filter_by(source=source, parent_id=parent_id).first()
            if not exists:
                doc = Docstore(
                    file_id=file_id_map[source],
                    source=source,
                    parent_id=parent_id,
                    text=entry["text"],
                    chunk_metadata=entry.get("metadata", {})
                )
                session.add(doc)
                saved +=1
        
        logger.debug(f"{saved} new parent chunks added to docstore")
        return saved
            
    except SQLAlchemyError as e:
            logger.error("Failed to save parents to docstore")
            raise RuntimeError (f"Failed to save parents to docstore: {e}")
             
async def async_upsert_parents_to_docstore(
        new_entries: list,
        session: AsyncSession,
        file_id_map: dict
) -> int:
    """
    Async version: Inserts parent chunks into the docstore table, skipping duplicates.
    Does not commit the transaction.

    Args:
        new_entries (list): List of dicts with keys: source, id, text, metadata
        session (AsyncSession): SQLAlchemy async session
        file_id_map (dict): Mapping from source -> file_id

    Returns:
        int: Number of new parent chunks saved
    """
    saved = 0
    if not session:
        logger.debug("No async database session provided for docstore.")
        return saved

    try:
        for entry in new_entries:
            source = entry["metadata"]["source"]
            parent_id = entry["metadata"]["id"]

            # check if the same parent chunk already exists
            result = await session.execute(
                Docstore.__table__.select().where(
                    Docstore.source == source,
                    Docstore.parent_id == parent_id
                )
            )
            exists = result.scalar_one_or_none()

            if not exists:
                doc = Docstore(
                    file_id=file_id_map[source],
                    source=source,
                    parent_id=parent_id,
                    text=entry["text"],
                    chunk_metadata=entry.get("metadata", {})
                )
                session.add(doc)
                saved += 1
        logger.debug(f"{saved} new parent chunks added to docstore")
        return saved
            
    except SQLAlchemyError as e:
            logger.error("Failed to save parents to docstore")
            raise RuntimeError (f"Failed to save parents to docstore: {e}")
    

def retrieve_parent_chunks_from_docstore(hits: list, session: Session) -> list:
    """
    Fetch unique parent chunks from the docstore table based on child hits.

    Args:
        hits (list): List of objects with points containing payload metadata
        session (Session): SQLAlchemy session

    Returns:
        list: List of dicts containing parent chunk text and metadata
    """

    if not session:
        logger.debug(f"No database session for docstore is provided.")
        return []
    
    seen_keys = set()
    parent_chunks = []

    for hit in hits.points:
        source = hit.payload["metadata"]["source"]
        parent_id = hit.payload["metadata"]["parent_id"]

        key = (source, parent_id)
        if key in seen_keys:
            continue

        # Fetch parent chunk from DB
        parent = session.query(Docstore).filter_by(source=source, parent_id=parent_id).first()
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
    logger.debug(f"Retrieved {len(parent_chunks)} unique parent chunks from docstore")
    return parent_chunks

async def async_retrieve_parent_chunks_from_docstore(hits: list, session: AsyncSession) -> List[Dict[str,str]]:
    """
    Fetch unique parent chunks from the docstore table based on child hits (async version).

    Args:
        hits (list): List of objects with points containing payload metadata
        session (AsyncSession): SQLAlchemy async session

    Returns:
        list: List of dicts containing parent chunk text and metadata
    """
    if not session:
        logger.debug("No database session for docstore is provided.")
        return []

    seen_keys = set()
    parent_chunks = []

    for hit in hits.points:
        source = hit.payload["metadata"]["source"]
        parent_id = hit.payload["metadata"]["parent_id"]

        key = (source, parent_id)
        if key in seen_keys:
            continue

        # Async query for the parent chunk
        result = await session.execute(
            select(Docstore).where(
            Docstore.source == source,
            Docstore.parent_id == parent_id
            )
        )
        
        parent = result.scalar_one_or_none()

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

    logger.debug(f"Retrieved {len(parent_chunks)} unique parent chunks from docstore (async)")
    return parent_chunks

def retrieve_parent_neighbors(docs, session: Session):
    """
    Fetch ranked parent documents along with their immediate neighbors (previous and next) from the database.

    Args:
        docs (List[Dict]): List of ranked documents, each containing at least a "metadata" field with "id" and "source".
        session (Session): SQLAlchemy session for querying the database.

    Returns:
        List[Dict]: A list of dictionaries where each dict contains:
                    - "text": Concatenated text of the parent and its immediate neighbors
                    - "metadata": Dictionary containing the "source" of the parent document

    Notes:
        - Only parent documents with valid "id" and "source" in metadata are considered.
        - Neighboring documents are determined by `parent_id - 1` and `parent_id + 1`.
        - The resulting list preserves the order of the input `docs`.
    """
    if not session:
        logger.debug(f"No database session for docstore is provided.")
        return []

    if not docs:
        logger.warning("No documents provided to retrieve neighbors.")
        return []
    
    neighbors = []

    for doc in docs:
        parent_id = doc.get("metadata", {}).get("id")
        source = doc.get("metadata", {}).get("source")

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

async def async_retrieve_parent_neighbors(docs, session: AsyncSession) -> list:
    """
    Fetch ranked parent documents along with their immediate neighbors (previous and next) from the database.
    Uses AsyncSession.

    Args:
        docs (List[Dict]): List of ranked documents, each containing at least a "metadata" field with "id" and "source".
        session (Session): SQLAlchemy session for querying the database.

    Returns:
        List[Dict]: A list of dictionaries where each dict contains:
                    - "text": Concatenated text of the parent and its immediate neighbors
                    - "metadata": Dictionary containing the "source" of the parent document

    Notes:
        - Only parent documents with valid "id" and "source" in metadata are considered.
        - Neighboring documents are determined by `parent_id - 1` and `parent_id + 1`.
        - The resulting list preserves the order of the input `docs`.
    """
    if not session:
        logger.debug(f"No database session for docstore is provided.")
        return []
        
    if not docs:
        logger.warning("No documents provided to retrieve neighbors.")
        return []

    neighbors = []

    for doc in docs:
        parent_id = doc.get("metadata", {}).get("id")
        source = doc.get("metadata", {}).get("source")

        if parent_id is None or source is None:
            continue

        neighbor_ids = [parent_id - 1, parent_id, parent_id + 1]

        result = await session.execute(
            select(Docstore)
            .where(Docstore.source == source, Docstore.parent_id.in_(neighbor_ids))
            .order_by(Docstore.parent_id)
        )
        rows = result.scalars().all()

        concatenated_neighbors = " ".join([row.text for row in rows])
        neighbors.append({
            "text": concatenated_neighbors,
            "metadata": {"source": source}
        })

    logger.debug(f"Retrieved {len(neighbors)} parent neighbors (async)")
    return neighbors

