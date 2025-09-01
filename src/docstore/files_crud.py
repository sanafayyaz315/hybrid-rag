from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from fastapi import UploadFile, HTTPException
import logging
from src.docstore.models import File
from typing import List, Dict, Any, Optional, Union
from src.config import COLLECTION


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# func to check if a file name already exists in the files table
def ensure_unique_filenames(session: Session, file: UploadFile) -> None:
    """
    Ensure that the uploaded file has a unique filename in the database.

    Checks the `files` table for an existing row with the same filename. 
    If a file with the same name exists, raises an HTTPException to prevent overwriting.

    Args:
        session (Session): SQLAlchemy session used to query the database.
        file (UploadFile): FastAPI UploadFile object representing the file to check.

    Raises:
        HTTPException: If no file is provided or if a file with the same name already exists.
    """
    file_name =  file.filename
    if not file_name:
        logger.debug("No file provided.")
        raise HTTPException(status_code=400, detail="No file provided.")
       
    # check if file already exists
    existing_file = session.query(File).filter(File.name==file_name).first()
    if existing_file:
        logger.debug(f"File {file_name} already exists in DB")
        raise HTTPException(
            status_code=400,
            detail=f"File {file_name} already exists. Delete it first before re-uploading."
            )   
    logger.debug(f"File {file_name} is unique and safe to insert.")

async def async_ensure_unique_filenames(session: AsyncSession, file: UploadFile) -> None:
    """
    Ensure that the uploaded file has a unique filename in the database. Uses AsyncSession.

    Checks the `files` table for an existing row with the same filename. 
    If a file with the same name exists, raises an HTTPException to prevent overwriting.

    Args:
        session (Session): SQLAlchemy session used to query the database.
        file (UploadFile): FastAPI UploadFile object representing the file to check.

    Raises:
        HTTPException: If no file is provided or if a file with the same name already exists.
    """
    file_name =  file.filename
    if not file_name:
        logger.debug("No file provided.")
        raise HTTPException(status_code=400, detail="No file provided.")
    
    # check if file already exists
    result = await session.execute(select(File).where(File.name==file_name))
    existing_file = result.scalar_one_or_none()
    if existing_file:
        logger.debug(f"File {file_name} already exists in DB")
        raise HTTPException(
            status_code=400,
            detail=f"File {file_name} already exists. Delete it first before re-uploading."
            ) 
    else:
        logger.debug(f"File {file_name} is unique and safe to insert.")

# stage file names in files table
def stage_file_rows(session: Session, minio_metadata: Dict[str, Any]) -> File:
    """Add File row and flush to get ID. Not visible until commit."""
    try:
        # create a row for the new_file
        new_file = File(
            name=minio_metadata["object_name"],
            file_metadata=minio_metadata,
            qdrant_collection=COLLECTION
        )
        session.add(new_file)
        session.flush()
        logger.debug(f"Staged file {new_file.name} in DB (id: {new_file.id}).")
        return new_file
    
    except Exception as e:
        logger.exception(f"Failed to stage file {minio_metadata.get('object_name')}")
        raise RuntimeError(
            f"Failed to add a row for {minio_metadata.get('object_name')} in table files: {str(e)}"
        )

async def async_stage_file_rows(session: AsyncSession, minio_metadata: Dict[str, Any]) -> File:
    """Add File row and flush to get ID. Not visible until commit. Using Async session"""
    try:
        # create a row for the new_file
        new_file = File(
            name=minio_metadata["object_name"],
            file_metadata=minio_metadata,
            qdrant_collection=COLLECTION
        )
        session.add(new_file)
        await session.flush()
        logger.debug(f"Staged file {new_file.name} in DB (id: {new_file.id}).")
        return new_file

    except Exception as e:
        logger.exception(f"Failed to stage file {minio_metadata.get('object_name')}")
        raise RuntimeError(
            f"Failed to add a row for {minio_metadata.get('object_name')} in table files: {str(e)}"
        )

def commit_session(session: Session, file: File) -> File:
    """
    Commit all staged changes in the current DB session.
    Includes staged file row(s), parent chunks in docstore, and any other pending objects.
    Ensures rollback on failure and returns the committed File object for reference.
    """
    try:
        session.commit()
        logger.debug(f"Committed pipeline for file: {file.name}, id: {file.id} to the docstore database")
        return file
    
    except SQLAlchemyError as e:
        session.rollback()
        logger.exception(f"Failed to commit pipeline file: {getattr(file, 'name', None)} to the docstore database")
        raise RuntimeError(f"DB commit failed for file {getattr(file, 'name', None)}: {str(e)}")

async def async_commit_session(session: AsyncSession, file: File) -> File:
    """
    Commit all staged changes in the current DB session. Uses AsyncSession.
    Includes staged file row(s), parent chunks in docstore, and any other pending objects.
    Ensures rollback on failure and returns the committed File object for reference.
    """
    try:
        await session.commit()
        logger.debug(f"Committed pipeline for file: {file.name}, id: {file.id} to the docstore database")
        return file
    except SQLAlchemyError as e:
        await session.rollback()
        logger.exception(f"Failed to commit pipeline for file: {getattr(file, 'name', None)} to the docstore database")
        raise RuntimeError(f"DB commit failed for file {getattr(file, 'name', None)}: {str(e)}")

def delete_file_row(filename: str, session: Session, commit: bool = True) -> bool:
    """
    Delete a file row and all related docstore entries (via cascade).
    
    Args:
        filename (str): Name of the file to delete
        session (Session): SQLAlchemy session
        commit (bool): Whether to commit immediately

    Returns:
        bool: True if deleted, False if file not found
    """
    # Fetch the file row
    file = session.query(File).filter_by(name=filename).first()

    if not file:
        logger.debug(f"File {filename} not found in docstore.")
        return False
    logger.debug(f"Fetched file for deletion: {file}")

    session.delete(file)
    if commit:
        session.commit()
    logger.debug(f"File {filename} and related docstore rows deleted.")
    return True   

async def async_delete_file_row(filename: str, session: AsyncSession, commit: bool = True) -> bool:
    """
    Delete a file row and all related docstore entries (via cascade). Uses AsyncSession
    
    Args:
        filename (str): Name of the file to delete
        session (AsyncSession): SQLAlchemy session
        commit (bool): Whether to commit immediately

    Returns:
        bool: True if deleted, False if file not found
    """
    # Fetch the file row
    result = await session.execute(select(File).where(File.name == filename))
    file = result.scalar_one_or_none()  # gets the single object or None      
    if not file:
        logger.debug(f"File {filename} not found in docstore.")
        return False
    
    logger.debug(f"Fetched file for deletion: {file}")

    await session.delete(file)      # schedule deletion
    if commit:
        await session.commit()   # commit to DB  
    logger.debug(f"File {filename} and related docstore rows deleted (async).")
    return True
    
def list_files(session: Session) -> List[str]:
    """
    List all files in the given collection.
    Collection is defined at the start time of the application.

    Args:
        session (Session): SQLAlchemy session

    Returns:
        List[str]: List of file names
    """
    try:
        files = session.query(File).filter(File.qdrant_collection==COLLECTION).all()
        if not files:
            logger.debug(f"No files found in {COLLECTION} collection")
            return []
        
        filenames = [f.name for f in files]
        logger.debug(f"Found {len(filenames)} files in {COLLECTION} collection")
        return filenames
    
    except SQLAlchemyError as e:
        logger.exception(f"Could not list files from docstore for {COLLECTION} collection")
        raise RuntimeError(f"Failed to list files in {COLLECTION} collection: {e}")

async def async_list_files(session: AsyncSession) -> List[str]:
    """
    List all files in the given collection. Uses AsyncSession
    Collection is defined at the start time of the application.

    Args:
        session (AsyncSession): SQLAlchemy session

    Returns:
        List[str]: List of file names
    """
    try:
        results = await session.execute(select(File).where(File.qdrant_collection==COLLECTION))
        files = results.scalars().all()
        if not files:
            logger.debug(f"No files found in {COLLECTION} collection")
            return []
        filenames = [f.name for f in files]
        logger.debug(f"Found {len(filenames)} files in {COLLECTION} collection")
        return filenames
    
    except SQLAlchemyError as e:
        logger.exception(f"Could not list files from docstore for {COLLECTION} collection")
        raise RuntimeError(f"Failed to list files in {COLLECTION} collection: {e}")

def list_all_files(session: Session) -> List[str]:
    """
    List all files in the docstore.

    Args:
        session (Session): SQLAlchemy session

    Returns:
        List[str]: List of file names
    """
    try:
        files = session.query(File).all()
        if not files:
            logger.debug(f"No files found in docstore DB")
            return []
        
        filenames = [f.name for f in files]
        logger.debug(f"Found {len(filenames)} files in docstore DB")
        return filenames
    
    except SQLAlchemyError as e:
        logger.exception(f"Could not list files from docstore DB")
        raise RuntimeError(f"Failed to list files from docstore DB:  {e}")

async def async_list_all_files(session: AsyncSession) -> List[str]:
    """
    List all files in the docstore. Uses AsyncSession

    Args:
        session (AsyncSession): SQLAlchemy session

    Returns:
        List[str]: List of file names
    """
    try:
        results = await session.execute(select(File))
        files = results.scalars().all()
        if not files:
            logger.debug(f"No files found in docstore DB")
            return []
        filenames = [f.name for f in files]
        logger.debug(f"Found {len(filenames)} files in docstore DB")
        return filenames
    
    except SQLAlchemyError as e:
        logger.exception(f"Could not list files from docstore DB")
        raise RuntimeError(f"Failed to list files from docstore DB: {e}")

def get_file(filename: str, session: Session) -> Optional[File]:
    """
    Fetch a single file row from the files table by filename.

    Args:
        filename (str): Name of the file to fetch.
        session (Session): SQLAlchemy session.

    Returns:
        Optional[File]: File object if found, otherwise None.
    """

    try:
        file = session.query(File).filter(File.name == filename).first()
        if not file:
            logger.debug(f"No file named {filename} found")
            return None

        logger.debug(f"File {filename} found")
        return file
    
    except SQLAlchemyError as e:
        logger.debug(f"Could not fetch file {filename}")
        raise RuntimeError(f"Could not fetch file {filename}: {e}")

async def async_get_file(filename: str, session: AsyncSession) -> Optional[File]:
    """
    Fetch a single file row from the files table by filename. Uses AsyncSession

    Args:
        filename (str): Name of the file to fetch.
        session (Session): SQLAlchemy session.

    Returns:
        Optional[File]: File object if found, otherwise None.
    """

    try:
        result = await session.execute(select(File).where(File.name == filename))
        file = result.scalar_one_or_none()
        if not file:
            logger.debug(f"No file named {filename} found")
            return None

        logger.debug(f"File {filename} found")
        return file
    
    except SQLAlchemyError as e:
        logger.debug(f"Could not fetch file {filename}")
        raise RuntimeError(f"Could not fetch file {filename}: {e}")


        


        
        
        