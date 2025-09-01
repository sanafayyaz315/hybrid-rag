from minio import Minio
from fastapi import File, UploadFile
import os
import io
import logging
# from builder import build_minio_client
from src.config import MINIO_BUCKET

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# minio_client = build_minio_client()

def upload_file_to_minio(minio_client: Minio, file: UploadFile, minio_bucket:str=MINIO_BUCKET) -> dict:
    """
    Uploads a FastAPI UploadFile to MinIO.
    Returns metadata dict with bucket and object path.
    """
    content = file.read()  # read bytes
    minio_client.put_object(
        bucket_name=minio_bucket,
        object_name=file.filename,
        data=io.BytesIO(content),  # reset stream
        length=len(content),
    )
    return {"bucket": MINIO_BUCKET, "object_name": file.filename, "minio_path": MINIO_BUCKET + "/" + file.filename}


async def async_upload_file_to_minio(minio_client: Minio, file: UploadFile, minio_bucket: str=MINIO_BUCKET) -> dict:
    """
    Uploads a FastAPI UploadFile to MinIO.
    Returns metadata dict with bucket and object path.
    """
    content = await file.read()  # read bytes
    minio_client.put_object(
        bucket_name=minio_bucket,
        object_name=file.filename,
        data=io.BytesIO(content),  # reset stream
        length=len(content),
    )
    return {"bucket": MINIO_BUCKET, "object_name": file.filename, "minio_path": MINIO_BUCKET + "/" + file.filename}

def download_file(minio_client: Minio, object_name: str, local_dir: str, minio_bucket: str=MINIO_BUCKET) -> str:
    """
    Downloads a file from MinIO to local_dir.
    Returns the local path.
    """
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, object_name)

    response = minio_client.get_object(minio_bucket, object_name)
    with open(local_path, "wb") as f:
        for chunk in response.stream(32*1024):
            f.write(chunk)

    return local_path

async def async_download_file(minio_client: Minio, object_name: str, local_dir: str, minio_bucket: str=MINIO_BUCKET) -> str:
    """
    Downloads a file from MinIO to local_dir.
    Returns the local path.
    """
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, object_name)

    response = minio_client.get_object(minio_bucket, object_name)
    with open(local_path, "wb") as f:
        for chunk in response.stream(32*1024):
            f.write(chunk)

    return local_path


