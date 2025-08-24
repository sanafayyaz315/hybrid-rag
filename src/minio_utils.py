from minio import Minio
from fastapi import File, UploadFile
import os
import io
import logging
from config import (
    MINIO_ENDPOINT,
    MINIO_BUCKET,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY
)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create MinIO client
minio_client = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Create bucket if does not exist
if not minio_client.bucket_exists(MINIO_BUCKET):
    minio_client.make_bucket(MINIO_BUCKET)


async def upload_file_to_minio(file: UploadFile) -> dict:
    """
    Uploads a FastAPI UploadFile to MinIO.
    Returns metadata dict with bucket and object path.
    """
    content = await file.read()  # read bytes
    minio_client.put_object(
        bucket_name=MINIO_BUCKET,
        object_name=file.filename,
        data=io.BytesIO(content),  # reset stream
        length=len(content),
    )
    return {"bucket": MINIO_BUCKET, "object_name": file.filename, "minio_path": MINIO_BUCKET + "/" + file.filename}

def download_file(object_name: str, local_dir: str) -> str:
    """
    Downloads a file from MinIO to local_dir.
    Returns the local path.
    """
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, object_name)

    response = minio_client.get_object(MINIO_BUCKET, object_name)
    with open(local_path, "wb") as f:
        for chunk in response.stream(32*1024):
            f.write(chunk)

    return local_path



