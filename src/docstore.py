from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
from config import (
    DOCSTORE_HOST,
    DOCSTORE_PORT,
    DOCSTORE_USER,
    DOCSTORE_PASSWORD,
    DOCSTORE_NAME
)

connection_string = f"postgresql+psycopg2://{DOCSTORE_USER}:{DOCSTORE_PASSWORD}@{DOCSTORE_HOST}:{DOCSTORE_PORT}/{DOCSTORE_NAME}"

# 1. Create the engine
engine = create_engine(connection_string, echo=True)  # echo=True logs SQL queries

# 2. Create a configured "Session" class. This creates a session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# 3. Initialize tables (idempotent)
Base.metadata.create_all(bind=engine) # create_all is idempotent
