from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from models import Base
from config import (
    DOCSTORE_HOST,
    DOCSTORE_PORT,
    DOCSTORE_USER,
    DOCSTORE_PASSWORD,
    DOCSTORE_NAME
)

connection_string = f"postgresql+psycopg2://{DOCSTORE_USER}:{DOCSTORE_PASSWORD}@{DOCSTORE_HOST}:{DOCSTORE_PORT}/{DOCSTORE_NAME}"
POSTGRES_URL = f"{DOCSTORE_USER}:{DOCSTORE_PASSWORD}@{DOCSTORE_HOST}:{DOCSTORE_PORT}/{DOCSTORE_NAME}"

sync_conninfo = f"postgresql+psycopg2://{POSTGRES_URL}"
async_conninfo = f"postgresql+asyncpg://{POSTGRES_URL}"

pool_size = 10
pool_timeout = 30
max_overflow = 5
pool_recycle = 1800
pool_pre_ping = True
echo = False

# 1. Create the engine
sync_engine = create_engine(
    sync_conninfo,
    pool_size=pool_size,
    pool_timeout=pool_timeout,
    max_overflow=max_overflow,
    pool_recycle=pool_recycle,
    pool_pre_ping=pool_pre_ping,
    echo=echo,
)

async_engine = create_async_engine(
    async_conninfo,
    pool_size=pool_size,
    pool_timeout=pool_timeout,
    max_overflow=max_overflow,
    pool_recycle=pool_recycle,
    pool_pre_ping=pool_pre_ping,
    echo=echo,
)

# engine = create_engine(connection_string, echo=True)  # echo=True logs SQL queries

# 2. Create a configured "Session" class. This creates a session factory
SessionLocal = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 3. Initialize tables (idempotent)
Base.metadata.create_all(bind=sync_engine) # create_all is idempotent
