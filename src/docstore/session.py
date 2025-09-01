from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from src.config import (
    DOCSTORE_HOST,
    DOCSTORE_PORT,
    DOCSTORE_USER,
    DOCSTORE_PASSWORD,
    DOCSTORE_NAME
)

POSTGRES_URL = f"{DOCSTORE_USER}:{DOCSTORE_PASSWORD}@{DOCSTORE_HOST}:{DOCSTORE_PORT}/{DOCSTORE_NAME}"
sync_conninfo = f"postgresql+psycopg2://{POSTGRES_URL}"
async_conninfo = f"postgresql+asyncpg://{POSTGRES_URL}"

pool_size = 10
pool_timeout = 30
max_overflow = 5
pool_recycle = 1800
pool_pre_ping = True
echo = False

# Sync engine, session factory, and dependence injection
sync_engine = create_engine(
    sync_conninfo,
    pool_size=pool_size,
    pool_timeout=pool_timeout,
    max_overflow=max_overflow,
    pool_recycle=pool_recycle,
    pool_pre_ping=pool_pre_ping,
    echo=echo,
)
# Session Factory
SessionLocal = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)

# Sync Dependency to get DB session per request
def get_sync_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Async engine, async session factory and Dependency
async_engine = create_async_engine(
    async_conninfo,
    pool_size=pool_size,
    pool_timeout=pool_timeout,
    max_overflow=max_overflow,
    pool_recycle=pool_recycle,
    pool_pre_ping=pool_pre_ping,
    echo=echo,
)
# Session Factory
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

# Async dependency to get a session
async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session