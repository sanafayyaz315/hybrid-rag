from sqlalchemy.ext.declarative import declarative_base
from src.docstore.session import sync_engine

# Shared Base for all models to inherit from
Base = declarative_base()

def init_db():
    Base.metadata.create_all(bind=sync_engine)
