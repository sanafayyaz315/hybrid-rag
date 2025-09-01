from src.docstore.models import Base
from src.docstore.session import sync_engine

def init_db():
    Base.metadata.create_all(bind=sync_engine)

