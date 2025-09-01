from sqlalchemy import Column, Integer, String, JSON, Text,  DateTime, func, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Define models (schema) for tables

# Table to save file names and path
class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)   # "contract_123.pdf"
    file_metadata = Column(JSON, nullable=False)              # {"path": "..."}
    created_date = Column(DateTime, server_default=func.now())
    created_by = Column(String, nullable=True)    # e.g. "sana", "system", or a user id
    qdrant_collection = Column(String, nullable=True)
    # deleted_at = Column(DateTime, nullable=True)    # Soft delete column
    
    # One-to-many relationship between file and docstore
    docstore_entries = relationship(
        "Docstore",
        back_populates="file",
        cascade="all, delete-orphan"   # ensures deleting a File deletes its Docstore rows
    )

    def __repr__(self):
        return f"<File id={self.id} name={self.name!r}>"

# Table to store parent chunks
class Docstore(Base):
    __tablename__ = "docstore"
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    source = Column(String, nullable=False)
    parent_id = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, nullable=True)
    created_date = Column(DateTime, server_default=func.now())

    # Back reference to File
    file = relationship("File", back_populates="docstore_entries")

    __table_args__ = (
        # Index to have fast lookups by (file_id, parent_id)
        Index("ix_docstore_file_parent", "parent_id", "file_id"),
    )
    def __repr__(self):
        return f"<Docstore id={self.id}, parent_id={self.parent_id}, source={self.source}>"

