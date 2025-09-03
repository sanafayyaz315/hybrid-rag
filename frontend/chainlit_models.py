import uuid
from sqlalchemy import Column, Text, Boolean, Integer, ForeignKey, JSON, text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from src.db_setup import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    identifier = Column(Text, nullable=False, unique=True)
    _metadata = Column("metadata", JSON, nullable=True, key="metadata")  
    created_at = Column("createdAt", Text, nullable=True)  # exact column name
    threads = relationship("Thread", back_populates="user", cascade="all, delete-orphan")


class Thread(Base):
    __tablename__ = "threads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column("createdAt", Text, nullable=True)
    name = Column("name", Text, nullable=True)
    user_id = Column("userId", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    user_identifier = Column("userIdentifier", Text, nullable=True)
    tags = Column("tags", ARRAY(Text), nullable=True)
    _metadata = Column("metadata", JSON, nullable=True, key="metadata")  # <-- fix here


    user = relationship("User", back_populates="threads")
    elements = relationship("Element", back_populates="thread", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="thread", cascade="all, delete-orphan")
    steps = relationship("Step", back_populates="thread", cascade="all, delete-orphan")


class Element(Base):
    __tablename__ = "elements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column("threadId", UUID(as_uuid=True), ForeignKey("threads.id", ondelete="CASCADE"), nullable=True)
    type = Column("type", Text, nullable=True)
    url = Column("url", Text, nullable=True)
    chainlit_key = Column("chainlitKey", Text, nullable=True)
    name = Column("name", Text, nullable=False)
    display = Column("display", Text, nullable=True)
    object_key = Column("objectKey", Text, nullable=True)
    size = Column("size", Text, nullable=True)
    page = Column("page", Integer, nullable=True)
    language = Column("language", Text, nullable=True)
    for_id = Column("forId", UUID(as_uuid=True), nullable=True)
    mime = Column("mime", Text, nullable=True)
    props = Column("props", JSON, nullable=True)

    thread = relationship("Thread", back_populates="elements")


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    for_id = Column("forId", UUID(as_uuid=True), nullable=False)
    thread_id = Column("threadId", UUID(as_uuid=True), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
    value = Column("value", Integer, nullable=False)
    comment = Column("comment", Text, nullable=True)

    thread = relationship("Thread", back_populates="feedbacks")


class Step(Base):
    __tablename__ = "steps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column("name", Text, nullable=False)
    type = Column("type", Text, nullable=False)
    thread_id = Column("threadId", UUID(as_uuid=True), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
    parent_id = Column("parentId", UUID(as_uuid=True), nullable=True)
    streaming = Column("streaming", Boolean, nullable=False)
    wait_for_answer = Column("waitForAnswer", Boolean, nullable=True)
    is_error = Column("isError", Boolean, nullable=True)
    _metadata = Column("metadata", JSON, nullable=True, key="metadata")  # <-- fix here
    tags = Column("tags", ARRAY(Text), nullable=True)
    input = Column("input", Text, nullable=True)
    output = Column("output", Text, nullable=True)
    created_at = Column("createdAt", Text, nullable=True)
    start = Column("start", Text, nullable=True)
    end = Column("end", Text, nullable=True)
    generation = Column("generation", JSON, nullable=True)
    show_input = Column("showInput", Text, nullable=True)
    language = Column("language", Text, nullable=True)
    indent = Column("indent", Integer, nullable=True)
    default_open = Column("defaultOpen", Boolean, nullable=False, server_default=text("false"))

    thread = relationship("Thread", back_populates="steps")
