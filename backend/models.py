# models
# Defines the SQLAlchemy Conversation table for storing messages with user ID, role, and content, 
# and sets up the database engine and session.
from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config import DATABASE_URL

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # For multi-user support
    role = Column(String)
    content = Column(Text)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)