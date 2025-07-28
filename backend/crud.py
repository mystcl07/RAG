# crud 
# Provides database operations to save user/assistant messages and retrieve recent conversation history for a specific user.

from sqlalchemy.orm import Session
from backend.models import Conversation

def save_message(db: Session, user_id: str, role: str, content: str):
    message = Conversation(user_id=user_id, role=role, content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message

def get_conversations(db: Session, user_id: str, limit: int = 10):
    return db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.id.desc()).limit(limit).all()