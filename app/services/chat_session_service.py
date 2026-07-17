from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat_session import ChatSession


def touch_session(db: Session, session_id: str) -> ChatSession:
    """Create the session row if it's new, otherwise bump its message
    count. Called once per /chat request - `updated_at` refreshes via
    the column's onupdate=func.now() whenever this row changes.
    """
    session = db.get(ChatSession, session_id)
    if session is None:
        session = ChatSession(id=session_id, message_count=1)
        db.add(session)
    else:
        session.message_count += 1
    db.commit()
    db.refresh(session)
    return session


def list_sessions(db: Session) -> list[ChatSession]:
    return list(db.execute(select(ChatSession).order_by(ChatSession.updated_at.desc())).scalars())


def get_session(db: Session, session_id: str) -> ChatSession | None:
    return db.get(ChatSession, session_id)
