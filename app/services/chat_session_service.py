from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat_session import ChatSession

TITLE_MAX_LENGTH = 60


def _title_from_message(message: str) -> str:
    message = " ".join(message.split())
    if len(message) <= TITLE_MAX_LENGTH:
        return message
    return message[:TITLE_MAX_LENGTH].rstrip() + "..."


def touch_session(db: Session, session_id: str, first_message: str | None = None) -> ChatSession:
    """Create the session row if it's new, otherwise bump its message
    count. Called once per /chat request - `updated_at` refreshes via
    the column's onupdate=func.now() whenever this row changes.

    `title` is set once, from the first message of the session (like
    ChatGPT's sidebar) - it's never overwritten on later turns.
    """
    session = db.get(ChatSession, session_id)
    if session is None:
        title = _title_from_message(first_message) if first_message else None
        session = ChatSession(id=session_id, title=title, message_count=1)
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

def delete_session(db: Session, session_id: str) -> None:
    db.delete(session_id);
