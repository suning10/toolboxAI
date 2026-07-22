"""
Central logging setup. print() output is unreliable here: it can be
fully buffered (invisible until the process exits) once stdout isn't a
TTY - the common case under uvicorn --reload, systemd, Docker, or a
`> file.log` redirect. logging always goes through a configured
handler and flushes per record, so use `logging.getLogger(__name__)`
in your module instead of print().

configure_logging() must run once, before anything logs - it's called
at the top of app/main.py, which every request path imports first.
"""
import logging

from app.config import LOG_LEVEL


def configure_logging() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
