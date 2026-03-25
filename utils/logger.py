"""
utils/logger.py -- Structured logging with file rotation.

Uses a SINGLE shared RotatingFileHandler across all modules to prevent
Windows PermissionError when rotating log files.  On Windows, multiple
file handles to the same file block os.rename().  By attaching only ONE
file handler (via the root logger), rotation works cleanly.
"""

import io
import sys
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "omnisignal.log")
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
BACKUP_COUNT = 3

os.makedirs(LOG_DIR, exist_ok=True)

_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_initialized = False


def _safe_console_stream():
    """Wrap stdout in a UTF-8 stream so emoji/unicode log messages don't crash
    on Windows consoles that use non-UTF-8 codepages (e.g. cp1255, cp1252)."""
    try:
        return io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        return sys.stdout


def _init_root_handlers():
    """Attach console + file handlers to the ROOT logger exactly once.
    All child loggers created via get_logger() inherit these handlers."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=_safe_console_stream())
    ch.setLevel(logging.INFO)
    ch.setFormatter(_formatter)
    root.addHandler(ch)

    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_formatter)
    root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Handlers are on the root logger (shared)."""
    _init_root_handlers()
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger


_trade_logger = None


def get_trade_logger() -> logging.Logger:
    """Return a dedicated trade-event logger writing ONLY to logs/trades.log."""
    global _trade_logger
    if _trade_logger is not None:
        return _trade_logger
    _init_root_handlers()
    tl = logging.getLogger("omnisignal.trades")
    tl.propagate = False
    tl.setLevel(logging.INFO)
    trade_fmt = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    tfh = RotatingFileHandler(
        os.path.join(LOG_DIR, "trades.log"),
        maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    tfh.setLevel(logging.INFO)
    tfh.setFormatter(trade_fmt)
    tl.addHandler(tfh)
    _trade_logger = tl
    return tl

