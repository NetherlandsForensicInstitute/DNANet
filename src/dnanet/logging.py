"""Centralized logging configuration using loguru.

Design pattern: **Singleton / Single Point of Configuration**
    Instead of scattering `logging.getLogger("dnanet")` calls throughout the
    codebase (as in the original DNANet), we configure one global logger here.
    Every module simply does `from loguru import logger` and uses it directly.

    Loguru replaces the standard-library logging boilerplate:
    - No need for `getLogger`, handlers, formatters, or log levels per module.
    - Structured output with colors, timestamps, and source locations for free.
    - The `configure()` function below is called once at application startup
      (in `cli.py`) to set the desired verbosity and output file.

Usage in any module:
    >>> from loguru import logger
    >>> logger.info("Training started with lr={}", lr)
"""

import logging
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm


class _InterceptHandler(logging.Handler):
    """Forward standard-library logging records to loguru"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


_THIRD_PARTY_LOGGERS = [
    'lightning',
    'lightning.pytorch',
    'pytorch_lightning',
    'hydra',
    'omegaconf'
]


def configure(
    verbosity: str = "INFO",
    log_file: Path | None = None,
    serialize: bool = False,
) -> None:
    """Configure the global loguru logger.

    Called once at application startup. Removes the default stderr sink and
    replaces it with one that respects the requested verbosity. Optionally
    adds a file sink for persistent logs.

    Args:
        verbosity: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: If provided, also write logs to this file (with rotation).
        serialize: If True, write logs as structured JSON (useful for
            machine-parseable experiment logs).
    """
    # Remove default sink so we don't get duplicate output
    logger.remove()
    def _tqdm_sink(msg: str) -> None:
        tqdm.write(msg, end="")

    # Console sink — human-readable with colors
    logger.add(
        _tqdm_sink,
        level=verbosity.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


    # Optional file sink — rotated, with full detail
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level="DEBUG",  # file always captures everything
            rotation="10 MB",
            retention="30 days",
            serialize=serialize,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} - "
                "{message}"
            ),
        )

    _intercept_third_party_loggers(verbosity)
    logger.debug("Logging configured: verbosity={}, log_file={}", verbosity, log_file)


def _intercept_third_party_loggers(verbosity: str) -> None:
    handler = _InterceptHandler()

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(verbosity.upper())

    for name in _THIRD_PARTY_LOGGERS:
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False
        lg.setLevel(verbosity.upper())
