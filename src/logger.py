"""
Configures a module-level logger with stream output and custom formatting.

This module sets up a logger using Python's standard `logging` library.
The logging level is determined by the `LOG_LEVEL` configuration. It supports
levels: DEBUG, INFO, WARNING, ERROR, and CRITICAL. A stream handler is
attached to the logger with a formatter that outputs timestamp, module name,
log level, and message. Duplicate handlers are avoided if the module is
imported multiple times.

Attributes:
    log_levels (dict): Maps string representations of log levels to `logging` constants.
    log_level (int): Logging level selected based on `LOG_LEVEL` from config.
    logger (logging.Logger): Configured logger instance for this module.
    stream_handler (logging.StreamHandler): Stream handler for console output.
    formatter (logging.Formatter): Formatter defining log message format.
"""

import logging
from src.config import LOG_LEVEL

# Map config string to logging level
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

log_level = log_levels.get(LOG_LEVEL.upper(), logging.INFO)

# Create logger
logger = logging.getLogger(__name__)   # safer than root logger
logger.setLevel(log_level)

# Create handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler.setFormatter(formatter)

# Avoid duplicate handlers if re-imported
if not logger.handlers:
    logger.addHandler(stream_handler)
