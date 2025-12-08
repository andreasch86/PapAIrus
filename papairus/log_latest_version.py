# papairus/log.py
import inspect
import logging
import sys

from loguru import logger

logger = logger.opt(colors=True)
"""PapAIrus logging helpers configured for loguru and stdlib logging."""


class InterceptHandler(logging.Handler):
    """
    InterceptHandler is a custom logging handler that integrates the Loguru logger with the standard logging module. It intercepts log records from the standard logging module and logs them using the Loguru logger.

    Args:
        None

    Returns:
        None
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        """
        emit is a method of the InterceptHandler class that handles log records. It converts the log record's level to a Loguru level name and finds the caller from where the log message originated. It then logs the message using the Loguru logger with the appropriate depth and exception information.

        Args:
            record (logging.LogRecord): The log record to handle.

        Returns:
            None
        """
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def set_logger_level_from_config(log_level):
    """
    set_logger_level_from_config configures the Loguru logger with the specified log level and integrates it with the standard logging module.

    Args:
        log_level (str): The log level to set for Loguru (e.g., "DEBUG", "INFO", "WARNING").

    Returns:
        None

    This function:
        - Removes any existing Loguru handlers to ensure a clean slate.
        - Adds a new handler to Loguru, directing output to stderr with the specified level.
          - `enqueue=True` ensures thread-safe logging by using a queue, helpful in multi-threaded contexts.
          - `backtrace=False` minimizes detailed traceback to prevent overly verbose output.
          - `diagnose=False` suppresses additional Loguru diagnostic information for more concise logs.
        - Redirects the standard logging output to Loguru using the InterceptHandler, allowing Loguru to handle
              all logs consistently across the application.
    """
    logger.remove()
    # Use synchronous logging to avoid background queue threads writing to closed
    # streams during test teardown or CLI exit in constrained environments.
    logger.add(sys.stderr, level=log_level, enqueue=False, backtrace=False, diagnose=False)

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    logger.success(f"Log level set to {log_level}!")
