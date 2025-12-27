"""
Module for handling logging functionality.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any
import json
from pathlib import Path
from .paths import SESSION_LOGS_DIR

class SafeFileHandler(logging.FileHandler):
    """FileHandler that never raises (prevents 'I/O operation on closed file' from crashing runs)."""
    def emit(self, record):
        try:
            super().emit(record)
        except (OSError, IOError, ValueError):
            # ValueError can be raised when the underlying stream is closed.
            return

class SessionLogger:
    """
    Handles logging for individual optimization sessions.
    
    Attributes:
        session_id (str): ID of the session being logged
        app_logger (logging.Logger): Logger for application events
        dspy_logger (logging.Logger): Logger for DSPy-specific events
    """
    
    def __init__(self, session_id: str):
        """
        Initialize a new session logger.
        
        Args:
            session_id (str): Unique identifier for the session
        """
        self.session_id = session_id
        
        # Ensure the session logs directory exists
        SESSION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create unique log file names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        app_log_file = SESSION_LOGS_DIR / f'app_log_{timestamp}_{session_id}.log'
        dspy_log_file = SESSION_LOGS_DIR / f'dspy_log_{timestamp}_{session_id}.log'
        
        # Configure application logger
        self.app_logger = logging.getLogger(f'App_Session_{session_id}')
        self.app_logger.setLevel(logging.INFO)
        
        # Add handlers for app logger
        app_handler = SafeFileHandler(app_log_file, encoding="utf-8", delay=True)
        app_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.app_logger.addHandler(app_handler)
        
        # Configure DSPy logger
        self.dspy_logger = logging.getLogger('dspy')
        self.dspy_logger.setLevel(logging.DEBUG)
        # Prevent double-logging to root handlers (which may include fragile file handlers).
        self.dspy_logger.propagate = False
        
        # Add handler for DSPy logger
        # Guard against adding multiple handlers across sessions in the same process.
        if not getattr(self.dspy_logger, "_promptomatix_safe_filehandler_added", False):
            dspy_handler = SafeFileHandler(dspy_log_file, encoding="utf-8", delay=True)
            dspy_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.dspy_logger.addHandler(dspy_handler)
            self.dspy_logger._promptomatix_safe_filehandler_added = True
    
    def add_entry(self, entry_type: str, data: Dict[str, Any]) -> None:
        """
        Add a new log entry.
        
        Args:
            entry_type (str): Type of log entry (e.g., "ERROR", "INFO")
            data (Dict[str, Any]): Data to be logged
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "type": entry_type,
            **data
        }
        
        # Log as JSON for better parsing
        self.app_logger.info(json.dumps(log_entry))
        
        # If it's an error, also log to error level
        if entry_type == "ERROR":
            self.app_logger.error(json.dumps(log_entry)) 