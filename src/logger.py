import logging
import sys
from pathlib import Path
from datetime import datetime
import json

class CustomJsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging (without external dependency)"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_logger(name: str, log_level: str = 'INFO') -> logging.Logger:
    """Setup logger with both console and file handlers"""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler (JSON format for production)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = CustomJsonFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (for debugging)
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / f'{name}.log')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Default logger
app_logger = setup_logger('phishing_detector')