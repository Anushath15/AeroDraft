import sys
from pathlib import Path
from loguru import logger

def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    logger.remove()
    
    logger.add(
        sys.stderr, level=level, colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>"
    )
    
    logger.add(
        log_path / "aerodraft_{time:YYYY-MM-DD}.log", level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        rotation="1 day", retention="7 days", compression="gz"
    )
    
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log", level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}\n{exception}",
        rotation="1 day", retention="30 days"
    )
