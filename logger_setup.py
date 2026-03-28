"""
logger_setup.py
统一日志初始化：控制台 + 文件轮转。
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from config import (
    LOG_BACKUP_COUNT,
    LOG_DIR,
    LOG_LEVEL,
    LOG_MAX_BYTES,
    LOG_TO_CONSOLE,
)


def get_logger(name: str, log_file: str) -> logging.Logger:
    """
    获取模块日志器。
    - 自动创建日志目录
    - 自动启用文件轮转
    - 可选控制台输出
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    os.makedirs(LOG_DIR, exist_ok=True)

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path = os.path.join(LOG_DIR, log_file)
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
