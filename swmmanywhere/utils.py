# -*- coding: utf-8 -*-
"""Created on 2024-03-04.

@author: Barney
"""
import sys

import loguru


def get_logger() -> loguru.logger:
    """Get a logger."""
    logger = loguru.logger
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "colorize": True,
                "format": " | ".join(
                    [
                        "<cyan>{time:YYYY/MM/DD HH:mm:ss}</>",
                        "{message}",
                    ]
                ),
            }
        ]
    )
    # Disable by default
    logger.disable("swmmanywhere")
    return logger

logger = get_logger()
logger.test_logger = lambda : logger.info("This is a test message.")