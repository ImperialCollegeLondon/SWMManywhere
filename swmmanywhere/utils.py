# -*- coding: utf-8 -*-
"""Created on 2024-03-04.

@author: Barney
"""
import os
import sys

import loguru


def get_logger(verbose: bool = False) -> loguru.logger:
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
    if os.getenv("PACKAGE_NAME_VERBOSE", str(verbose)).lower() == "true":
        logger.enable("package_name")
    else:
        logger.disable("package_name")
    return logger

logger = get_logger()