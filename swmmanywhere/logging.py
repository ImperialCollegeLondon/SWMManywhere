# -*- coding: utf-8 -*-
"""Created on 2024-03-04.

@author: Barney
"""
import os
import sys

import loguru


def dynamic_filter(record):
    """A dynamic filter."""
    if os.getenv("SWMMANYWHERE_VERBOSE", "false").lower() == "true":
        return True
    return False

def get_logger() -> loguru.logger:
    """Get a logger."""
    logger = loguru.logger
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "filter" : dynamic_filter,
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
    return logger

# Get the logger
logger = get_logger()

# Add a test_logger method to the logger
logger.test_logger = lambda : logger.info("This is a test message.")

# Store the original add method
original_add = logger.add

# Define a new function that wraps the original add method
def new_add(sink, **kwargs):
    """A new add method to wrap existing one but with the filter."""
    # Include the dynamic filter in the kwargs if not already specified
    if 'filter' not in kwargs:
        kwargs['filter'] = dynamic_filter
    # Call the original add method with the updated kwargs
    return original_add(sink, **kwargs)

# Replace the logger's add method with new_add
logger.add = new_add