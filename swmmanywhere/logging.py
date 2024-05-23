"""Logging module for SWMManywhere.

Example:
>>> import os
>>> os.environ["SWMMANYWHERE_VERBOSE"] = "true" 
>>> # logging is now enabled in any swmmanywhere module
>>> from swmmanywhere.logging import logger # You can now log yourself
>>> logger.info("This is an info message.") # Write to stdout # doctest: +SKIP
This is an info message.
>>> logger.add("file.log") # Add a log file # doctest: +SKIP
>>> os.environ["SWMMANYWHERE_VERBOSE"] = "false" # Disable logging
"""
from __future__ import annotations

import os
import sys

import loguru
from tqdm import tqdm as tqdm_original


def tqdm(*args, **kwargs):
    """Custom tqdm function.
    
    A custom tqdm function that checks for the verbosity. If verbose, it
    returns the actual tqdm progress bar. Otherwise, it returns a simple 
    iterator without the progress bar.
    """
    verbose = os.getenv("SWMMANYWHERE_VERBOSE", "false").lower() == "true"

    if verbose:
        return tqdm_original(*args, **kwargs)
    else:
        iterator = args[0]
        return iterator
    
def dynamic_filter(record):
    """A dynamic filter."""
    return os.getenv("SWMMANYWHERE_VERBOSE", "false").lower() == "true"

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