# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
from pathlib import Path
from tempfile import NamedTemporaryFile

from swmmanywhere.utils import logger


def test_logger():
    """Test the get_logger function."""
    assert logger is not None
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    with NamedTemporaryFile(suffix='.log',
                            mode = 'w+b',
                            delete=False) as temp_file:
        fid = Path(temp_file.name)
        logger.add(fid)
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")
        assert temp_file.read() != b""
        logger.remove()
    fid.unlink()