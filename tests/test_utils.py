# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
from pathlib import Path
from tempfile import NamedTemporaryFile

from swmmanywhere.utils import logger


def test_logger():
    """Test logger."""
    logger.enable('swmmanywhere')
    assert logger is not None
    logger.test_logger()
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
        logger.test_logger()
        assert temp_file.read() != b""
        logger.remove()
    fid.unlink()

def test_logger_disable():    
    """Test the disable function."""
    with NamedTemporaryFile(suffix='.log',
                            mode = 'w+b',
                            delete=False) as temp_file:
        fid = Path(temp_file.name)
        logger.disable('swmmanywhere')
        logger.add(fid)
        logger.test_logger()
        assert temp_file.read() == b""
        logger.remove()
    fid.unlink()

def test_logger_reimport():
    """Reimport logger to check that changes from disable are persistent."""
    from swmmanywhere.utils import logger
    with NamedTemporaryFile(suffix='.log',
                            mode = 'w+b',
                            delete=False) as temp_file:    
        fid = Path(temp_file.name)
        logger.add(fid)
        logger.test_logger()
        assert temp_file.read() == b""
        logger.remove()
    fid.unlink()