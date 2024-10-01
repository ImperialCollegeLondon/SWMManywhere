# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from swmmanywhere.logging import logger, verbose


def test_logger():
    """Test logger."""
    os.environ["SWMMANYWHERE_VERBOSE"] = "true"
    assert logger is not None
    logger.test_logger()
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    with NamedTemporaryFile(suffix=".log", mode="w+b", delete=False) as temp_file:
        fid = Path(temp_file.name)
        logger.add(fid)
        logger.test_logger()
        assert temp_file.read() != b""
        logger.remove()
    fid.unlink()
    os.environ["SWMMANYWHERE_VERBOSE"] = "false"


def test_logger_disable():
    """Test the disable function."""
    with NamedTemporaryFile(suffix=".log", mode="w+b", delete=False) as temp_file:
        fid = Path(temp_file.name)
        os.environ["SWMMANYWHERE_VERBOSE"] = "false"
        logger.add(fid)
        logger.test_logger()
        assert temp_file.read() == b""
        logger.remove()
    fid.unlink()


def test_logger_reimport():
    """Reimport logger to check that changes from disable are persistent."""
    from swmmanywhere.logging import logger

    with NamedTemporaryFile(suffix=".log", mode="w+b", delete=False) as temp_file:
        fid = Path(temp_file.name)
        logger.add(fid)
        logger.test_logger()
        assert temp_file.read() == b""
        logger.remove()
    fid.unlink()


def test_logger_again():
    """Test the logger after these changes to make sure still works."""
    os.environ["SWMMANYWHERE_VERBOSE"] = "true"
    with NamedTemporaryFile(suffix=".log", mode="w+b", delete=False) as temp_file:
        fid = Path(temp_file.name)
        logger.add(fid)
        logger.test_logger()
        assert temp_file.read() != b""
        logger.remove()
    fid.unlink()
    os.environ["SWMMANYWHERE_VERBOSE"] = "false"


def test_verbose():
    """Test the verbose function."""
    os.environ["SWMMANYWHERE_VERBOSE"] = "true"
    assert verbose()

    os.environ["SWMMANYWHERE_VERBOSE"] = "false"
    assert not verbose()

    del os.environ["SWMMANYWHERE_VERBOSE"]
    assert not verbose()
