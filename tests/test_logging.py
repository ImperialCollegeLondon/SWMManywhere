# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from tqdm import tqdm as tqdm_original

from swmmanywhere.logging import logger, tqdm


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
    with NamedTemporaryFile(suffix='.log',
                            mode = 'w+b',
                            delete=False) as temp_file:
        fid = Path(temp_file.name)
        logger.add(fid)
        logger.test_logger()
        assert temp_file.read() != b""
        logger.remove()
    fid.unlink()
    os.environ["SWMMANYWHERE_VERBOSE"] = "false"

def test_logger_disable():    
    """Test the disable function."""
    with NamedTemporaryFile(suffix='.log',
                            mode = 'w+b',
                            delete=False) as temp_file:
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
    with NamedTemporaryFile(suffix='.log',
                            mode = 'w+b',
                            delete=False) as temp_file:    
        fid = Path(temp_file.name)
        logger.add(fid)
        logger.test_logger()
        assert temp_file.read() == b""
        logger.remove()
    fid.unlink()

def test_logger_again():
    """Test the logger after these changes to make sure still works."""
    os.environ["SWMMANYWHERE_VERBOSE"] = "true"
    with NamedTemporaryFile(suffix='.log',
                            mode = 'w+b',
                            delete=False) as temp_file:    
        fid = Path(temp_file.name)
        logger.add(fid)
        logger.test_logger()
        assert temp_file.read() != b""
        logger.remove()
    fid.unlink()
    os.environ["SWMMANYWHERE_VERBOSE"] = "false"

def test_tqdm():
    """Test custom tqdm with true verbose."""
    # Set SWMMANYWHERE_VERBOSE to True
    os.environ["SWMMANYWHERE_VERBOSE"] = "true"

    # Create a mock iterator
    mock_iterator = iter(range(10))

    # Patch the original tqdm function
    with patch("swmmanywhere.logging.tqdm_original",
               wraps=tqdm_original) as mock_tqdm:
        # Call the custom tqdm function
        result = [i for i in tqdm(mock_iterator)]
        
        # Check if the original tqdm was called
        mock_tqdm.assert_called()

        # Check if the progress_bar is the same as the mocked tqdm
        assert result == list(range(10))

def test_tqdm_not_verbose():
    """Test custom tqdm with false verbose."""
    # Set SWMMANYWHERE_VERBOSE to False
    os.environ["SWMMANYWHERE_VERBOSE"] = "false"

    # Create a mock iterator
    mock_iterator = iter(range(10))
    with patch("swmmanywhere.logging.tqdm_original") as mock_tqdm:
        # Call the custom tqdm function
        result = [i for i in tqdm(mock_iterator)]

        mock_tqdm.assert_not_called()

        # Check if the progress_bar is the same as the mock_iterator
        assert result == list(range(10))

def test_tqdm_verbose_unset():
    """Test custom tqdm with no verbose."""
    # Unset SWMMANYWHERE_VERBOSE
    os.environ["SWMMANYWHERE_VERBOSE"] = "true"
    if "SWMMANYWHERE_VERBOSE" in os.environ:
        del os.environ["SWMMANYWHERE_VERBOSE"]

    # Create a mock iterator
    mock_iterator = iter(range(10))

    with patch("swmmanywhere.logging.tqdm_original") as mock_tqdm:
        # Call the custom tqdm function
        result = [i for i in tqdm(mock_iterator)]

        mock_tqdm.assert_not_called()
        
        # Check if the progress_bar is the same as the mock_iterator
        assert result == list(range(10))