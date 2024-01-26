# -*- coding: utf-8 -*-
"""Created on Tue Oct 18 10:35:51 2022.

@author: Barney
"""

# import pytest
import os

import geopandas as gpd
import rasterio

from swmmanywhere import prepare_data as downloaders


# Test get_country
def test_get_uk():
    """Check a UK point."""
    # Coordinates for London, UK
    x = -0.1276
    y = 51.5074

    result = downloaders.get_country(x, y)

    assert result[2] == 'GB'
    assert result[3] == 'GBR'

def test_get_us():
    """Check a US point."""
    x = -113.43318
    y = 33.81869

    result = downloaders.get_country(x, y)

    assert result[2] == 'US'
    assert result[3] == 'USA'

def test_building_downloader():
    """Check buildings are downloaded."""
    # Coordinates for small country (VAT)
    x = 7.41839
    y = 43.73205
    temp_fid = 'temp.parquet'
    # Download
    response = downloaders.download_buildings(temp_fid, x,y)
    # Check response
    assert response == 200

    # Check file exists
    assert os.path.exists(temp_fid), "Buildings data file not found after download."

    try: 
        # Load data
        gdf = gpd.read_parquet(temp_fid)

        # Make sure has some rows
        assert gdf.shape[0] > 0

    finally:
        # Regardless of test outcome, delete the temp file
        if os.path.exists(temp_fid):
            os.remove(temp_fid)

def test_street_downloader():
    """Check streets are downloaded and a specific point in the graph."""
    bbox = (-0.17929,51.49638, -0.17383,51.49846)
    G = downloaders.download_street(bbox)

    # Not sure if they they are likely to change the osmid
    assert 26389449 in G.nodes

def test_river_downloader():
    """Check rivers are downloaded and a specific point in the graph."""
    bbox = (0.0402, 51.55759, 0.09825591114207548, 51.6205)
    G = downloaders.download_river(bbox)

    # Not sure if they they are likely to change the osmid
    assert 21473922 in G.nodes

def test_elevation_downloader():
    """Check elevation downloads, writes, contains data, and a known elevation."""
    # Please do not reuse api_key
    test_api_key = 'b206e65629ac0e53d599e43438560d28' 

    bbox = (-0.17929,51.49638, -0.17383,51.49846)

    temp_fid = 'temp.tif'    

    # Download
    response = downloaders.download_elevation(temp_fid, bbox, test_api_key)

    # Check response
    assert response == 200
    
    # Check response
    assert os.path.exists(temp_fid), "Elevation data file not found after download."

    try: 
        # Load data
        with rasterio.open(temp_fid) as src:
            data = src.read(1)  # Reading the first band as an example

        # Make sure it has some values
        assert data.size > 0, "Elevation data should have some values."
        
        # Test some property of data (not sure if they may change this 
        # data)
        assert data.max().max() > 25, "Elevation data should be higher."

    finally:
        # Regardless of test outcome, delete the temp file
        if os.path.exists(temp_fid):
            os.remove(temp_fid)
