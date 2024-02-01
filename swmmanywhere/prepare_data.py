# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""

import os
import shutil
from typing import cast

import cdsapi
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
import xarray as xr
import yaml
from geopy.geocoders import Nominatim

# Some minor comment (to remove)

def get_country(x: float, 
                y: float) -> dict[int, str]:
    """Identify which country a point (x,y) is in.
    
    Return the two and three letter ISO code for coordinates x (longitude)
    and y (latitude)

    Args:
        x (float): Longitude.
        y (float): Latitude.

    Returns:
        dict: A dictionary containing the two and three letter ISO codes.
            Example: {2: 'GB', 3: 'GBR'}
    """
    # Get the directory of the current module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # TODO use 'load_yaml_from_defs'
    # Create the path to iso_converter.yml
    iso_path = os.path.join(current_dir, 
                            "defs", 
                            "iso_converter.yml")


    # Initialize geolocator
    geolocator = Nominatim(user_agent="get_iso")

    # Load ISO code mapping from YAML file
    with open(iso_path, "r") as file:
        data = yaml.safe_load(file)

    # Get country ISO code from coordinates
    location = geolocator.reverse("{0}, {1}".format(y, x), exactly_one=True)
    iso_country_code = location.raw.get("address", {}).get("country_code")
    iso_country_code = iso_country_code.upper()

    # Return a dictionary with the two and three letter ISO codes
    return {2: iso_country_code, 3: data.get(iso_country_code, '')}

def download_buildings(file_address: str, 
                       x: float, 
                       y: float) -> int:
    """Download buildings data based on coordinates and save to a file.

    Args:
        file_address (str): File address to save the downloaded data.
        x (float): Longitude.
        y (float): Latitude.
    
    Returns:
        status_code (int): Reponse status code
    """
    # Get three letter ISO code
    iso_country_code = get_country(x, y)[3]

    # Construct API URL
    api_url = f"https://data.source.coop/vida/google-microsoft-open-buildings/geoparquet/by_country/country_iso={iso_country_code}/{iso_country_code}.parquet"

    # Download data
    response = requests.get(api_url)
    if response.status_code == 200:
        # Save data to the specified file address
        with open(file_address, "wb") as file:
            file.write(response.content)
        print(f"Data downloaded and saved to {file_address}")
    else:
        print(f"Error downloading data. Status code: {response.status_code}")
    return response.status_code

def download_street(bbox: tuple[float, float, float, float]) -> nx.MultiDiGraph:
    """Get street network within a bounding box using OSMNX.
    
    [CREDIT: Taher Cheghini busn_estimator package]

    Args:
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form 
            of (west, south, east, north) at EPSG:4326.

    Returns:
        nx.MultiDiGraph: Street network with type drive and 
            ``truncate_by_edge set`` to True.
    """
    west, south, east, north = bbox
    graph = ox.graph_from_bbox(
        north, south, east, west, network_type="drive", truncate_by_edge=True
    )
    return cast("nx.MultiDiGraph", graph)

def download_river(bbox: tuple[float, float, float, float]) -> nx.MultiDiGraph:
    """Get water network within a bounding box using OSMNX.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form 
            of (west, south, east, north) at EPSG:4326.

    Returns:
        nx.MultiDiGraph: River network with type waterway and 
            ``truncate_by_edge set`` to True.
    """
    west, south, east, north = bbox
    graph = ox.graph_from_bbox(
        north, 
        south, 
        east, 
        west,
        truncate_by_edge=True, 
        custom_filter='["waterway"]')
    
    return cast("nx.MultiDiGraph", graph)

def download_elevation(fid: str, 
                       bbox: tuple[float, float, float, float], 
                       api_key: str ='<your_api_key>') -> int:
    """Download NASADEM elevation data from OpenTopography API.

    Downloads elevation data in GeoTIFF format from OpenTopography API based on
      the specified bounding box.

    Args:
        fid (str): File path to save the downloaded elevation data.
        bbox (tuple): Bounding box coordinates in the format 
            (minx, miny, maxx, maxy).
        api_key (str, optional): Your OpenTopography API key. 
            Defaults to '<your_api_key>'.

    Returns:
        status_code (int): Reponse status code

    Raises:
        requests.exceptions.RequestException: If there is an error in the API 
            request.

    Example:
        ```
        bbox = (-120, 35, -118, 37)  # Example bounding box coordinates
        download_elevation('elevation_data.tif', 
                            bbox, 
                            api_key='your_actual_api_key')
        ```

    Note:
        To obtain an API key, you need to sign up on the OpenTopography 
        website.

    """
    minx, miny, maxx, maxy = bbox
    url = f'https://portal.opentopography.org/API/globaldem?demtype=NASADEM&south={miny}&north={maxy}&west={minx}&east={maxx}&outputFormat=GTiff&API_Key={api_key}'
    
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        with open(fid, 'wb') as rast_file:
            shutil.copyfileobj(r.raw, rast_file)
            
        print('Elevation data downloaded successfully.')

    except requests.exceptions.RequestException as e:
        print(f'Error downloading elevation data: {e}')
    
    return r.status_code

def download_precipitation(bbox: tuple[float, float, float, float], 
                           start_date: str = '2015-01-01',
                           end_date: str = '2015-01-05',
                           username: str = '<your_username>',
                           api_key: str = '<your_api_key>') -> pd.DataFrame:
    """Download precipitation data within bbox from ERA5.
    
    Register for an account and API key at: https://cds.climate.copernicus.eu. 
    Produces hourly gridded data (31km grid) in CRS EPSG:4326.
    More information at: https://github.com/ecmwf/cdsapi.

    Args:
        bbox (tuple): Bounding box coordinates in the format 
            (minx, miny, maxx, maxy).
        start_date (str, optional): Start date. Defaults to '2015-01-01'.
        end_date (str, optional): End date. Defaults to '2015-01-05'.
        username (str, optional): CDS api username. 
            Defaults to '<your_username>'.
        api_key (str, optional): CDS api key. Defaults to '<your_api_key>'.

    Returns:
        df (DataFrame): DataFrame containing downloaded data.
    """
    # Notes for docstring:
        # looks like api will give nearest point if not in bbox
        
    # Define the request parameters
    request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'total_precipitation',
            'date' : '/'.join([start_date,end_date]),
            'time': '/'.join([f'{str(x).zfill(2)}:00' for x in range(24)]),
            'area': bbox,
        }

    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api/v2', 
                      key='{0}:{1}'.format(username, api_key))
    # Get data
    c.retrieve('reanalysis-era5-single-levels', 
               request, 
               'download.nc'
               )
    
    with xr.open_dataset('download.nc') as data:
        # Convert the xarray Dataset to a pandas DataFrame
        df = data.to_dataframe()
        df['unit'] = 'm/hr'
        
    
    # Delete nc file
    os.remove('download.nc')

    return df