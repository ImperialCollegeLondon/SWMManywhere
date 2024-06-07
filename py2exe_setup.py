from distutils.core import setup
import py2exe
import shapely
import geopandas
import os

setup(
    name = 'swmmanywhere',
    windows=[{"script": "swmmanywhere/__main__.py"}],
    options={"py2exe": {"bundle_files": 1, "packages": ["shapely", "geopandas","fiona"]}},
    zipfile=None,
)
