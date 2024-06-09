import sys
from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": ["os", "shapely"],  # Add any required packages here
    "includes": ["geopandas"],  # Add any additional includes here
    "include_files": ["data/"]  # Include any additional data files or directories
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="SwmmAnywhere",
    version="1.0",
    description="SWMM Application",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", base=base)]
)