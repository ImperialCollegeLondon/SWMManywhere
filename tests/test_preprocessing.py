from __future__ import annotations

from pathlib import Path

from swmmanywhere.parameters import FilePaths


def test_getattr():
    """Test the __getattr__ method."""
    filepath = Path(__file__).parent
    addresses = FilePaths(base_dir=filepath,
                          project_name='test',
                          bbox_number=1,
                          model_number=1,
                          extension = 'parquet')
    assert addresses.model_number == 1
    assert addresses.base_dir == filepath
    assert addresses.project == filepath / 'test'
    assert addresses.national == addresses.project / 'national'
    assert addresses.bbox == addresses.project / 'bbox_1'
    assert addresses.download == addresses.bbox / 'download'
    assert addresses.elevation == addresses.download / 'elevation.tif'
    assert addresses.building == addresses.download / 'building.geoparquet'
    assert addresses.model == addresses.bbox / 'model_1'
    assert addresses.subcatchments == addresses.model / 'subcatchments.geoparquet'
    assert addresses.precipitation == addresses.download / 'precipitation.parquet'

    addresses.elevation = filepath
    assert addresses.elevation == filepath

    addresses.bbox = filepath
    assert addresses.bbox == filepath
    assert addresses.elevation == filepath
    assert addresses.precipitation == filepath / 'download/precipitation.parquet'