"""Test the parameters module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from swmmanywhere.filepaths import FilePaths, filepaths_from_yaml


def test_getattr():
    """Test the __getattr__ method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir)
        addresses = FilePaths(
            base_dir=filepath, bbox_bounds=[0, 1, 0, 1], project_name="test"
        )
        assert addresses.model_paths.model_number == 1
        assert addresses.project_paths.base_dir == filepath
        assert addresses.project_paths.project == filepath / "test"
        assert (
            addresses.project_paths.national
            == addresses.project_paths.project / "national"
        )
        assert addresses.bbox_paths.bbox == addresses.project_paths.project / "bbox_1"
        assert addresses.bbox_paths.download == addresses.bbox_paths.bbox / "download"
        assert (
            addresses.bbox_paths.elevation
            == addresses.bbox_paths.download / "elevation.tif"
        )
        assert (
            addresses.bbox_paths.building
            == addresses.bbox_paths.download / "building.geoparquet"
        )
        assert addresses.model_paths.model == addresses.bbox_paths.bbox / "model_1"
        assert (
            addresses.model_paths.subcatchments
            == addresses.model_paths.model / "subcatchments.geoparquet"
        )
        assert (
            addresses.bbox_paths.precipitation
            == addresses.bbox_paths.download / "precipitation.parquet"
        )

        assert addresses.model_paths.model.exists()
        assert addresses.bbox_paths.download.exists()
        assert addresses.project_paths.national.exists()

        addresses.set_bbox_number(2)
        assert addresses.bbox_paths.bbox == addresses.project_paths.project / "bbox_2"
        assert addresses.bbox_paths.download == addresses.bbox_paths.bbox / "download"
        assert addresses.model_paths.model == addresses.bbox_paths.bbox / "model_1"


def test_to_yaml_normal():
    """Test the to_yaml and from_yaml methods."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        addresses = FilePaths(base_dir, "test", [0, 1, 0, 1])

        # Write and read
        addresses.to_yaml(base_dir / "test.yaml")
        addresses_ = filepaths_from_yaml(base_dir / "test.yaml")

        # Check
        paths_ = {
            "project_paths": ["base_dir", "project", "national", "national_building"],
            "bbox_paths": [
                "bbox",
                "download",
                "building",
                "precipitation",
                "elevation",
                "river",
                "street",
            ],
            "model_paths": [
                "model",
                "subcatchments",
                "streetcover",
                "edges",
                "nodes",
                "graph",
                "inp",
            ],
        }
        for cat, keys in paths_.items():
            for key in keys:
                assert getattr(getattr(addresses, cat), key) == getattr(
                    getattr(addresses_, cat), key
                )


def test_to_yaml_normal_with_overrides():
    """Test the to_yaml and from_yaml methods."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        dummy_file = base_dir / "dummy.txt"
        dummy_file.touch()
        addresses = FilePaths(
            base_dir,
            "test",
            [0, 1, 0, 1],
            elevation=dummy_file,
            inp=dummy_file,
            new_file=dummy_file,
        )

        # Model number override
        addresses.model_paths.model_number = 2

        # Write and read
        addresses.to_yaml(base_dir / "test.yaml")
        addresses_ = filepaths_from_yaml(base_dir / "test.yaml")

        # Check
        paths_ = {
            "project_paths": ["base_dir", "project", "national", "national_building"],
            "bbox_paths": [
                "bbox",
                "download",
                "building",
                "precipitation",
                "elevation",
                "river",
                "street",
            ],
            "model_paths": [
                "model",
                "subcatchments",
                "streetcover",
                "edges",
                "nodes",
                "graph",
                "inp",
            ],
        }
        for cat, keys in paths_.items():
            for key in keys:
                assert getattr(getattr(addresses, cat), key) == getattr(
                    getattr(addresses_, cat), key
                )
        assert addresses_.get_path("new_file") == dummy_file
