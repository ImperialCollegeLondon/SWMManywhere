"""Utilities for YAML save/load.

Author: cheginit
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    SafeDumper = yaml.SafeDumper
    from yaml.nodes import Node
else:
    Node = Any
    SafeDumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)

yaml_load = functools.partial(
    yaml.load, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)
)


class PathDumper(SafeDumper):
    """A dumper that can represent pathlib.Path objects as strings."""

    def represent_data(self, data: Any) -> Node:
        """Represent data."""
        if isinstance(data, Path):
            return self.represent_scalar("tag:yaml.org,2002:str", str(data))
        return super().represent_data(data)


def yaml_dump(o: Any, stream: Any = None, **kwargs: Any) -> str:
    """Dump YAML.

    Notes:
    -----
    When python/mypy#1484 is solved, this can be ``functools.partial``
    """
    return yaml.dump(
        o,
        Dumper=PathDumper,
        stream=stream,
        default_flow_style=False,
        indent=2,
        sort_keys=False,
        **kwargs,
    )
