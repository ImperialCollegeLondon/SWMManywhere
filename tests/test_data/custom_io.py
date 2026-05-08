from __future__ import annotations

from swmmanywhere.post_processing import register_io


@register_io
def new_io(m, **kw):
    """New io function."""
    return m
