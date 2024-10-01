"""Debugging the derive_rc_alt function.

This is an alternative function for derive_rc in geospatial_utilities/derive_rc
it is used to double check that they are performing correctly. It may also be
more computationally efficient.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import shapely


def derive_rc_alt(
    subcatchments: gpd.GeoDataFrame,
    building_footprints: gpd.GeoDataFrame,
    streetcover: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Derive the Runoff Coefficient (RC) of each subcatchment (alt)."""
    building_footprints = building_footprints.copy()
    streetcover = streetcover.copy()
    subcatchments = subcatchments.copy()

    # Derive impervious without overlaps
    intersecting_geom_sidx, intersecting_geom_bidx = building_footprints.sindex.query(
        streetcover.geometry, predicate="intersects"
    )
    intersecting_building_geom = building_footprints.iloc[
        intersecting_geom_bidx
    ].geometry
    intersecting_street_geom = streetcover.iloc[intersecting_geom_sidx].geometry

    if intersecting_geom_sidx.shape[0] > 0:
        unified_geom = intersecting_building_geom.unary_union.union(
            intersecting_street_geom.unary_union
        )
        unified_geom = gpd.GeoDataFrame(
            geometry=[unified_geom], crs=building_footprints.crs
        )
    else:
        unified_geom = gpd.GeoDataFrame(geometry=[], crs=building_footprints.crs)
    building_footprints = building_footprints.drop(
        building_footprints.index[intersecting_geom_bidx], axis=0
    )
    streetcover = streetcover.drop(streetcover.index[intersecting_geom_sidx], axis=0)

    new_geoms = [
        g for g in [unified_geom, building_footprints, streetcover] if g.shape[0] > 0
    ]
    # Create the "unified" impervious geometries
    impervious = gpd.GeoDataFrame(pd.concat(new_geoms), crs=building_footprints.crs)
    subcat_tree = subcatchments.sindex
    bf_pidx, sb_pidx = subcat_tree.query(impervious.geometry, predicate="intersects")
    sb_idx = subcatchments.iloc[sb_pidx].index

    # Calculate impervious area and runoff coefficient (rc)
    subcatchments["impervious_area"] = 0.0

    # Calculate all intersection-impervious geometries
    intersection_area = shapely.intersection(
        subcatchments.iloc[sb_pidx].geometry.to_numpy(),
        impervious.iloc[bf_pidx].geometry.to_numpy(),
    )

    # Indicate which catchment each intersection is part of
    intersections = pd.DataFrame(
        [
            {"sb_idx": ix, "impervious_geometry": ia}
            for ix, ia in zip(sb_idx, intersection_area)
        ]
    )

    # Aggregate by catchment
    areas = (
        intersections.groupby("sb_idx")
        .apply(shapely.ops.unary_union)
        .apply(shapely.area)
    )

    # Store as impervious area in subcatchments
    subcatchments["impervious_area"] = 0
    subcatchments.loc[areas.index, "impervious_area"] = areas
    subcatchments["rc"] = (
        subcatchments["impervious_area"] / subcatchments.geometry.area * 100
    )
    return subcatchments
