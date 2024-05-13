"""Debug results by recalculating metrics.

This script provides a way to load a model file from the default setup in 
experimenter.py and recalculate the metrics. This is useful for recreating 
how a metric is calculated to verify that it is being done correctly. In this 
example we reproduce code from `metric_utilities.py` to check how timeseries 
data are aligned and compared. 
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from swmmanywhere.graph_utilities import load_graph
from swmmanywhere.metric_utilities import (
    align_by_shape,
    best_outlet_match,
    create_grid,
    dominant_outlet,
    extract_var,
    iterate_metrics,
    pbias,
)
from swmmanywhere.parameters import MetricEvaluation
from swmmanywhere.swmmanywhere import load_config

if __name__ == 'main':
    project = 'subselect'
    base = Path.home() / "Documents" / "data" / "swmmanywhere"
    config_path = base / project / f'{project}.yml'
    config = load_config(config_path, validation = False)
    config['base_dir'] = base / project
    real_dir = config['base_dir'] / 'real'

    model_number = 24

    model_dir = config['base_dir'] / 'bbox_1' / f'model_{model_number}'

    syn_results = pd.read_parquet(model_dir / 'results.parquet')
    real_results = pd.read_parquet(real_dir / 'real_results.parquet')

    syn_G = load_graph(model_dir / 'assign_id_graph.json')
    real_G = load_graph(real_dir / 'graph.json')

    syn_subcatchments = gpd.read_file(model_dir / 'subcatchments.geoparquet')
    real_subcatchments = gpd.read_file(real_dir / 'subcatchments.geojson')

    
    syn_metrics = iterate_metrics(syn_results, 
                                  syn_subcatchments,
                                  syn_G,
                                  real_results,
                                  real_subcatchments,
                                  real_G,
                                  config['metric_list'],
                                  MetricEvaluation()
                                  )
    
    # Check outlet scale
    synthetic_results = syn_results.copy()
    real_results_ = real_results.copy()
    sg_syn, syn_outlet = best_outlet_match(syn_G, real_subcatchments)
    sg_real, real_outlet = dominant_outlet(real_G, real_results)

    # Check nnodes
    print(f'n syn nodes {len(sg_syn.nodes)}')
    print(f'n real nodes {len(sg_real.nodes)}')

    # Check contributing area
    syn_subcatchments['impervious_area'].sum() / syn_subcatchments['area'].sum()
    real_subcatchments['impervious_area'].sum() /\
          real_subcatchments['Area'].mul(10000).sum()
    
    variable = 'flooding'
    method = 'grid'

    if method == 'grid':
        if variable != 'flooding':
            raise ValueError('Only flooding can be compared by grid')
        scale = 100
        grid = create_grid(real_subcatchments.total_bounds,
                       scale)
        grid.crs = real_subcatchments.crs
        shps = grid
    elif method == 'subcatchment':
        shps = real_subcatchments

    #e.g., subs
    results = align_by_shape(variable,
                                synthetic_results = synthetic_results,
                                real_results = real_results,
                                shapes = shps,
                                synthetic_G = syn_G,
                                real_G = real_G)
    if method == 'grid':
        gb = (
            results
            .groupby(['date','sub_id'])
            .sum()
            .reset_index()
            .pivot(index = 'date',
                columns = 'sub_id',
                values = ['value_real','value_syn']
                )
        )
        val = (
            results
            .groupby(['date','sub_id'])
            .sum()
            .reset_index()
            .groupby('sub_id')
            .apply(lambda x: pbias(x.value_real, x.value_syn))
            .rename('pbias')
            .reset_index()
        )
        val = val[np.isfinite(val)]
        grid = pd.merge(grid, val, on = 'sub_id')


        for sub in gb['value_real'].columns:        
            f,ax=plt.subplots()
            gb['value_real'][[sub]].plot(color='r',ax=ax)
            gb['value_syn'][[sub]].plot(color='b',ax=ax)
            plt.legend(['real','syn'])
            f.savefig(model_dir / f'debug{sub}.png')
            plt.close(f)

    # e.g., outlet
    if variable == 'flow':
        syn_ids = [d['id'] for u,v,d in syn_G.edges(data=True)
                        if v == syn_outlet]
        real_ids = [d['id'] for u,v,d in real_G.edges(data=True)
                if v == real_outlet]
    else:
        syn_ids = list(sg_syn.nodes)
        real_ids = list(sg_real.nodes)
    synthetic_results['date'] = pd.to_datetime(synthetic_results['date'])
    real_results['date'] = pd.to_datetime(real_results['date'])

    # Help alignment
    synthetic_results["id"] = synthetic_results["id"].astype(str)
    real_results["id"] = real_results["id"].astype(str)
    syn_ids = [str(x) for x in syn_ids]
    real_ids = [str(x) for x in real_ids]
    # Extract data
    syn_data = extract_var(synthetic_results, variable)
    
    # All syn flowing to outlet
    syn_nodes = gpd.read_file(model_dir / 'nodes.geoparquet')
    nodes_group = syn_nodes.loc[syn_nodes.outlet == syn_outlet]
    syn_group = syn_data.loc[
        syn_data.id.isin([f'{u}-{v}' for u,v in syn_G.edges() 
                          if v in nodes_group.id.values])
                          ]
    
    # Sum all flow to outlet
    syn_data = syn_data.loc[syn_data["id"].isin(syn_ids)]
    syn_data = syn_data.groupby('date').value.sum()

    real_data = extract_var(real_results, variable)
    real_data = real_data.loc[real_data["id"].isin(real_ids)]
    real_data = real_data.groupby('date').value.sum()
    
    
    # Align data
    df = pd.merge(syn_data, 
                real_data, 
                left_index = True,
                right_index = True,
                suffixes=('_syn', '_real'), 
                how='outer').sort_index()

    # Interpolate to time in real data
    df['value_syn'] = df.value_syn.interpolate().to_numpy()
    df = df.dropna(subset=['value_real'])

    f,ax = plt.subplots()
    df.plot(ax=ax)
    f.savefig(model_dir / 'debug.png')