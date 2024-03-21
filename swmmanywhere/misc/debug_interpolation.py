"""Test how well SWMM simulations line up."""
from pathlib import Path

import geopandas as gpd
import pandas as pd

from swmmanywhere import metric_utilities as mu
from swmmanywhere.graph_utilities import load_graph

base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere\cranbrook')
real_results = pd.read_parquet(base_dir / 'real' / 'real_results.parquet')
synthetic_results = pd.read_parquet(base_dir / 'bbox_1' / 'model_1' /\
                                     'results.parquet')
synthetic_G = load_graph(base_dir / 'bbox_1' / 'model_1' / 'assign_id_graph.json')
real_G = load_graph(base_dir / 'real' / 'graph.json')
real_subs = gpd.read_file(base_dir / 'real' / 'subcatchments.geojson')

# [from mu.outlet_nse_flow]
# Identify synthetic and real arcs that flow into the best outlet node
_, syn_outlet = mu.best_outlet_match(synthetic_G, real_subs)
syn_ids = [d['id'] for u,v,d in synthetic_G.edges(data=True)
            if v == syn_outlet]
_, real_outlet =  mu.dominant_outlet(real_G, real_results)
real_ids = [d['id'] for u,v,d in real_G.edges(data=True)
            if v == real_outlet]

variable = 'flow'

# [from mu.align_calc_nse]
# Format dates
synthetic_results['date'] = pd.to_datetime(synthetic_results['date'])
real_results['date'] = pd.to_datetime(real_results['date'])

# Extract data
syn_data = mu.extract_var(synthetic_results, variable)
syn_data = syn_data.loc[syn_data.id.isin(syn_ids)]
syn_data = syn_data.groupby('date').value.sum()

real_data = mu.extract_var(real_results, variable)
real_data = real_data.loc[real_data.id.isin(real_ids)]
real_data = real_data.groupby('date').value.sum()

# Align data
df = pd.merge(syn_data, 
                real_data, 
                left_index = True,
                right_index = True,
                suffixes=('_syn', '_real'), 
                how='outer').sort_index()

print(str(df.dropna().shape[0] / df.shape[0] * 100))
# We find that syn_data and real_data perfect align 0.2% of the time. Thus, 
# the interpolation in align_calc_nse is certainly needed.