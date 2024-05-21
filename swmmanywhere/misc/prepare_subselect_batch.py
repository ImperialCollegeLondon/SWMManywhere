"""Prepare subselect cut."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

from swmmanywhere import swmmanywhere
from swmmanywhere.misc.subselect import subselect_cut

os.environ['SWMMANYWHERE_VERBOSE'] = "true"

cuts = ['G60U01F_G60K200_l1',
            'G80F380_G80F370_l1',
            'G70F320_G70F300_l1',
            'G70F130_G70F120_l1',
            'G71O020_G71O010_l1',
            'G71R062_G71R050_l1',
            'G71F06R_G71F060_l1',
            'G72R220_G72R210_l1']
base_project = 'bellinge'
base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere')

base_config = swmmanywhere.load_config(base_dir / base_project / 'bf.yml')
base_config['graphfcn_list'].remove('clip_to_catchments')

hpc_config = deepcopy(base_config)
hpc_address = Path(r'/rds/general/user/bdobson/ephemeral/swmmanywhere')
hpc_config['base_dir'] = hpc_address
hpc_config['address_overrides']['precipitation'] = \
    Path(r'/rds/general/user/bdobson/home/SWMManywhere/swmmanywhere/defs/storm.dat')

for cut in cuts:
    subselect_cut(base_dir, base_project, cut)
    cut_dir = base_dir / f'{base_project}_{cut}'
    base_config['project'] = f'{base_project}_{cut}'
    base_config['real']['inp'] = cut_dir / 'real' / 'model.inp'
    base_config['real']['graph'] = cut_dir / 'real' / 'graph.json'
    base_config['real']['subcatchments'] = cut_dir / 'real' / 'subcatchments.geojson'

    with (cut_dir / 'real' / 'real_bbox.json').open('r') as f:
        base_config['bbox'] = json.load(f)['bbox']

    inp, metrics = swmmanywhere.swmmanywhere(base_config)

    cut_hpc_dir = hpc_address / f'{base_project}_{cut}'
    hpc_config['project'] = cut
    hpc_config['real']['inp'] = None
    hpc_config['real']['graph'] = cut_hpc_dir / 'graph.json'
    hpc_config['real']['subcatchments'] = cut_hpc_dir / 'subcatchments.geojson'
    hpc_config['real']['results'] = cut_hpc_dir / 'real_results.parquet'

    swmmanywhere.save_config(hpc_config, cut_dir / 'config.yml')
