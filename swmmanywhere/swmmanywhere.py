# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
from pathlib import Path

import geopandas as gpd
import jsonschema
import pandas as pd
import pyswmm
import yaml

import swmmanywhere.geospatial_utilities as go
from swmmanywhere import preprocessing
from swmmanywhere.graph_utilities import iterate_graphfcns, load_graph, save_graph
from swmmanywhere.logging import logger
from swmmanywhere.metric_utilities import iterate_metrics
from swmmanywhere.parameters import get_full_parameters
from swmmanywhere.post_processing import synthetic_write


def swmmanywhere(config: dict):
    """Run SWMManywhere processes.
    
    This function runs the SWMManywhere processes, including downloading data,
    preprocessing the graphfcns, running the model, and comparing the results 
    to real data using metrics.

    Args:
        config (dict): The loaded config as a dict.

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """
    # Create the project structure
    addresses = preprocessing.create_project_structure(config['bbox'],
                                                       config['project'],
                                                       config['base_dir']
                                                       )

    for key, val in config.get('address_overrides', {}).items():
        setattr(addresses, key, val)

    # Run downloads
    api_keys = yaml.safe_load(config['api_keys'].open('r'))
    preprocessing.run_downloads(config['bbox'],
                                addresses,
                                api_keys
                                )

    # Identify the starting graph
    if config['starting_graph']:
        G = load_graph(config['starting_graph'])
    else:
        G = preprocessing.create_starting_graph(addresses)

    # Load the parameters and perform any manual overrides
    parameters = get_full_parameters()
    for category, overrides in config.get('parameter_overrides', {}).items():
        for key, val in overrides.items():
            setattr(parameters[category], key, val)

    # Iterate the graph functions
    G = iterate_graphfcns(G, 
                          config['graphfcn_list'], 
                          parameters,
                          addresses)

    # Save the final graph
    go.graph_to_geojson(G, 
                        addresses.nodes,
                        addresses.edges,
                        G.graph['crs']
                        )
    save_graph(G, addresses.graph)
    # Write to .inp
    synthetic_write(addresses)
                    
    # Run the model
    synthetic_results = run(addresses.inp, 
                            **config['run_settings'])
    
    # Get the real results
    if config['real']['results']:
        # TODO.. bit messy
        real_results = pd.read_parquet(config['real']['results'])
    elif config['real']['inp']:
        real_results = run(config['real']['inp'],
                           **config['run_settings'])
    else:
        logger.info("No real network provided, returning SWMM .inp file.")
        return addresses.inp
    
    # Iterate the metrics
    metrics = iterate_metrics(synthetic_results,
                              gpd.read_file(addresses.subcatchments),
                              G,
                              real_results,
                              gpd.read_file(config['real']['subcatchments']),
                              load_graph(config['real']['graph']),
                              config['metric_list'])

    return metrics

def check_top_level_paths(config: dict):
    """Check the top level paths in the config.

    Args:
        config (dict): The configuration.

    Raises:
        FileNotFoundError: If a top level path does not exist.
    """
    for key in ['base_dir', 'api_keys']:
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"{key} not found at {config[key]}")
        config[key] = Path(config[key])
    return config

def check_address_overrides(config: dict):
    """Check the address overrides in the config.

    Args:
        config (dict): The configuration.

    Raises:
        FileNotFoundError: If an address override path does not exist.
    """
    overrides = config.get('address_overrides', None)
    
    if not overrides:
        return config
    
    for key, path in overrides.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{key} not found at {path}")
        config['address_overrides'][key] = Path(path)
    return config

def check_real_network_paths(config: dict):
    """Check the paths to the real network in the config.

    Args:
        config (dict): The configuration.

    Raises:
        FileNotFoundError: If a real network path does not exist.
    """
    real = config.get('real', None)
    
    if not real:
        return config
    
    for key, path in real.items():
        if not isinstance(path, str):
            continue
        if not Path(path).exists():
            raise FileNotFoundError(f"{key} not found at {path}")
        config['real'][key] = Path(path)

    return config

def load_config(config_path: Path):
    """Load, validate, and convert Paths in a configuration file.

    Args:
        config_path (Path): The path to the configuration file.

    Returns:
        dict: The configuration.
    """
    # Load the schema
    schema_fid = Path(__file__).parent / 'defs' / 'schema.yml'
    with schema_fid.open('r') as file:
        schema = yaml.safe_load(file)

    with config_path.open('r') as f:
        # Load the config
        config = yaml.safe_load(f)

    # Validate the config
    jsonschema.validate(instance = config, schema = schema)

    # Check top level paths
    config = check_top_level_paths(config)
    
    # Check address overrides
    config = check_address_overrides(config)
        
    # Check real network paths
    config = check_real_network_paths(config)
    
    return config


def run(model: Path,
        reporting_iters: int = 50,
        duration: int = 86400,
        storevars: list[str] = ['flooding','flow']):
    """Run a SWMM model and store the results.

    Args:
        model (Path): The path to the SWMM model .inp file.
        reporting_iters (int, optional): The number of iterations between
            storing results. Defaults to 50.
        duration (int, optional): The duration of the simulation in seconds.
            Starts at the 'START_DATE' and 'START_TIME' defined in the 'model'
            .inp file Defaults to 86400.
        storevars (list[str], optional): The variables to store. Defaults to
            ['flooding','flow'].

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """
    with pyswmm.Simulation(str(model)) as sim:
        sim.start()

        # Define the variables to store
        variables = {
            'flooding': {'class': pyswmm.Nodes, 'id': '_nodeid'},
            'depth': {'class': pyswmm.Nodes, 'id': '_nodeid'},
            'flow': {'class': pyswmm.Links, 'id': '_linkid'},
            'runoff': {'class': pyswmm.Subcatchments, 'id': '_subcatchmentid'}
        }

        results_list = []
        for var, info in variables.items():
            if var not in storevars:
                continue
            # Rather than calling eg Nodes or Links, only call them if they
            # are needed for storevars because they carry a significant 
            # overhead
            pobjs = info['class'](sim)
            results_list += [{'object': x, 
                            'variable': var, 
                            'id': info['id']} for x in pobjs]
        
        # Iterate the model
        results = []
        t_ = sim.current_time
        ind = 0
        while ((sim.current_time - t_).total_seconds() <= duration) & \
            (sim.current_time < sim.end_time) & (not sim._terminate_request):
            
            ind+=1

            # Iterate the main model timestep
            time = sim._model.swmm_step()
            
            # Break condition
            if time < 0:
                sim._terminate_request = True
                break
            
            # Check whether to save results
            if ind % reporting_iters != 1:
                continue

            # Store results in a list of dictionaries
            for storevar in results_list:
                results.append({'date' : sim.current_time,
                                'value' : getattr(storevar['object'],
                                                  storevar['variable']),
                                'variable' : storevar['variable'],
                                'id' : getattr(storevar['object'],
                                               storevar['id'])})
            
            
    return pd.DataFrame(results)