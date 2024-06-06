"""The main SWMManywhere module to generate and run a synthetic network."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import jsonschema
import pandas as pd
import pyswmm
from tqdm.auto import tqdm

import swmmanywhere.geospatial_utilities as go
from swmmanywhere import parameters, preprocessing
from swmmanywhere.graph_utilities import iterate_graphfcns, load_graph, save_graph
from swmmanywhere.logging import logger, verbose
from swmmanywhere.metric_utilities import iterate_metrics
from swmmanywhere.post_processing import synthetic_write
from swmmanywhere.utilities import yaml_dump, yaml_load


def swmmanywhere(config: dict) -> tuple[Path, dict | None]:
    """Run SWMManywhere processes.
    
    This function runs the SWMManywhere processes, including downloading data,
    preprocessing the graphfcns, running the model, and comparing the results 
    to real data using metrics. The function will always return the path to 
    the generated .inp file. If real data (either a results file or the .inp, 
    as well as graph, and subcatchments) is provided, the function will also 
    return the metrics comparing the synthetic network with the real.

    Args:
        config (dict): The loaded config as a dict.

    Returns:
        tuple[Path, dict | None]: The address of generated .inp and metrics.
    """
    # Create the project structure
    logger.info("Creating project structure.")
    addresses = preprocessing.create_project_structure(config['bbox'],
                                config['project'],
                                config['base_dir'],
                                config.get('model_number',None)
                                )
    addresses.extension='json'
    logger.info(f"Project structure created at {addresses.base_dir}")
    logger.info(f"Project name: {config['project']}")
    logger.info(f"Bounding box: {config['bbox']}, number: {addresses.bbox_number}")
    logger.info(f"Model number: {addresses.model_number}")

    for key, val in config.get('address_overrides', {}).items():
        logger.info(f"Setting {key} to {val}")
        setattr(addresses, key, val)

    # Load the parameters and perform any manual overrides
    logger.info("Loading and setting parameters.")
    params = parameters.get_full_parameters()
    for category, overrides in config.get('parameter_overrides', {}).items():
        for key, val in overrides.items():
            logger.info(f"Setting {category} {key} to {val}")
            setattr(params[category], key, val)
            
    # Save config file
    if verbose():
        save_config(config, addresses.model / 'config.yml')

    # Run downloads
    logger.info("Running downloads.")
    api_keys = yaml_load(config['api_keys'].read_text())
    preprocessing.run_downloads(config['bbox'],
                addresses,
                api_keys,
                network_types = params['topology_derivation'].allowable_networks
                )

    # Identify the starting graph
    logger.info("Iterating graphs.")
    if config.get('starting_graph', None):
        G = load_graph(config['starting_graph'])
    else:
        G = preprocessing.create_starting_graph(addresses)

    # Load the parameters and perform any manual overrides
    logger.info("Loading and setting parameters.")
    params = parameters.get_full_parameters()
    for category, overrides in config.get('parameter_overrides', {}).items():
        for key, val in overrides.items():
            logger.info(f"Setting {category} {key} to {val}")
            setattr(params[category], key, val)

    # Iterate the graph functions
    logger.info("Iterating graph functions.")
    G = iterate_graphfcns(G, 
                          config['graphfcn_list'], 
                          params,
                          addresses)

    # Save the final graph
    logger.info("Saving final graph and writing inp file.")
    go.graph_to_geojson(G, 
                        addresses.nodes,
                        addresses.edges,
                        G.graph['crs']
                        )
    save_graph(G, addresses.graph)
    # Write to .inp
    synthetic_write(addresses)
                    
    # Run the model
    logger.info("Running the synthetic model.")
    synthetic_results = run(addresses.inp, 
                            **config['run_settings'])
    logger.info("Writing synthetic results.")
    if verbose():
        synthetic_results.to_parquet(addresses.model /\
                                      f'results.{addresses.extension}')

    # Get the real results
    if config['real'].get('results',None):
        logger.info("Loading real results.")
        # TODO.. bit messy
        real_results = pd.read_parquet(config['real']['results'])
    elif config['real']['inp']:
        logger.info("Running the real model.")
        real_results = run(config['real']['inp'],
                           **config['run_settings'])
        if verbose():
            real_results.to_parquet(config['real']['inp'].parent /\
                                     f'real_results.{addresses.extension}')
    else:
        logger.info("No real network provided, returning SWMM .inp file.")
        return addresses.inp, None
    
    # Iterate the metrics
    logger.info("Iterating metrics.")
    metrics = iterate_metrics(synthetic_results,
                              gpd.read_file(addresses.subcatchments),
                              G,
                              real_results,
                              gpd.read_file(config['real']['subcatchments']),
                              load_graph(config['real']['graph']),
                              config['metric_list'],
                              params['metric_evaluation'])
    logger.info("Metrics complete")
    return addresses.inp, metrics

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

def check_parameters_to_sample(config: dict):
    """Check the parameters to sample in the config.

    Args:
        config (dict): The configuration.

    Raises:
        ValueError: If a parameter to sample is not in the parameters
            dictionary.
    """
    params = parameters.get_full_parameters_flat()
    for param in config.get('parameters_to_sample',{}):
        # If the parameter is a dictionary, the values are bounds, all we are 
        # checking here is that the parameter exists, we only need the first 
        # entry.
        if isinstance(param, dict):
            if len(param) > 1:
                raise ValueError("""If providing new bounds in the config, a dict 
                                 of len 1 is required, where the key is the 
                                 parameter to change and the values are 
                                 (new_lower_bound, new_upper_bound).""")
            param = list(param.keys())[0]

        # Check that the parameter is available
        if param not in params:
            raise ValueError(f"{param} not found in parameters dictionary.")
        
        # Check that the parameter is sample-able
        required_attrs = set(['minimum', 'maximum', 'default', 'category'])
        correct_attrs = required_attrs.intersection(params[param])
        missing_attrs = required_attrs.difference(correct_attrs)
        if any(missing_attrs):
            raise ValueError(f"{param} missing {missing_attrs} so cannot be sampled.")
        
    return config

def check_starting_graph(config: dict):
    """Check the starting graph in the config.

    Args:
        config (dict): The configuration.

    Raises:
        FileNotFoundError: If the starting graph path does not exist.
    """
    # If no starting graph, return
    if not config.get('starting_graph', None):
        return config
    
    # Check the starting graph exists and convert to Path
    config['starting_graph'] = Path(config['starting_graph'])
    if not config['starting_graph'].exists():
        raise FileNotFoundError(f"""starting_graph not found at 
                                {config['starting_graph']}""")

    return config

def check_parameter_overrides(config: dict):
    """Check the parameter overrides in the config.

    Args:
        config (dict): The configuration.

    Raises:
        ValueError: If a parameter override is not in the parameters
            dictionary.
    """
    params = parameters.get_full_parameters()
    for category, overrides in config.get('parameter_overrides',{}).items():
        if category not in params:
            raise ValueError(f"""{category} not a category of parameter. Must
                             be one of {params.keys()}.""")
        
        # Get the available properties for a category
        cat_properties = params[category].model_json_schema()['properties']

        for key, val in overrides.items():
            # Check that the parameter is available
            if key not in cat_properties:
                raise ValueError(f"{key} not found in {category}.")            
            
    return config

def save_config(config: dict, config_path: Path):
    """Save the configuration to a file.

    Args:
        config (dict): The configuration.
        config_path (Path): The path to save the configuration.
    """
    yaml_dump(config, config_path.open('w'))

def load_config(config_path: Path, validation: bool = True):
    """Load, validate, and convert Paths in a configuration file.

    Args:
        config_path (Path): The path to the configuration file.
        validation (bool, optional): Whether to validate the configuration.

    Returns:
        dict: The configuration.
    """
    # Load the schema
    schema_fid = Path(__file__).parent / 'defs' / 'schema.yml'
    schema = yaml_load(schema_fid.read_text())

    # Load the config
    config = yaml_load(config_path.read_text())

    if not validation:
        return config
    
    # Validate the config
    jsonschema.validate(instance = config, schema = schema)

    # Check top level paths
    config = check_top_level_paths(config)
    
    # Check address overrides
    config = check_address_overrides(config)
        
    # Check real network paths
    config = check_real_network_paths(config)
    
    # Check the parameters to sample
    config = check_parameters_to_sample(config)

    # Check starting graph
    config = check_starting_graph(config)

    # Check parameter overrides
    config = check_parameter_overrides(config)

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
        logger.info(f"{model} initialised in pyswmm")

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
        logger.info(f"Starting simulation for: {model}")

        progress_bar = tqdm(total=duration, disable = not verbose())

        offset = 0
        while (offset <= duration) & \
            (sim.current_time < sim.end_time) & (not sim._terminate_request):
            
            progress_bar.update((sim.current_time - t_).total_seconds() - offset)
            offset = (sim.current_time - t_).total_seconds()
            
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
            
    logger.info("Model run complete.")
    return pd.DataFrame(results)