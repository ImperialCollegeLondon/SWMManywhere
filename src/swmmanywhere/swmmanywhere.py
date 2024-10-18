"""The main SWMManywhere module to generate and run a synthetic network."""

from __future__ import annotations

import importlib
from pathlib import Path

import geopandas as gpd
import jsonschema
import pandas as pd
import pyswmm
from tqdm.auto import tqdm

import swmmanywhere.geospatial_utilities as go
from swmmanywhere import filepaths, parameters, preprocessing
from swmmanywhere.graph_utilities import (
    iterate_graphfcns,
    load_graph,
    save_graph,
    validate_graphfcn_list,
)
from swmmanywhere.logging import logger, verbose
from swmmanywhere.metric_utilities import iterate_metrics, validate_metric_list
from swmmanywhere.post_processing import synthetic_write
from swmmanywhere.utilities import yaml_dump, yaml_load


def _check_defaults(config: dict) -> dict:
    """Check the config for needed values and add them from defaults if missing.

    Args:
        config (dict): The configuration.

    Returns:
        dict: The configuration with defaults added.
    """
    config_ = load_config(validation=False)
    for key in ["run_settings", "graphfcn_list", "metric_list"]:
        if key not in config:
            config[key] = config_[key]

    return config


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
    # Check for defaults
    config = _check_defaults(config)

    # Currently precipitation must be provided via address_overrides, otherwise
    # the default storm.dat file will be used
    if not Path(
        config.get("address_overrides", {}).get(
            "precipitation", Path("precipitation.dat")
        )
    ).exists():
        config["address_overrides"] = config.get("address_overrides", {})
        config["address_overrides"]["precipitation"] = (
            Path(__file__).parent / "defs" / "storm.dat"
        )

    # Create the project structure
    logger.info("Creating project structure.")
    addresses = filepaths.FilePaths(
        config["base_dir"],
        config["project"],
        config["bbox"],
        config.get("bbox_number", None),
        config.get("model_number", None),
        config.get("extension", "parquet"),
        **config.get("address_overrides", {}),
    )

    logger.info(f"Project structure created at {addresses.project_paths.base_dir}")
    logger.info(f"Project name: {config['project']}")
    logger.info(
        f"""Bounding box: {config['bbox']}, 
                number: {addresses.bbox_paths.bbox_number}"""
    )
    logger.info(f"Model number: {addresses.model_paths.model_number}")

    # Save config file
    if verbose():
        save_config(config, addresses.model_paths.model / "config.yml")

    # Load the parameters and perform any manual overrides
    logger.info("Loading and setting parameters.")
    params = parameters.get_full_parameters()
    for category, overrides in config.get("parameter_overrides", {}).items():
        for key, val in overrides.items():
            logger.info(f"Setting {category} {key} to {val}")
            setattr(params[category], key, val)

    # If `allowable_networks` has been changed, force a redownload of street graph.
    if "allowable_networks" in config.get("parameter_overrides", {}).get(
        "topology_derivation", {}
    ):
        logger.info("Allowable networks have been changed, removing old street graph.")
        addresses.bbox_paths.street.unlink(missing_ok=True)

    # Run downloads
    logger.info("Running downloads.")
    preprocessing.run_downloads(
        config["bbox"],
        addresses,
        network_types=params["topology_derivation"].allowable_networks,
    )

    # Identify the starting graph
    logger.info("Iterating graphs.")
    if config.get("starting_graph", None):
        G = load_graph(config["starting_graph"])
    else:
        G = preprocessing.create_starting_graph(addresses)

    # Validate the graphfcn order
    validate_graphfcn_list(config["graphfcn_list"], G)

    # Iterate the graph functions
    logger.info("Iterating graph functions.")
    G = iterate_graphfcns(G, config["graphfcn_list"], params, addresses)

    # Save the final graph
    logger.info("Saving final graph and writing inp file.")
    go.graph_to_geojson(
        G, addresses.model_paths.nodes, addresses.model_paths.edges, G.graph["crs"]
    )
    save_graph(G, addresses.model_paths.graph)

    # Check any edges
    if len(G.edges) == 0:
        logger.warning("No edges in graph, returning graph file.")
        return addresses.model_paths.graph, None

    # Write to .inp
    synthetic_write(addresses)

    # Run the model
    logger.info("Running the synthetic model.")
    synthetic_results = run(addresses.model_paths.inp, **config["run_settings"])
    logger.info("Writing synthetic results.")
    if verbose():
        synthetic_results.to_parquet(addresses.model_paths.model / "results.parquet")

    # Get the real results
    if config.get("real", {}).get("results", None):
        logger.info("Loading real results.")
        real_results = pd.read_parquet(config["real"]["results"])
    elif config.get("real", {}).get("inp", None):
        logger.info("Running the real model.")
        real_results = run(config["real"]["inp"], **config["run_settings"])
        if verbose():
            real_results.to_parquet(
                config["real"]["inp"].parent / "real_results.parquet"
            )
    else:
        logger.info("No real network provided, returning SWMM .inp file.")
        return addresses.model_paths.inp, None

    # Iterate the metrics
    logger.info("Iterating metrics.")
    if addresses.model_paths.subcatchments.suffix == ".geoparquet":
        subs = gpd.read_parquet(addresses.model_paths.subcatchments)
    else:
        subs = gpd.read_file(addresses.model_paths.subcatchments)

    if config["real"]["subcatchments"].suffix == ".geoparquet":
        real_subs = gpd.read_parquet(config["real"]["subcatchments"])
    else:
        real_subs = gpd.read_file(config["real"]["subcatchments"])
    metrics = iterate_metrics(
        synthetic_results,
        subs,
        G,
        real_results,
        real_subs,
        load_graph(config["real"]["graph"]),
        config["metric_list"],
        params["metric_evaluation"],
    )
    logger.info("Metrics complete")
    return addresses.model_paths.inp, metrics


def check_top_level_paths(config: dict):
    """Check the top level paths (`base_dir`) in the config.

    Args:
        config (dict): The configuration.

    Raises:
        FileNotFoundError: If a top level path does not exist.
    """
    key = "base_dir"
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
    overrides = config.get("address_overrides", None)

    if not overrides:
        return config

    for key, path in overrides.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{key} not found at {path}")
        config["address_overrides"][key] = Path(path)
    return config


def check_real_network_paths(config: dict):
    """Check the paths to the real network in the config.

    Args:
        config (dict): The configuration.

    Raises:
        FileNotFoundError: If a real network path does not exist.
    """
    real = config.get("real", None)

    if not real:
        return config

    for key, path in real.items():
        if not isinstance(path, str):
            continue
        if not Path(path).exists():
            raise FileNotFoundError(f"{key} not found at {path}")
        config["real"][key] = Path(path)

    return config


def check_starting_graph(config: dict):
    """Check the starting graph in the config.

    Args:
        config (dict): The configuration.

    Raises:
        FileNotFoundError: If the starting graph path does not exist.
    """
    # If no starting graph, return
    if not config.get("starting_graph", None):
        return config

    # Check the starting graph exists and convert to Path
    config["starting_graph"] = Path(config["starting_graph"])
    if not config["starting_graph"].exists():
        raise FileNotFoundError(
            f"""starting_graph not found at 
                                {config['starting_graph']}"""
        )

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
    for category, overrides in config.get("parameter_overrides", {}).items():
        if category not in params:
            raise ValueError(
                f"""{category} not a category of parameter. Must
                             be one of {params.keys()}."""
            )

        # Get the available properties for a category
        cat_properties = params[category].model_json_schema()["properties"]

        for key, val in overrides.items():
            # Check that the parameter is available
            if key not in cat_properties:
                raise ValueError(f"{key} not found in {category}.")

    return config


def check_and_register_custom_graphfcns(config: dict):
    """Check, register and validate custom graphfcns in the config.

    Args:
        config (dict): The configuration.

    Raises:
        ValueError: If a graphfcn module does not exist.
        ValueError: If a custom graphfcn is not successfully registered.
    """
    for custom_graphfcn_module in config.get("custom_graphfcn_modules", []):
        custom_graphfcn_module = Path(custom_graphfcn_module)

        # Check that the custom graphfcn exists
        if not custom_graphfcn_module.exists():
            raise FileNotFoundError(
                f"Custom graphfcn not found at {custom_graphfcn_module}"
            )

        # Import the custom graphfcn module
        spec = importlib.util.spec_from_file_location(  # type: ignore[attr-defined]
            custom_graphfcn_module.stem, custom_graphfcn_module
        )
        custom_graphfcn_module = importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
        spec.loader.exec_module(custom_graphfcn_module)

    # Validate the import
    validate_graphfcn_list(config.get("graphfcn_list", []))

    return config


def check_and_register_custom_metrics(config: dict):
    """Check, register and validate custom metrics in the config.

    Args:
        config (dict): The configuration.

    Raises:
        ValueError: If the custom metrics module does not exist.
    """
    for custom_metric_module in config.get("custom_metric_modules", []):
        custom_metric_module = Path(custom_metric_module)

        # Check that the custom graphfcn exists
        if not custom_metric_module.exists():
            raise FileNotFoundError(
                f"Custom graphfcn not found at {custom_metric_module}"
            )

        # Import the custom graphfcn module
        spec = importlib.util.spec_from_file_location(  # type: ignore[attr-defined]
            custom_metric_module.stem, custom_metric_module
        )
        custom_metric_module = importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
        spec.loader.exec_module(custom_metric_module)

    # Validate metric list
    validate_metric_list(config.get("metric_list", []))

    return config


def save_config(config: dict, config_path: Path):
    """Save the configuration to a file.

    Args:
        config (dict): The configuration.
        config_path (Path): The path to save the configuration.
    """
    yaml_dump(config, config_path.open("w"))


def load_config(
    config_path: Path = Path(__file__).parent / "defs" / "demo_config.yml",
    validation: bool = True,
    schema_fid: Path | None = None,
):
    """Load, validate, and convert Paths in a configuration file.

    Note, if using a custom graphfcn, load_config must be called with validation=True.

    Args:
        config_path (Path): The path to the configuration file.
        validation (bool, optional): Whether to validate the configuration.
            Defaults to True.
        schema_fid (Path, optional): The path to the schema file. Defaults to
            None.

    Returns:
        dict: The configuration.
    """
    # Load the schema
    schema_fid = (
        Path(__file__).parent / "defs" / "schema.yml"
        if schema_fid is None
        else Path(schema_fid)
    )
    schema = yaml_load(schema_fid.read_text())

    # Load the config
    config = yaml_load(config_path.read_text())

    if not validation:
        return config

    # Validate the config
    jsonschema.validate(instance=config, schema=schema)

    # Check top level paths
    config = check_top_level_paths(config)

    # Check address overrides
    config = check_address_overrides(config)

    # Check real network paths
    config = check_real_network_paths(config)

    # Check starting graph
    config = check_starting_graph(config)

    # Check parameter overrides
    config = check_parameter_overrides(config)

    # Check and register custom metrics
    config = check_and_register_custom_metrics(config)

    # Check custom graphfcns
    config = check_and_register_custom_graphfcns(config)

    return config


def run(
    model: Path,
    reporting_iters: int = 50,
    duration: int = 86400,
    storevars: list[str] = ["flooding", "flow"],
):
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
            "flooding": {"class": pyswmm.Nodes, "id": "_nodeid"},
            "depth": {"class": pyswmm.Nodes, "id": "_nodeid"},
            "flow": {"class": pyswmm.Links, "id": "_linkid"},
            "runoff": {"class": pyswmm.Subcatchments, "id": "_subcatchmentid"},
        }

        results_list = []
        for var, info in variables.items():
            if var not in storevars:
                continue
            # Rather than calling eg Nodes or Links, only call them if they
            # are needed for storevars because they carry a significant
            # overhead
            pobjs = info["class"](sim)
            results_list += [
                {"object": x, "variable": var, "id": info["id"]} for x in pobjs
            ]

        # Iterate the model
        results = []
        t_ = sim.current_time
        ind = 0
        logger.info(f"Starting simulation for: {model}")

        progress_bar = tqdm(total=duration, disable=not verbose())

        offset = 0
        while (
            (offset <= duration)
            & (sim.current_time < sim.end_time)
            & (not sim._terminate_request)
        ):
            progress_bar.update((sim.current_time - t_).total_seconds() - offset)
            offset = (sim.current_time - t_).total_seconds()

            ind += 1

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
                results.append(
                    {
                        "date": sim.current_time,
                        "value": getattr(storevar["object"], storevar["variable"]),
                        "variable": storevar["variable"],
                        "id": getattr(storevar["object"], storevar["id"]),
                    }
                )

    logger.info("Model run complete.")
    return pd.DataFrame(results)
