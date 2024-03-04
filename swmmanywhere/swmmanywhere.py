# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
from pathlib import Path

import pandas as pd
import pyswmm


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