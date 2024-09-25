"""Module for graphfcns that design the pipe inverts and diameters."""

from __future__ import annotations

from itertools import product
from typing import Hashable

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from swmmanywhere import parameters
from swmmanywhere.graph_utilities import BaseGraphFunction, register_graphfcn
from swmmanywhere.logging import logger, verbose


def design_pipe(
    ds_elevation: float,
    chamber_floor: float,
    edge_length: float,
    hydraulic_design: parameters.HydraulicDesign,
    Q: float,
) -> nx.Graph:
    """Design a pipe.

    This function designs a pipe by iterating over a range of diameters and
    depths. It returns the diameter and depth of the pipe that minimises the
    cost function, while also maintaining or minimising feasibility parameters
    associated with: surcharging, velocity and filling ratio.

    Args:
        ds_elevation (float): The downstream elevationq
        chamber_floor (float): The elevation of the chamber floor
        edge_length (float): The length of the edge
        hydraulic_design (parameters.HydraulicDesign): A HydraulicDesign parameter
            object
        Q (float): The flow rate

    Returns:
        diam (float): The diameter of the pipe
        depth (float): The depth of the pipe
    """
    designs = product(
        hydraulic_design.diameters,
        np.linspace(
            hydraulic_design.min_depth, hydraulic_design.max_depth, 10
        ),  # TODO should 10 be a param?
    )
    pipes = []
    for diam, depth in designs:
        A = np.pi * diam**2 / 4
        n = 0.012  # mannings n
        R = A / (np.pi * diam)  # hydraulic radius
        # TODO... presumably need to check depth > (diam + min_depth)

        elev_diff = chamber_floor - (ds_elevation - depth)
        slope = elev_diff / edge_length
        # Always pick a pipe that is feasible without surcharging
        # if available
        surcharge_feasibility = 0.0
        # Use surcharged elevation
        while slope <= 0:
            surcharge_feasibility += 0.05
            slope = (
                chamber_floor + surcharge_feasibility - (ds_elevation - depth)
            ) / edge_length
            # TODO could make the feasibility penalisation increase
            # when you get above surface_elevation[node]... but
            # then you'd need a feasibility tracker and an offset
            # tracker
        v = (slope**0.5) * (R ** (2 / 3)) / n
        filling_ratio = Q / (v * A)
        # buffers from: https://www.polypipe.com/sites/default/files/Specification_Clauses_Underground_Drainage.pdf
        average_depth = (depth + chamber_floor) / 2
        V = edge_length * (diam + 0.3) * (average_depth + 0.1)
        cost = 1.32 / 2000 * (9579.31 * diam**0.5737 + 1153.77 * V**1.31)
        v_feasibility = max(hydraulic_design.min_v - v, 0) + max(
            v - hydraulic_design.max_v, 0
        )
        fr_feasibility = max(filling_ratio - hydraulic_design.max_fr, 0)
        """
        TODO shear stress... got confused here
        density = 1000
        dyn_visc = 0.001
        hydraulic_diameter = 4 * (A * filling_ratio**2) / \
            (np.pi * diam * filling_ratio)
        Re = density * v * 2 * (diam / 4) * (filling_ratio ** 2) / dyn_visc
        fd = 64 / Re
        shear_stress = fd * density * v**2 / fd
        shear_feasibility = max(min_shear - shear_stress, 0)
        """
        slope = (chamber_floor - (ds_elevation - depth)) / edge_length
        pipes.append(
            {
                "diam": diam,
                "depth": depth,
                "slope": slope,
                "v": v,
                "fr": filling_ratio,
                # 'tau' : shear_stress,
                "cost": cost,
                "v_feasibility": v_feasibility,
                "fr_feasibility": fr_feasibility,
                "surcharge_feasibility": surcharge_feasibility,
                # 'shear_feasibility' : shear_feasibility
            }
        )

    pipes_df = pd.DataFrame(pipes).dropna()
    if pipes_df.shape[0] > 0:
        ideal_pipe = pipes_df.sort_values(
            by=[
                "surcharge_feasibility",
                "v_feasibility",
                "fr_feasibility",
                # 'shear_feasibility',
                "depth",
                "cost",
            ],
            ascending=True,
        ).iloc[0]
        return ideal_pipe.diam, ideal_pipe.depth
    else:
        raise Exception("something odd - no non nan pipes")


def process_successors(
    G: nx.Graph,
    node: Hashable,
    surface_elevations: dict[Hashable, float],
    chamber_floor: dict[Hashable, float],
    edge_diams: dict[tuple[Hashable, Hashable, int], float],
    hydraulic_design: parameters.HydraulicDesign,
) -> None:
    """Process the successors of a node.

    This function processes the successors of a node. It designs a pipe to the
    downstream node and sets the diameter and downstream invert level of the
    pipe. It also sets the downstream invert level of the downstream node. It
    returns None but modifies the edge_diams and chamber_floor dictionaries.

    Args:
        G (nx.Graph): A graph
        node (Hashable): A node
        surface_elevations (dict): A dictionary of surface elevations keyed by
            node
        chamber_floor (dict): A dictionary of chamber floor elevations keyed by
            node
        edge_diams (dict): A dictionary of pipe diameters keyed by edge
        hydraulic_design (parameters.HydraulicDesign): A HydraulicDesign parameter
            object
    """
    for ix, ds_node in enumerate(G.successors(node)):
        edge = G.get_edge_data(node, ds_node, 0)
        # Find contributing area with ancestors
        # TODO - could do timearea here if i hated myself enough
        anc = nx.ancestors(G, node).union([node])
        tot = sum([G.nodes[anc_node]["contributing_area"] for anc_node in anc])

        M3_PER_HR_TO_M3_PER_S = 1 / 60 / 60
        Q = tot * hydraulic_design.precipitation * M3_PER_HR_TO_M3_PER_S

        # Design the pipe to find the diameter and invert depth
        diam, depth = design_pipe(
            surface_elevations[ds_node],
            chamber_floor[node],
            edge["length"],
            hydraulic_design,
            Q,
        )
        edge_diams[(node, ds_node, 0)] = diam
        chamber_floor[ds_node] = surface_elevations[ds_node] - depth
        if ix > 0:
            logger.warning(
                """a node has multiple successors, 
                not sure how that can happen if using shortest path
                to derive topology"""
            )


@register_graphfcn
class pipe_by_pipe(
    BaseGraphFunction,
    required_edge_attributes=["length"],
    required_node_attributes=["contributing_area", "surface_elevation"],
    adds_edge_attributes=["diameter"],
    adds_node_attributes=["chamber_floor_elevation"],
):
    """pipe_by_pipe class."""

    # If doing required_graph_attributes - it would be something like 'dendritic'

    def __call__(
        self, G: nx.Graph, hydraulic_design: parameters.HydraulicDesign, **kwargs
    ) -> nx.Graph:
        """Pipe by pipe hydraulic design.

        Starting from the most upstream node, design a pipe to the downstream node
        specifying a diameter and downstream invert level. A range of diameters and
        invert levels are tested (ranging between conditions defined in
        hydraulic_design). From the tested diameters/inverts, a selection is made based
        on each pipe's satisfying feasibility constraints on: surcharge velocity,
        filling ratio, (and shear stress - not currently implemented). Prioritising
        feasibility in this order it identifies pipes with the preferred feasibility
        level. If multiple pipes are feasible, it picks the lowest cost pipe. Once
        the feasible pipe is identified, the diameter and downstream invert are set
        and then the next downstream pipe can be designed.

        This approach is based on the pipe-by-pipe design proposed in:
            https://doi.org/10.1016/j.watres.2021.117903

        Args:
            G (nx.Graph): A graph
            hydraulic_design (parameters.HydraulicDesign): A HydraulicDesign parameter
                object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()
        surface_elevations = nx.get_node_attributes(G, "surface_elevation")
        topological_order = list(nx.topological_sort(G))
        chamber_floor = {}
        edge_diams: dict[tuple[Hashable, Hashable, int], float] = {}
        # Iterate over nodes in topological order
        for node in tqdm(topological_order, disable=not verbose()):
            # Check if there's any nodes upstream, if not set the depth to min_depth
            if len(nx.ancestors(G, node)) == 0:
                chamber_floor[node] = (
                    surface_elevations[node] - hydraulic_design.min_depth
                )

            process_successors(
                G, node, surface_elevations, chamber_floor, edge_diams, hydraulic_design
            )

        nx.function.set_edge_attributes(G, edge_diams, "diameter")
        nx.function.set_node_attributes(G, chamber_floor, "chamber_floor_elevation")
        return G
