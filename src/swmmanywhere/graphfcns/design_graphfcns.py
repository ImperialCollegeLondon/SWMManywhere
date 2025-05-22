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

        edge_designs: dict[str, dict[tuple[Hashable, Hashable, int], float]] = {
            parameter: {} for parameter in hydraulic_design.edge_design_parameters
        }

        # Iterate over nodes in topological order
        for node in tqdm(topological_order, disable=not verbose()):
            # Check if there's any nodes upstream, if not set the depth to min_depth
            if len(nx.ancestors(G, node)) == 0:
                chamber_floor[node] = (
                    surface_elevations[node] - hydraulic_design.min_depth
                )

            self.process_successors(
                G,
                node,
                surface_elevations,
                chamber_floor,
                edge_designs,
                hydraulic_design,
            )

        for parameter in hydraulic_design.edge_design_parameters:
            nx.function.set_edge_attributes(G, edge_designs[parameter], parameter)

        nx.function.set_node_attributes(G, chamber_floor, "chamber_floor_elevation")
        return G

    @staticmethod
    def get_designs(hydraulic_design: parameters.HydraulicDesign) -> product:
        """Get the designs for the pipe.

        This function generates a grid of designs for the pipe based on the
        diameters and depths specified in the hydraulic design parameters. It
        returns an iterable product of the designs.

        Args:
            hydraulic_design (parameters.HydraulicDesign): A HydraulicDesign parameter
                object

        Returns:
            product: An iterable product object containing the designs
        """
        return product(
            hydraulic_design.diameters,
            np.linspace(
                hydraulic_design.min_depth,
                hydraulic_design.max_depth,
                hydraulic_design.depth_nbins,
            ),
        )

    @staticmethod
    def calculate_cost(V: float, diam: float) -> float:
        """Calculate the cost of the pipe.

        This function calculates the cost of the pipe based on the excavation volume and
        diameter.

        Cost equation from: https://doi.org/10.2166/hydro.2016.105

        Args:
            V (float): The excavation volume of the pipe
            diam (float): The diameter of the pipe
        Returns:
            float: The cost of the pipe in USD
        """
        return 1.32 / 2000 * (9579.31 * diam**0.5737 + 1163.77 * V**1.31)

    def evaluate_design(
        self,
        ds_elevation: float,
        chamber_floor: float,
        edge_length: float,
        hydraulic_design: parameters.HydraulicDesign,
        Q: float,
        diam: float,
        depth: float,
    ) -> dict[Hashable, float]:
        """Evaluate the design of a pipe.

        This function evaluates the design of a pipe by calculating the cost,
        velocity, filling ratio, and feasibility parameters. It returns a dictionary
        containing the design parameters and their values.

        Args:
            ds_elevation (float): The downstream elevation
            chamber_floor (float): The elevation of the chamber floor
            edge_length (float): The length of the edge
            hydraulic_design (parameters.HydraulicDesign): A HydraulicDesign parameter
                object
            Q (float): The flow rate
            diam (float): The diameter of the pipe
            depth (float): The depth of the pipe

        Returns:
            dict: A dictionary containing the designed pipe's parameters and values
        """
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
        cost = self.calculate_cost(V, diam)
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
        return {
            "diameter": diam,
            "depth": depth,
            "slope": slope,
            "v": v,
            "fr": filling_ratio,
            # 'tau' : shear_stress,
            "cost_usd": cost,
            "v_feasibility": v_feasibility,
            "fr_feasibility": fr_feasibility,
            "surcharge_feasibility": surcharge_feasibility,
            # 'shear_feasibility' : shear_feasibility
        }

    def select_design(self, pipes_df: pd.DataFrame) -> dict[Hashable, float]:
        """Select the ideal design from the dataframe.

        This function selects the ideal design from the dataframe by sorting the
        dataframe based on the feasibility parameters and cost. It returns the
        diameter, depth, and cost of the ideal design.

        Args:
            pipes_df (pd.DataFrame): A dataframe containing the designs and their
                parameters

        Returns:
            dict: A dictionary containing the ideal design
        """
        if pipes_df.shape[0] <= 0:
            raise ValueError("No non nan pipes designed. Shouldn't happen.")

        ideal_pipe = pipes_df.sort_values(
            by=[
                "surcharge_feasibility",
                "v_feasibility",
                "fr_feasibility",
                # 'shear_feasibility',
                "depth",
                "cost_usd",
            ],
            ascending=True,
        ).iloc[0]

        return ideal_pipe.to_dict()

    def design_pipe(
        self,
        ds_elevation: float,
        chamber_floor: float,
        edge_length: float,
        hydraulic_design: parameters.HydraulicDesign,
        Q: float,
    ) -> dict[Hashable, float]:
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
            dict: A dictionary containing the selected design
        """
        # Generate designs
        designs = self.get_designs(hydraulic_design)

        # Evaluate designs
        pipes = [
            self.evaluate_design(
                ds_elevation,
                chamber_floor,
                edge_length,
                hydraulic_design,
                Q,
                diam,
                depth,
            )
            for diam, depth in designs
        ]

        # Return selected design
        return self.select_design(pd.DataFrame(pipes).dropna())

    @staticmethod
    def calculate_flow(G: nx.Graph, node: Hashable, design_precipitation) -> float:
        """Calculate the flow to a node.

        This function calculates the flow to a node by summing the flow from its
        predecessors. It returns the total flow to the node.

        Args:
            G (nx.Graph): A graph
            node (Hashable): A node
            design_precipitation (float): The design precipitation

        Returns:
            float: The total flow to the node
        """
        # Find contributing area with ancestors
        # TODO - could do timearea here if i hated myself enough
        anc = nx.ancestors(G, node).union([node])
        tot = sum([G.nodes[anc_node]["contributing_area"] for anc_node in anc])
        M3_PER_HR_TO_M3_PER_S = 1 / 60 / 60

        return tot * design_precipitation * M3_PER_HR_TO_M3_PER_S

    def process_successors(
        self,
        G: nx.Graph,
        node: Hashable,
        surface_elevations: dict[Hashable, float],
        chamber_floor: dict[Hashable, float],
        edge_designs: dict[str, dict[tuple[Hashable, Hashable, int], float]],
        hydraulic_design: parameters.HydraulicDesign,
    ) -> None:
        """Process the successors of a node.

        This function processes the successors of a node. It designs a pipe to the
        downstream node and sets the diameter and downstream invert level of the
        pipe. It also sets the downstream invert level of the downstream node. It
        returns None but modifies the hydraulic_design.edge_design_parameters entries
        inside edge_designs and node chamber_floor dictionaries.

        Args:
            G (nx.Graph): A graph
            node (Hashable): A node
            surface_elevations (dict): A dictionary of surface elevations keyed by
                node
            chamber_floor (dict): A dictionary of chamber floor elevations keyed by
                node
            edge_designs (dict): A dictionary of pipe designs keyed by parameter and
                then edge
            hydraulic_design (parameters.HydraulicDesign): A HydraulicDesign parameter
                object
        """
        Q = self.calculate_flow(G, node, hydraulic_design.precipitation)

        for ix, ds_node in enumerate(G.successors(node)):
            edge = G.get_edge_data(node, ds_node, 0)

            # Design the pipe to find the diameter and invert depth
            pipe = self.design_pipe(
                surface_elevations[ds_node],
                chamber_floor[node],
                edge["length"],
                hydraulic_design,
                Q,
            )

            for parameter in hydraulic_design.edge_design_parameters:
                edge_designs[parameter][(node, ds_node, 0)] = pipe[parameter]

            chamber_floor[ds_node] = min(
                surface_elevations[ds_node] - pipe["depth"],
                chamber_floor.get(ds_node, np.inf),
            )
            if ix > 0:
                logger.warning(
                    """a node has multiple successors, 
                    not sure how that can happen if using shortest path
                    to derive topology"""
                )
