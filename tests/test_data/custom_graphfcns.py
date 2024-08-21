from __future__ import annotations

import networkx as nx

from swmmanywhere.graph_utilities import BaseGraphFunction, register_graphfcn


@register_graphfcn
class new_graphfcn(BaseGraphFunction, adds_edge_attributes=["new_attrib"]):
    """New graphfcn class."""

    def __call__(self, G: nx.graph, **kwargs) -> nx.Graph:
        """Adds new_attrib to the graph."""
        G = G.copy()
        nx.set_edge_attributes(G, "new_value", "new_attrib")
        return G
