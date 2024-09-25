from __future__ import annotations

import networkx as nx

from swmmanywhere.metric_utilities import metrics


@metrics.register
def new_metric(synthetic_G: nx.Graph, real_G: nx.Graph, **kwargs) -> float:
    """New metric function."""
    return len(synthetic_G.edges) / len(real_G.edges)
