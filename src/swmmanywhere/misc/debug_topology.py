# -*- coding: utf-8 -*-
"""Created 2024-03-07.

@author: Barnaby Dobson
"""

from __future__ import annotations

from time import time

import pandas as pd

from swmmanywhere import metric_utilities as mu
from swmmanywhere.prepare_data import download_street

if __name__ == "__main__":
    # Download streets
    Gs = download_street((-0.09547, 51.52196, -0.09232, 51.52422))
    Gm = download_street((-0.09547, 51.52196, -0.08876, 51.52661))
    Gl = download_street((-0.09547, 51.52196, -0.08261, 51.53097))
    Gsep = download_street((-0.08261, 51.53097, -0.06946, 51.53718))

    # Define topological metrics
    metrics = [
        "nc_deltacon0",
        "nc_laplacian_dist",
        "nc_laplacian_norm_dist",
        "nc_adjacency_dist",
        "nc_resistance_distance",
        "nc_vertex_edge_distance",
        "kstest_betweenness",
        "kstest_edge_betweenness",
    ]

    # Calculate metrics
    results = []
    for G1, l1 in zip([Gs, Gm, Gl, Gsep], ["a_small", "b_med", "c_large", "d_sep"]):
        for G2, l2 in zip([Gs, Gm, Gl, Gsep], ["a_small", "b_med", "c_large", "d_sep"]):
            for func in metrics:
                start = time()
                val_ = getattr(mu.metrics, func)(synthetic_G=G1, real_G=G2)
                end = time()
                results.append(
                    {"func": func, "val": val_, "time": end - start, "l1": l1, "l2": l2}
                )
    df = pd.DataFrame(results).sort_values(by=["l1", "l2"])

    """
    # Plot heatmap by func
    f,axs = plt.subplots(4,2,figsize=(10,10))
    for (func,grp),ax in zip(df.groupby('func'),axs.reshape(-1)):
        grp = grp.pivot(index='l1',
                    columns= 'l2', 
                    values='val')
        ax.imshow(grp,cmap='Reds')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(grp.columns,rotation=90)
        ax.set_yticklabels(grp.index)
        ax.set_title(func)
    plots = Path(__file__).parent
    f.tight_layout()
    f.savefig(plots / 'heatmap.png')
    """
