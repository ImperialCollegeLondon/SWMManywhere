# Graph functions guide

SWMManywhere works by starting with a graph of plausible pipe locations
(typically the street network) and iteratively applying functions to transform
that network gradually into a UDM. A graph function is actually a class, of type
[`BaseGraphFunction`](reference-graph-utilities.md#BaseGraphFunction), that can
be called with a function that takes a graph (and some arguments) and returns
an updated graph.

Let's look at a very simple `graphfcn`, which is simply a wrapper for
[`networkx.to_undirected`](https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.to_undirected.html).

::: swmmanywhere.graph_utilities.to_undirected

## Registering graph functions

## Iterate graph functions

## Add a new graph function
