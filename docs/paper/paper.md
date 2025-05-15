---
title: 'SWMManywhere: Synthesise Urban Drainage Network Models Anywhere in the World'
tags:
  - python
  - stormwater
  - hydrology-stormwater-analysis
  - swmm5
  - swmm
  - hydraulic-modelling
authors:
  - name: "Barnaby Dobson"
    orcid: "0000-0002-0149-4124"
    affiliation: 1
  - name: "Diego Alonso-Álvarez"
    orcid: "0000-0002-0060-9495"
    affiliation: 1
  - name: "Taher Chegini"
    orcid: "0000-0002-5430-6000"
    affiliation: 2
affiliations:
 - name: Imperial College London, UK
   index: 1
 - name: Purdue University, US
   index: 2
date: 16 October 2024
bibliography: paper.bib
---

## Summary

Urban drainage network models (UDMs) contain pipe and manhole information for drainage networks in urban areas. When driven by precipitation timeseries data, they can be used to simulate the flow of water through the network, which is useful for a variety of purposes, most notably simulating and alleviating pluvial flooding. Despite the clear usefulness of UDMs, they are often not used owing to the difficulty and expense of creating them. This creates a significant gap for users attempting to generate UDMs if they are not able to perform an expensive underground survey.
SWMManywhere automates the full workflow of UDM synthesis, from global data acquisition and preprocessing through to hydraulically plausible model generation. No previous expertise with hydraulic modelling is required, only a bounding box is needed, however, users may tune a variety of parameters to better understand UDM synthesis for their case [@Dobson2025].

## Statement of need

A variety of literature exists to derive UDMs from GIS data, producing hydraulically feasible models that closely approximate real-world systems [@Blumensaat2012-hd;@Chahinian2019-lg;@Reyes-Silva2022-pr;@Chegini2022-oo]. We identify some key limitations of these approaches, most notably the lack of automatic data acquisition and preprocessing, that all approaches are closed-source to date, and that a key feature of such an approach should be to facilitate extension and customisation. An open-source approach exists for sanitary sewer systems, however it does not provide automatic data acquisition [@sanne2024pysewer].

SWMManywhere is an open-source Python package designed for the global synthesis of urban drainage networks. SWMManywhere integrates publicly available geospatial data and automates data acquisition and preprocessing, reducing the technical burden on users. Designed for both researchers and practitioners in urban water management, SWMManywhere responds to the limitations of existing methods by providing an end-to-end, open-source, and customisable solution. Although SWMManywhere has been used in research applications [@Dobson2025], currently missing is a description focussed on the software implementation and key features, which we provide below.

## Features

SWMManywhere includes a variety of key features aimed to improve useability and usefulness. A command line interface (CLI) offers a flexible workflow, providing an accessible entry point to using and customising synthesis. Its parameterized design enables detailed sensitivity analyses, allowing users to understand and manage uncertainties inherent in urban drainage modelling [@Dobson2025]. By emphasizing user control, SWMManywhere allows tuning of outputs with parameters to meet local requirements, making it adaptable to a wide range of scenarios. We provide further details on the data and general approach below.

### Data

A variety of datasets were selected to enable SWMManywhere to be applied globally, \autoref{table:table1}.

: SWMManywhere data sources. \label{table:table1}

| Data Source | Description | Reference |
|-------------|-------------| --------- |
| **OpenStreetMap (OSM)** | Provides global street and river data, used to define potential pipe locations and outfall points for drainage networks. | [@Boeing2017;@OpenStreetMap] |
| **Google-Microsoft Open Buildings** | A dataset of global building footprints, used for estimating impervious surfaces essential for runoff calculations. | [@OpenStreetMap-overture;@VIDA2023] |
| **NASADEM** | Provides 30m resolution global digital elevation model (DEM) data to support sub-catchment delineation and slope calculation. | [@Crippen2016] |

These datasets are global in their coverage, and we consider them of sufficient quality in locations that we have tested [@Dobson2025], however, we urge users to check data in their specific case study.

### Approach and customisation

The core task in SWMManywhere is to begin with a 'starting graph' (e.g., an OSM street graph), refine this graph first into manhole locations and potential pipe locations, eliminate pipes from unnecessary locations, and then dimension the resulting pipe network which is then simulated in the software [SWMM](https://www.epa.gov/sites/default/files/2019-02/documents/epaswmm5_1_manual_master_8-2-15.pdf) using the `pyswmm` package [@mcdonnell2020pyswmm]. These operations take place in an iterative approach, where each function takes a graph, and returns the transformed graph, thus each operation is referred to as a 'graph function'. The use of graph functions in SWMManywhere enables modular packaging of functions, easy customisation of the approach (e.g., by adding/removing/reordering graph functions), and explicit definition of parameters for each graph function. Explanations for making these customisations are available in the [documentation](https://imperialcollegelondon.github.io/SWMManywhere/). Ultimately, this customisability facilitates exploring uncertainty in urban drainage modelling in a way that reflects not just the model itself but the model creation process, as is demonstrated in [@Dobson2025].

We visualise the example from the [extended demonstration](https://imperialcollegelondon.github.io/SWMManywhere/notebooks/extended_demo/) in the documentation to illustrate how changing relatively few parameter values in a strategic way can dramatically change the nature of the synthesised network \autoref{fig:fig1}.

![Example of output customisation with SWMManywhere. Black nodes are manholes, black lines are pipes, red nodes are outfalls.\label{fig:fig1}](extended_demo.png)

### Comparisons with real networks

Because manhole and pipe locations rarely coincide between a synthetic and survey-derived UDM, direct element-by-element comparisons are infeasible [@Chahinian2019-lg]. SWMManywhere therefore implements a comprehensive suite of metrics mapped to each stage of the synthesis pipeline: system description (e.g., pipe lengths, counts), topological structure (e.g., Laplacian and vertex–edge distances, betweenness distributions), hydraulic design (e.g., diameter and depth statistics), and simulation performance (e.g., outfall flows and network flooding volumes). This multi‐tiered approach validates model realism against observed networks and facilitates understanding of where in the SWMManywhere workflow the synthetic and real UDM diverge, thereby transforming UDM synthesis into an explicit, uncertainty‐driven workflow.

## Outlook

While we believe that SWMManywhere is a useful tool it has a variety of current limitations that present an exciting outlook for future research. Key improvements to the overall realism of the approach may be made in the future, in particular,

- Based on the findings of a sensitivity analysis [@Dobson2025], better identification of manhole locations and outfalls will be critical to narrowing uncertainty in simulation outputs and improving realism.
- Capturing the gradual evolution of a network over time is known to be important in UDM synthesis [@Rauch2017-jz], and further illustrated by SWMManywhere results [@Dobson2025]. We do not know of a global database that provides the information that would be necessary to capture this, but it may exist in the future or for local applications.

# Acknowledgements

BD is funded through the Imperial College Research Fellowship scheme, which also funded the software development. We acknowledge computational resources and support provided by the [Imperial College Research Computing Service](http://doi.org/10.14469/hpc/2232).

# References
