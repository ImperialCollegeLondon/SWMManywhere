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
  - family-names: "Dobson"
    given-names: "Barnaby"
    orcid: "https://orcid.org/0000-0002-0149-4124"
    affiliation: 1
  - family-names: "Alonso-Álvarez"
    given-names: "Diego"
    orcid: "https://orcid.org/0000-0002-0060-9495"
    affiliation: 1
  - family-names: "Chegini"
    given-names: "Taher"
    orcid: "https://orcid.org/0000-0002-5430-6000"
    affiliation: 2
affiliations:
 - name: Imperial College London, UK
   index: 1
 - name: Purdue University, US
   index: 2
date: 16 October 2024
bibliography: paper.bib

---

# Summary

Urban drainage network models (UDMs) are useful for a variety of purposes, most notably simulating and alleviating pluvial flooding. Despite the clear usefulness of UDMs, they are often not used owing to the difficulty and expense of creating them. This creates a significant gap for users attempting to generate UDMs if they are not able to perform an expensive underground survey. A variety of literature exists to derive such UDMs from GIS data, producing hydraulically feasible models that closely approximate real-world systems [@Blumensaat2012-hd;@Chahinian2019-lg;@Reyes-Silva2022-pr]. We identify some key limitations of these approaches, most notably the lack of automatic data acquisition and preprocessing, that all approaches are closed-source to date, and that a key feature of such an approach should be to facilitate extension and customisation.

# Statement of need

`SWMManywhere` is an open-source Python package designed for the global synthesis of urban drainage networks. SWMManywhere integrates publicly available geospatial data and automates data acquisition and preprocessing, reducing the technical burden on users. The CLI offers a flexible workflow, providing an accessible entry point to using and customising synthesis. Its parameterized design enables detailed sensitivity analyses, allowing users to understand and manage uncertainties inherent in urban drainage modelling [@Dobson2024-dv]. By emphasizing user control, SWMManywhere allows tuning of outputs with parameters to meet local requirements, making it adaptable to a wide range of scenarios. Designed for both researchers and practitioners in urban water management, SWMManywhere responds to the limitations of existing methods by providing an end-to-end, open-source, and customisable solution.

## Outlook

While we believe that SWMManywhere is a useful tool it has a variety of current limitations that present an exciting outlook for future research. A variety of improvements to the overall realism of the approach may be made in the future.

- Foremost, based on the findings of a sensitivity analysis [@Dobson2024-dv], better identification of manhole locations and outfalls will be critical to narrowing uncertainty in simulation outputs and improving realism.
- Implementation of local design regulations is common practice in UDM synthesis [@Chegini2022-oo], but not implemented in SWMManywhere owing to its global scope. Users may customise design parameters, but implementing a global database of design parameters tailored to different regions may also be possible.
- Capturing the gradual evolution of a network over time is known to be important in UDM synthesis [@Rauch2017-jz], and further illustrated by SWMManywhere results [@Dobson2024-dv]. We do not know of a global database that provides the information that would be necessary to capture this, but it may exist in the future or for local applications.
- Hydraulic structures often dominate the behaviour of a drainage network [@Dobson2022-wq;@Thrysoe2019-pi], but are not accounted for by SWMManywhere. We envisage that a machine learning approach may be able to improve identification of structures such as weirs, while elements such as pumps have been incorporated into network synthesis for sanitary networks [@Khurelbaatar2021-sp].

# Acknowledgements

BD is funded through the Imperial College Research Fellowship scheme, which also funded the software development. We acknowledge computational resources and support provided by the [Imperial College Research Computing Service](http://doi.org/10.14469/hpc/2232).

# References
