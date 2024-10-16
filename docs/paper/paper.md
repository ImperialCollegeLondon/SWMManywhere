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

Urban drainage network models (UDMs) are useful for a variety of purposes, most notably simulating and alleviating pluvial flooding. Despite the clear usefulness of UDMs, they are often not used owing to the difficulty and expense of creating them. This creates a significant gap for users attempting to generate UDMs if they are not able to perform an expensive survey underground survey. A variety of literature exists to derive such UDMs from GIS data [refs]. We identify some key limitations of these approaches, most notably the lack of automatic data acquisition and preprocessing, that all approaches are closed-source to date, and that a key feature of such an approach should be to facilitate extension and customisation.

# Statement of need

Existing methods for synthesizing urban drainage networks have advanced significantly, producing hydraulically feasible models that closely approximate real-world systems. However, these methods face three critical limitations. First, many are closed-source, restricting accessibility and limiting transparency for users and researchers. Second, they typically require users to source, preprocess, and input geospatial data manually, which creates a high technical barrier, especially in regions where data acquisition is challenging. Third, while the output of these methods is often sensitive to input parameters, they offer limited control over the generation process, making it difficult to adjust and refine models according to specific needs. This combination of complexity and lack of control can lead to significant uncertainty in outputs, which remains difficult to manage.

SWMManywhere addresses these limitations by offering an open-source, globally applicable tool that automates data acquisition and preprocessing while providing users with a flexible, customizable workflow. Its parameterized design allows for detailed sensitivity analysis [ref], enabling users to better understand and manage uncertainties in urban drainage network synthesis. Unlike existing methods, SWMManywhere emphasizes user control, allowing for fine-tuning of outputs to meet specific project requirements. By integrating publicly available geospatial data, the tool minimizes the technical burden and ensures applicability in data-scarce regions, making it an ideal solution for both researchers and practitioners in urban water management.

## Outlook

While we believe that SWMManywhere is a useful tool it has a variety of current limitations that present an exciting outlook for future research. Foremost, based on the findings of a sensitivity analysis [ref], better identification of manhole locations and outfalls will be critical to

# Acknowledgements

BD is funded through the Imperial College Research Fellowship scheme, which also funded the software development. We acknowledge computational resources and support provided by the [Imperial College Research Computing Service](http://doi.org/10.14469/hpc/2232).

# References
