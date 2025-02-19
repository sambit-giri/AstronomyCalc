---
title: 'AstronomyCalc: A python toolkit for teaching Astronomical Calculations and Data Analysis methods'
tags:
  - Python
  - astronomy
  - cosmology
  - teaching
  - dark matter
  - galaxies
authors:
  - name: Sambit K. Giri
    orcid: 0000-0002-2560-536X
    affiliation: "1" 
affiliations:
 - name: Nordita, KTH Royal Institute of Technology and Stockholm University, Hannes Alfvéns väg 12, SE-106 91 Stockholm, Sweden
   index: 1
date: 26 August 2024
bibliography: paper.bib

---

# Summary

Understanding astrophysical and cosmological processes can be challenging due to their complexity and the lack of simple, everyday analogies. To address this, we present `AstronomyCalc`, a user-friendly Python package designed to facilitate the learning of these processes and help develop insights based on the variation theory of learning [@lo2011towards;@ling2012variation].

`AstronomyCalc` enables students and educators to engage with key astrophysical and cosmological calculations, such as solving the Friedmann equations, which are fundamental to modeling the dynamics of the universe. The package allows users to construct and explore various cosmological models, including the de Sitter and Einstein-de Sitter universes [see @ryden2017introduction for more examples], by adjusting key parameters such as matter density and the Hubble constant. This interactive approach helps users intuitively grasp how variations in these parameters affect properties like expansion rates and cosmic time evolution. Additionally, the package is designed to be easily expanded with additional astronomical calculations as needed for a course. 

`AstronomyCalc` also includes modules for generating synthetic astronomical data or accessing publicly available datasets. In its current version, users can generate synthetic Type Ia supernova measurements of cosmological distances [@vanderplas2012introduction] or utilize the publicly available Pantheon+ dataset [@brout2022pantheonplus]. Additionally, the package supports the download and analysis of the SPARC dataset, which contains galaxy rotation curves for 175 disk galaxies [@lelli2016sparc].

The datasets provided in the package can be analyzed within the package to test cosmological and astrophysical models, offering a hands-on experience that mirrors the scientific research process in astronomy. Simplified implementations of advanced data analysis techniques, such as Importance Sampling [@tokdar2010importance] and the Metropolis-Hastings algorithm [@robert2004metropolis], are included to introduce users to Monte Carlo Markov Chain sampling and statistical data interpretation. By integrating theoretical concepts with observational data analysis, `AstronomyCalc` not only aids in conceptual learning but also provides insights into the empirical methods used in the field.

The current version of `AstronomyCalc` contains several Jupyter notebooks featuring tutorials on various topics in astronomical and cosmological calculations and data analysis. These notebooks are designed for effortless use, either locally or through online Python environments such as [BinderHub](https://binderhub.readthedocs.io/) and [Google Colab](https://colab.research.google.com/). This flexibility allows students to engage with the tutorials in a manner that best suits their needs. Moreover, these resources can be used as templates to create customized tutorials, enabling educators to tailor content for specific courses, expand the library of tutorials, and address diverse learning objectives.

# Statement of Need

The field of astronomy and cosmology requires a deep understanding of complex processes that are often difficult to visualize or grasp through traditional learning methods. `AstronomyCalc` addresses this challenge by offering an interactive, user-friendly tool that bridges the gap between theoretical knowledge and practical application.

Designed with the variation theory of learning in mind, this package enables students and educators to experiment with and explore key astrophysical and cosmological models in an intuitive manner. By varying parameters and observing the resulting changes, users can develop a more profound understanding of the underlying physical processes.

Furthermore, `AstronomyCalc` equips users with the tools needed to analyze real or simulated astronomical data, thereby providing a comprehensive learning experience that reflects the true nature of scientific inquiry in astronomy. This makes `AstronomyCalc` an invaluable resource for education in astronomy and cosmology, enhancing both the depth and quality of learning in these fields.

# Acknowledgements

We acknowledge the higher education teaching and learning courses offered by Stockholm University. Nordita is supported in part by NordForsk.

# References
