# A data-driven framework for dimensionality reduction and causal inference in climate fields

Fabrizio Falasca, Pavel Perezhogin, Laure Zanna

Published version: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.109.044202

(The published verson is behind a paywall. Here a preprint version in case: https://arxiv.org/abs/2306.14433)

Contact: fabri.falasca@nyu.edu

We propose a data-driven method to 

(a) Coarse-grain spatiotemporal climate fields into a set of regional modes of variability. The dimensionality reduction step is based on community detection. We further contribute to community detection in climate data by proposing a simple heuristics to identify local communities/patterns in longitude-latitude space. The communities are found using the Infomap algorithm based on the Map Equation. See for example: https://www.mapequation.org/publications.html#Rosvall-Axelsson-Bergstrom-2009-Map-equation

(b) Causal inference between time series.
Causality is inferred through the fluctuation-dissipation theory as first shown in here: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043436 
We further contribute to the framework by proposing an analytical null model for the fluctuation-dissipation relation (FDR). 

We note that the implementation proposed here has not been optimized: it works just fine in low-dimensional spaces. For very high-dimensional systems there may be other ways to implement a faster version of the method.

### A few qualitative consideration on the application of the Fluctuation-Dissipation Theorem for climate data

(a) See Appendix A of this paper https://arxiv.org/pdf/2408.12585 for some qualitative considerations on the application of FDT in chaotic systems

(b) Coarse-graining is a fundamental step in applications of FDT in Climate and not just a mere technical detail (see for example Colangeli et al. (2012) https://iopscience.iop.org/article/10.1088/1742-5468/2012/04/L04002). Our paper offers 1 way to coarse-grain a spatiotemporal field, but there are many others, going back to more traditional ones like EOFs. An example of successful application of FDT using EOFs can be found in this paper by Gritsun et al. https://doi.org/10.1175/JAS3943.1 Coarse-graining is necessary and the way you coarse-grain will impact the final results: so much thinking/tests should be spent on this step.

(c) The framework proposed here tackles the problem of causal inference, i.e. identifying causal-effect mechanisms by quantifying how perturbations travel along a system. Many other tools focus in the literature instead on the problem of causal discovery aiming instead in reconstructing a causal graph from observations.

Finally, a few mathematical guidances on the application of FDT in climate data, together with tests on simple nonlinear systems can be found in Majda et al (2010) https://journals.ametsoc.org/view/journals/atsc/67/4/2009jas3264.1.xml

### Packages needed
Main packages needed here are 

- "Infomap". See this page for installation: https://www.mapequation.org/infomap/ ; Download Infomap version 2.2.0
- NetworkX. See https://networkx.org/ ; Download NetworkX version 2.8
- netCDF4. See https://unidata.github.io/netcdf4-python/

##### Note: In the Synthetic folder you find a notebook with synthetic tests. This allows to understand and test the causal framework adopted here. For this none of the packages listed above are needed.

## Folders


### Synthetic. (For the causal inference step, this is all you need.)
In the main.ipynb notebook we show how to infer causality between a set of time series using the FDR formalism. In the notebook "explaining.ipynb" we provide step-by-steps examples explaining the formalism. 

- These codes are general and can be used on other sets of time series. If you already have a set of time series and want to study their causal relationships, the code main.ipynb in this folder is all you need.

- 

### Global_SST
All codes to (a) dimensionality reduction and (b) causal inference of global sea surface temperature data. All results of the paper are reported in the notebook "main.ipynb".

### Preprocessing
Here we give an example of the main preprocessing. To preprocess the data we use the CDO package https://code.mpimet.mpg.de/projects/cdo 
The preprocessing file is in bash. To evaluate just type: sh preprocess.sh
