# Causal inference in spatiotemporal climate fields through linear response theory

Fabrizio Falasca, Pavel Perezhogin, Laure Zanna
Contact: fabri.falasca@nyu.edu

We propose a data-driven method to 

(a) Coarse-grain spatiotemporal climate fields into a set of regional modes of variability. The dimensionality reduction step is based on community detection. We further contribute to community detection in climate data by proposing a simple heuristics to identify local communities/patterns in longitude-latitude space. The communities are found using the Infomap algorithm based on the Map Equation. See for example: https://www.mapequation.org/publications.html#Rosvall-Axelsson-Bergstrom-2009-Map-equation

(b) Causal inference between time series.
Causality is inferred through the fluctuation-dissipation theory as first shown in here: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043436 
We further contribute to the framework by proposing an analytical null model for the fluctuation-dissipation relation (FDR). 

### Packages needed
Main packages needed here are 

- "Infomap". See this page for installation: https://www.mapequation.org/infomap/
- NetworkX. See https://networkx.org/
- netCDF4. See https://unidata.github.io/netcdf4-python/

##### Note: In the Synthetic folder you find a notebook with synthetic tests. This allows to understand and test the causal framework adopted here. For this none of the packages listed above are needed.

## Folders


### Synthetic
In the linear_response.ipynb notebook we provide examples and codes on how to infer causality between time series using the FDR formalism. Moreover, these codes are general and can be used on other sets of time series.

### Global_SST
All codes to (a) dimensionality reduction and (b) causal inference of global sea surface temperature data. All results of the paper are reported in the notebook "infomap_community_Global_q095_original_monthly.ipynb".

### Tropical_Pacific_SST
All codes to (a) dimensionality reduction and (b) causal inference of sea surface temperature data in the tropical Pacific. All results of the paper are reported in the notebook "infomap_community_Pacific_q095_original_weekly.ipynb".

### Preprocessing
Here we give an example of the main preprocessing. To preprocess the data we use the CDO package https://code.mpimet.mpg.de/projects/cdo 
The preprocessing file is in bash. To evaluate just type: sh preprocess.sh

### testing_responses
This is Pavel trying to write faster and faster algorithms for this problem. He succeeded, making this code as fast as the Спутник 1 before leaving us for Space Exploration.
