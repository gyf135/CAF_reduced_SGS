# CAF_reduced_SGS

PYTHON SCRIPT ACCOMPANYING:

W.Edeling, D. Crommelin, "Deriving reduced subgrid scale models from data", submitted to Computers & Fluids, 2019.

## Abstract
Recent years have seen a growing interest in using data-driven (machine-learning) techniques for the construction of cheap surrogate models of turbulent subgrid scale stresses. These stresses display complex spatio-temporal structures, and constitute a difficult surrogate target. In this paper we propose a data-preprocessing step, in which we derive alternative subgrid scale models which are virtually exact for a user-specified set of spatially integrated quantities of interest, i.e. for time series. The unclosed component of these new subgrid scale models is of the same size as this set of integrated quantities of interest. As a result, the corresponding training data is massively reduced in size, decreasing the complexity of the subsequent surrogate construction.

## Funding
This research is funded by the Netherlands Organization for Scientific Research (NWO) through the Vidi project "Stochastic
models for unresolved scales in geophysical flows", and from the European Union Horizon 2020 research and innovation programme under grant agreement \#800925 ([VECMA](https://www.vecma.eu/) project). 

## Reproduction of main results

### Dependencies
+ Python 3
+ Numpy
+ Scipy
+ Matplotlib
+ [h5py](https://github.com/h5py/h5py)
