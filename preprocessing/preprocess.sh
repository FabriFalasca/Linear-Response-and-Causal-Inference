#!/bin/bash

########## Example of preprocessing via CDO
# Variable "tos" is the sea surface temperature
# We start from a 650 years long simulation at daily temporal resolution...

# Consider the last 300 years of the simulation
cdo selyear,350/650 ./tos.nc tos_300yrs.nc

# Bilinear remapping to 1 by 1 degree
cdo -L remapbil,r360x180 tos_300yrs.nc tos_300yrs_1deg.nc

# Remove high latitudes
cdo sellonlatbox,180,-180,-60,60 tos_300yrs_1deg.nc tos_300yrs_1deg_nohl.nc

# Remove daily climatology
cdo -L -ydaysub tos_300yrs_1deg_nohl.nc -ydaymean tos_300yrs_1deg_nohl.nc tos_300yrs_1deg_nohl_a.nc

# Compute the monthly average
cdo monmean tos_300yrs_1deg_nohl_a.nc tos_300yrs_1deg_nohl_a_monthly.nc

########## High-pass filtering of the data

# Set the land to 0 before high-pass filtering (needed when using CDO)
cdo setmisstoc,0 tos_300yrs_1deg_nohl_a_monthly.nc tos_300yrs_1deg_nohl_a_monthly_anomalies_new_land0.nc

# High pass filtering with f = 1/10 years
cdo highpass,0.1 tos_300yrs_1deg_nohl_a_monthly_anomalies_new_land0.nc tos_300yrs_1deg_nohl_a_monthly_anomalies_new_land0_filter10yr.nc

