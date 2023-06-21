# -*- coding: utf-8 -*-
# importing libraries

import numpy as np
import scipy.io
import numpy.ma as ma
from scipy import signal, spatial
from scipy.spatial.distance import pdist, cdist, squareform
from collections import Counter

# import utils
import utils
# import preprocessing tools
import preprocessing
# Network libraries
import networkx as nx
# Infomap
import infomap

# Fabri Falasca
# fabrifalasca@gmail.com

'''
Given time series embedded in a spatiotemporal grid,
this code find local clusters/modes based on their correlation
networks. The clustering problem is cast as a community detection
problem.

The community detection algorithm considered is 

Infomap:

- -  https://www.mapequation.org/infomap/
- -  https://www.mapequation.org/assets/publications/EurPhysJ2010Rosvall.pdf

These are the arguments

########## Main function

Inputs

(a) path to netcdf file
(b) name of the climate variable (e.g., 'sst')
(c) name for the "longitude" variable (e.g., 'lon')
(d) name for the "latitude" variable (e.g., 'lat')
(e) rand_samples: # of samples to infer the k parameter
(f) q: quantile to set the k parameter. E.g., q = 0.95 means that
       k will be chosen as the 0.95 quantile of the correlation matrix.
(g) q_distances: quantile to set the d_threshold parameter. E.g., q = 0.15 means that
       threshold will be chosen as the 0.15 quantile of the PDF of all distances.
(h) k: threshold to infer links

########## Removing noisy communities

Small communities (i.e., smaller than N nodes, with N as input parameter)
can be considered as noise.
The function "larger_communities(x,N,x_t):" remove small communities and return
a new set of communities, signals and community map.

'''


def community_detection(path, climate_variable, lon_variable, lat_variable, rand_samples, q, q_distances, k=None):

    print('Loading dataset')

    # Import data
    data = utils.load_data(path, climate_variable)
    longitudes = utils.load_data(path, lon_variable)
    latitudes = utils.load_data(path, lat_variable)

    # Dimension of the field
    dimX = np.shape(data)[2]  # Points in X axis (long)
    dimY = np.shape(data)[1]  # Points in Y axis (lat)
    dimT = np.shape(data)[0]  # Points in time
    
    print('Time series with all zeros are set to nan')
    # Check: if a grid point is masked with nan at time t it should be masked at all t
    for i in range(dimY):
        for j in range(dimX):
            if np.isnan(np.sum(data[:, i, j])):
                data[:, i, j] = np.nan
            elif np.count_nonzero(data[:,i,j]==0) == dimT:
                data[:, i, j] = np.nan

    # From mask array to np.array
    data = utils.masked_array_to_numpy(data)
    longitudes = utils.masked_array_to_numpy(longitudes)
    latitudes = utils.masked_array_to_numpy(latitudes)

    print('Standardizing each time series')
    # normalize time series to zero mean and unit variance
    data = preprocessing.remove_mean(data)
    data = preprocessing.remove_variance(data)

    # From a [dimT,dimY,dimX] array
    # to a list of (dimY x dimX) spectra of length dimV
    flat_data_masked = data.reshape(dimT, dimY*dimX).transpose()

    # indices where we have numbers (non-nans)
    # it will be useful later to assign a community label to each grid point
    indices = []
    for i in range(len(flat_data_masked)):

        # number of nans
        n_nans = np.sum(np.isnan(flat_data_masked[i]))
        # if this is equal to the length of the time series
        # then this time series is all nan and it is useless to us
        # if not, we save the index
        if n_nans < dimT:
            indices.append(i)

    indices = np.array(indices)

    flat_data = flat_data_masked[indices]

    print('Computing the k parameter')

    # Compute the k parameter
    if k == None:
        k = utils.estimate_k(flat_data, rand_samples, q)

    print('k = '+str(k))

    import networkx as nx
    # initialize network
    G = nx.Graph()
    # Number of time series
    n = np.shape(flat_data)[0]

    # Each time series constitutes a node
    G.add_nodes_from(np.arange(n))
    
    ########## Enforcing spatial contiguity
    
    # lat and lon from degrees to radians
    
    latitudes_rad = np.radians(latitudes.copy());
    longitudes_rad = np.radians(longitudes.copy());
    
    # coordinates and grid cells
    coords = [];
    grid_cell = []
    for i,lat in enumerate(latitudes_rad):
        for j,lon in enumerate(longitudes_rad):
            coords.append([lat,lon]);
            grid_cell.append([i,j]);
            
    coords = np.asarray(coords,dtype=np.float32);
    coords_notNaN = coords[indices] # consider the non-NaN values

    grid_cell = np.asarray(grid_cell,dtype = np.int32);
    grid_cell_notNaN = grid_cell[indices] # consider the non-NaN values
    
    # Distance metrics

    from sklearn.metrics import pairwise_distances
    
    if q_distances == 1:
    
        # Add links
        # We add a link between time series x(t) and y(t)
        # if their correlation r(x(t),y(t),tau=0) is
        # (a) r(x(t),y(t)) >= k
        # (b) d(x,y) <= distance_threshold
        print('No Heuristic for spatial contiguous cluster')
        print('Infer graph')

        for i in range(n):

            # Define x_i(t)
            x_i = flat_data[i]

            for j in range(i+1, n):  # (i+1) means we do not include the diagonal in the calculation

                # Define x_j(t)
                x_j = flat_data[j]

                # We draw the link x_i <-> x_j only if
                # (a) the correlation is larger than k
                # (b) the maximum correlation is at lag 0

                # Check first if r(x_i,x_j,tau = 0)>k
                # if not we already know that there is no link
                corr_lag_zero = utils.lagged_correlation(x_i, x_j, 0)[1]

                if corr_lag_zero >= k:
                
                    # in this case, add an edge
                    G.add_edge(i, j)
                    
    elif q_distances < 1:
    
        print('Computing Haversine distances')
        
        # Compute all distances
        distances = pairwise_distances(coords,metric='haversine',n_jobs = -1)

        # Focus on distances between the non-NaN points
        distances_nonNaN = np.zeros([len(indices),len(indices)])
        for i in range(len(indices)):
            distances_nonNaN[i] = distances[indices[i]][indices]
        
        # Compute the distances threshold as an high quantile
        # of the PDFs of distances
        distance_threshold = np.quantile(distances[distances>0],q_distances)
        
        print('Distance threshold set as using quantile '+str(q_distances))
        print('Distance threshold = '+str(distance_threshold))

        # Add links
        # We add a link between time series x(t) and y(t)
        # if their correlation r(x(t),y(t),tau=0) is
        # (a) r(x(t),y(t)) >= k
        # (b) d(x,y) <= distance_threshold

        print('Infer graph')

        for i in range(n):

            # Define x_i(t)
            x_i = flat_data[i]

            for j in range(i+1, n):  # (i+1) means we do not include the diagonal in the calculation

                # Define x_j(t)
                x_j = flat_data[j]

                # We draw the link x_i <-> x_j only if
                # (a) the correlation is larger than k
                # (b) the maximum correlation is at lag 0

                # Check first if r(x_i,x_j,tau = 0)>k
                # if not we already know that there is no link
                corr_lag_zero = utils.lagged_correlation(x_i, x_j, 0)[1]

                if corr_lag_zero >= k:
                
                    # if the distance is inside the circle given by "distance_threshold"...
                    if distances_nonNaN[i,j] <= distance_threshold:
                
                        # in this case, add an edge
                        G.add_edge(i, j)


    ########################## Infomap start

    print('Community detection via Infomap')

    im = infomap.Infomap("--two-level --verbose --silent")

    # Add nodes
    im.add_nodes(G.nodes)

    # Add linnks
    for e in G.edges():
        im.addLink(*e)

    # run infomap
    im.run()

    print(f"Found {im.num_top_modules} communities")

    # List of nodes ids
    nodes = []
    # List of respective community
    communities = []

    for node, module in im.modules:

        nodes.append(node)
        communities.append(module)

    partition = np.transpose([nodes, communities])

    # How many communities?
    n_com = np.max(partition[:, 1])

    ########################## Infomap end

    print('Embed communities in the map')

    # Let's embed the results on the grid
    # for now this community maps is flattened
    community_grid_flattened = flat_data_masked[:, 0]  # at time t = 0
    community_grid_flattened[indices] = partition[:, 1]  # these are the labels
    community_map = community_grid_flattened.reshape(dimY, dimX)

    # Finally, let's also save single community maps
    # Output:
    # For each domain we have a 2-Dimensional map where grid points belonging to the domain have value 1
    # points that do not belong to the domain have are nan
    single_communities = []

    for i in range(n_com):
        i = i + 1  # labels start from 1 not from 0
        community = community_map.copy()
        community[community != i] = np.nan
        community[community == i] = 1
        single_communities.append(community)
    single_communities = np.array(single_communities)

    print('Compute signals')

    # We compute the signals as the average time series inside each community
    # To do so we first weight the data with the cosine of the latitude

    # First we re-load the data (we need to as we do not want them normalized now)

    print('Load dataset again')

    # Import data
    data = utils.load_data(path, climate_variable)
    data = utils.masked_array_to_numpy(data)

    # Transform latitudes in radians
    lat = np.radians(latitudes)
    # Assign a weight to each latitude phi
    lat_weights = np.cos(lat).reshape(len(lat), 1)
    # Define the weighted domain
    weighted_communities = single_communities*lat_weights
    # Now compute the signals of each community as the cumulative anomaly inside
    average_signals = []
    cumulative_signals = []

    print('Compute average and cumulative anomalies')
    for i in range(len(weighted_communities)):
        # Extract the mode
        extract_mode = data*weighted_communities[i]
        # Compute the signal as the mean time series (nans are not considered)
        average_signal = np.nanmean(extract_mode, axis=(1, 2))
        cumulative_signal = np.nansum(extract_mode, axis=(1, 2))
        # Store the result
        average_signals.append(average_signal)
        cumulative_signals.append(cumulative_signal)

    average_signals = np.array(average_signals)
    cumulative_signals = np.array(cumulative_signals)

    return community_map, single_communities, average_signals, cumulative_signals

# Function to remove small small communities


'''
Communities with less than N nodes can be considered as noise and removed
'''

def larger_communities(x, N, x_t):

    # input:
    # - x is an array of format nlat,lon)
    # - -  where n is the number of communities
    # - - in each x[i] we have "1"s where we have communities
    # - N: minimum size of communities accepted
    # - x_t: signals

    # output:
    # - new communties
    # - new signals
    # - new community_map

    # how many communities?
    n = len(x)

    # check if you have more than N points. If less than N
    # points we do not save these communities

    # new communities
    new_x = []
    # new signals
    new_x_t = []

    for i in range(n):
        if np.nansum(x[i]) > N:
            new_x.append(x[i])
            new_x_t.append(x_t[i])
        else:
            continue

    new_x = np.array(new_x)
    new_x_t = np.array(new_x_t)

    # new community map 
    new_x_map = []

    for i in range(len(new_x)):
        new_x_map.append(new_x[i]*(i+1))

    new_x_map = np.array(new_x_map)
    new_x_map = np.nansum(new_x_map, axis=0)
    new_x_map[new_x_map == 0] = np.nan

    return new_x, new_x_t, new_x_map
