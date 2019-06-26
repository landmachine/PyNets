# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
from __future__ import division
import os
import numpy as np
np.warnings.filterwarnings('ignore')
import networkx as nx
import warnings
warnings.simplefilter("ignore")


def timeout(seconds):
    """
    Timeout function for hung calculations during automated graph analysis.
    """
    from functools import wraps
    import errno
    import os
    import signal

    class TimeoutError(Exception):
        pass

    def decorator(func):
        def _handle_timeout(signum, frame):
            error_message = os.strerror(errno.ETIME)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


@timeout(20)
def average_shortest_path_length_for_all(G):
    """
    Helper function, in the case of graph disconnectedness,
    that returns the average shortest path length, calculated
    iteratively for each distinct subgraph of the graph G.

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    average_shortest_path_length : float
        The length of the average shortest path for graph G.

    Notes
    -----
    A helper function for calculating average shortest path length
    in the case that a graph is disconnected. Calculation occurs
    across all subgraphs detected in G.
    """
    import math
    subgraphs = [sbg for sbg in nx.connected_component_subgraphs(G) if len(sbg) > 1]
    return math.fsum(nx.average_shortest_path_length(sg) for sg in subgraphs) / len(subgraphs)


def average_local_efficiency(G, weight=None):
    """
    Return the average local efficiency of all of the nodes in the graph G.

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    weight : float
        The edge attribute that holds the numerical value used as a weight.
        If None, then each edge has weight 1. Default is None.

    Returns
    -------
    average_local_efficiency : float
        The average of local efficiencies across all nodes of graph G.
    """
    eff = nx.algorithms.local_efficiency(G, weight)
    total = sum(eff.values())
    N = len(eff)
    return total/N


def create_communities(node_comm_aff_mat, node_num):
    """
    Create a 1D vector of community assignments from a community affiliation matrix.

    Parameters
    ----------
    node_comm_aff_mat : array
        Community affiliation matrix produced from modularity estimation (e.g. Louvain).

    node_num : int
        Number of total connected nodes in the graph used to estimate node_comm_aff_mat.

    Returns
    -------
    com_assign : array
        1D numpy vector of community assignments.
    """
    com_assign = np.zeros((node_num,1))
    for i in range(len(node_comm_aff_mat)):
        community = node_comm_aff_mat[i,:]
        for j in range(len(community)):
            if community[j] == 1:
                com_assign[j,0] = i
    return com_assign


@timeout(20)
def participation_coef(W, ci, degree='undirected'):
    '''
    ## ADAPTED FROM BCTPY ##

    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    degree : str
        Flag to describe nature of graph 'undirected': For undirected graphs
                                         'in': Uses the in-degree
                                         'out': Uses the out-degree
    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors
    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P


def participation_coef_sign(W, ci):
    '''
    ## ADAPTED FROM BCTPY ##

    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector
    Returns
    -------
    Ppos : Nx1 np.ndarray
        participation coefficient from positive weights
    Pneg : Nx1 np.ndarray
        participation coefficient from negative weights
    '''
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices

    def pcoef(W_):
        S = np.sum(W_, axis=1)  # strength
        # neighbor community affil.
        Gc = np.dot(np.logical_not(W_ == 0), np.diag(ci))
        Sc2 = np.zeros((n,))

        for i in range(1, int(np.max(ci) + 1)):
            Sc2 += np.square(np.sum(W_ * (Gc == i), axis=1))

        P = np.ones((n,)) - Sc2 / np.square(S)
        P[np.where(np.isnan(P))] = 0
        P[np.where(np.logical_not(P))] = 0  # p_ind=0 if no (out)neighbors
        return P

    #explicitly ignore compiler warning for division by zero
    with np.errstate(invalid='ignore'):
        Ppos = pcoef(W * (W > 0))
        Pneg = pcoef(-W * (W < 0))

    return Ppos, Pneg


def diversity_coef_sign(W, ci):
    '''
    ## ADAPTED FROM BCTPY ##

    The Shannon-entropy based diversity coefficient measures the diversity
    of intermodular connections of individual nodes and ranges from 0 to 1.
    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector
    Returns
    -------
    Hpos : Nx1 np.ndarray
        diversity coefficient based on positive connections
    Hneg : Nx1 np.ndarray
        diversity coefficient based on negative connections
    '''

    def entropy(w_):
        # Strength
        S = np.sum(w_, axis=1)
        # Node-to-module degree
        Snm = np.zeros((n, m))
        for i in range(m):
            Snm[:, i] = np.sum(w_[:, ci == i + 1], axis=1)
        pnm = Snm / (np.tile(S, (m, 1)).T)
        pnm[np.isnan(pnm)] = 0
        pnm[np.logical_not(pnm)] = 1
        return -np.sum(pnm * np.log(pnm), axis=1) / np.log(m)

    n = len(W)
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    # Number of modules
    m = np.max(ci)


    # Explicitly ignore compiler warning for division by zero
    with np.errstate(invalid='ignore'):
        Hpos = entropy(W * (W > 0))
        Hneg = entropy(-W * (W < 0))

    return Hpos, Hneg


def prune_disconnected(G):
    """
    Returns a copy of G with isolates pruned.

    Parameters
    ----------
    G : Obj
        NetworkX graph with isolated nodes present.

    Returns
    -------
    G : Obj
        NetworkX graph with isolated nodes pruned.
    pruned_nodes : list
        List of indices of nodes that were pruned from G.
    """
    print('Pruning fully disconnected...')

    # List because it returns a generator
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    components_isolated = list(components[0])

    # Remove disconnected nodes
    pruned_nodes = []
    s = 0
    for node in list(G.nodes()):
        if node not in components_isolated:
            G.remove_node(node)
            pruned_nodes.append(s)
        s = s + 1

    return G, pruned_nodes


def most_important(G):
     """
     Returns a copy of G with isolates and low-importance nodes pruned

     Parameters
     ----------
     G : Obj
         NetworkX graph.

     Returns
     -------
     G : Obj
         NetworkX graph with isolated and low-importance nodes pruned.
     pruned_nodes : list
        List of indices of nodes that were pruned from G.
     """
     print('Pruning fully disconnected and low importance nodes (3 SD < M)...')
     ranking = nx.betweenness_centrality(G).items()
     #print(ranking)
     r = [x[1] for x in ranking]
     m = sum(r)/len(r) - 3*np.std(r)
     Gt = G.copy()
     pruned_nodes = []
     i = 0
     # Remove near-zero isolates
     for k, v in ranking:
        if v < m:
            Gt.remove_node(k)
            pruned_nodes.append(i)
        i = i + 1

     # List because it returns a generator
     components = list(nx.connected_components(Gt))
     components.sort(key=len, reverse=True)
     components_isolated = list(components[0])

     # Remove disconnected nodes
     s = 0
     for node in list(Gt.nodes()):
         if node not in components_isolated:
             Gt.remove_node(node)
             pruned_nodes.append(s)
         s = s + 1

     return Gt, pruned_nodes


@timeout(60)
def raw_mets(G, i, custom_weight):
    """
    API that iterates across NetworkX algorithms for a graph G.

    Parameters
    ----------
    G : Obj
        NetworkX graph.
    i : str
        Name of the NetworkX algorithm.
    custom_weight : float
        The edge attribute that holds the numerical value used as a weight.
        If None, then each edge has weight 1. Default is None.

    Returns
    -------
    net_met_val : float
        Value of the graph metric i that was calculated from G.
    """
    if i is 'average_shortest_path_length':
        if nx.is_connected(G) is True:
            net_met_val = float(i(G))
        else:
            # Case where G is not fully connected
            print('WARNING: Calculating average shortest path length for a disconnected graph. '
                  'This might take awhile...')
            net_met_val = float(average_shortest_path_length_for_all(G))
    else:
        if custom_weight is not None and i is 'degree_assortativity_coefficient' or i is 'global_efficiency' or i is 'average_local_efficiency' or i is 'average_clustering':
            custom_weight_param = 'weight = ' + str(custom_weight)
            net_met_val = float(i(G, custom_weight_param))
        else:
            net_met_val = float(i(G))
    return net_met_val


# Extract network metrics interface
def extractnetstats(ID, network, thr, conn_model, est_path, roi, prune, node_size, norm, binary, custom_weight=None):
    """
    Function interface for performing fully-automated graph analysis.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    thr : float
        The value, between 0 and 1, used to threshold the graph using any variety of methods
        triggered through other options.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path : str
        File path to the thresholded graph, conn_matrix_thr, saved as a numpy array in .npy format.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    custom_weight : float
        The edge attribute that holds the numerical value used as a weight.
        If None, then each edge has weight 1. Default is None.

    Returns
    -------
    out_path : str
        Path to .csv file where graph analysis results are saved.
    """
    import pandas as pd
    import yaml
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pathlib import Path
    from pynets import thresholding, utils

    # Advanced options
    fmt = 'edgelist_ssv'
    est_path_fmt = "%s%s" % ('.', est_path.split('.')[-1])

    # Load and threshold matrix
    if est_path_fmt == '.txt':
        in_mat_raw = np.array(np.genfromtxt(est_path))
    else:
        in_mat_raw = np.array(np.load(est_path))

    # De-diagnal
    in_mat = np.array(np.array(thresholding.autofix(in_mat_raw)))

    # Normalize connectivity matrix
    # Force edges to values between 0-1
    if norm == 1:
        in_mat = thresholding.normalize(in_mat)
    # Apply log10
    elif norm == 2:
        in_mat = np.log10(in_mat)
    else:
        pass

    # Correct nan's and inf's
    in_mat[np.isnan(in_mat)] = 0
    in_mat[np.isinf(in_mat)] = 1

    # Get hyperbolic tangent (i.e. fischer r-to-z transform) of matrix if non-covariance
    if (conn_model == 'corr') or (conn_model == 'partcorr'):
        in_mat = np.arctanh(in_mat)

    # Binarize graph
    if binary is True:
        in_mat = thresholding.binarize(in_mat)

    # Get dir_path
    dir_path = os.path.dirname(os.path.realpath(est_path))

    # Load numpy matrix as networkx graph
    G_pre = nx.from_numpy_matrix(in_mat)

    # Prune irrelevant nodes (i.e. nodes who are fully disconnected from the graph and/or those whose betweenness
    # centrality are > 3 standard deviations below the mean)
    if prune == 1:
        [G, _] = prune_disconnected(G_pre)
    elif prune == 2:
        [G, _] = most_important(G_pre)
    else:
        G = G_pre

    # Get corresponding matrix
    in_mat = np.array(nx.to_numpy_matrix(G))

    # Saved pruned
    if prune > 0:
        final_mat_path = "%s%s%s" % (est_path.split(est_path_fmt)[0], '_pruned_mat', est_path_fmt)
        utils.save_mat(in_mat, final_mat_path, fmt)

    # Print graph summary
    print("%s%.2f%s" % ('\n\nThreshold: ', 100*float(thr), '%'))
    print("%s%s" % ('Source File: ', est_path))
    info_list = list(nx.info(G).split('\n'))[2:]
    for i in info_list:
        print(i)

    if nx.is_connected(G) is True:
        frag = False
        print('Graph is connected...')
    else:
        frag = True
        print('Warning: Graph is fragmented...\n')

    # Create Length matrix
    mat_len = thresholding.weight_conversion(in_mat, 'lengths')

    # Load numpy matrix as networkx graph
    G_len = nx.from_numpy_matrix(mat_len)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # Calculate global and local metrics from graph G # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    import community
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, betweenness_centrality, eigenvector_centrality, communicability_betweenness_centrality, clustering, degree_centrality, rich_club_coefficient, omega, global_efficiency, local_efficiency
    from pynets.stats.netstats import average_local_efficiency, participation_coef, participation_coef_sign, diversity_coef_sign
    # For non-nodal scalar metrics from custom functions, add the name of the function to metric_list and add the
    # function (with a G-only input) to the netstats module.
    metric_list_glob = [global_efficiency, average_local_efficiency, degree_assortativity_coefficient,
                        average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient,
                        graph_number_of_cliques, transitivity, omega]
    metric_list_comm = ['louvain_modularity']
    # with open("%s%s" % (str(Path(__file__).parent), '/global_graph_measures.yaml'), 'r') as stream:
    #     try:
    #         metric_dict_global = yaml.load(stream)
    #         metric_list_global = metric_dict_global['metric_list_global']
    #         print("%s%s%s" % ('\n\nCalculating global measures:\n', metric_list_global, '\n\n'))
    #     except FileNotFoundError:
    #         print('Failed to parse global_graph_measures.yaml')

    with open("%s%s" % (str(Path(__file__).parent), '/nodal_graph_measures.yaml'), 'r') as stream:
        try:
            metric_dict_nodal = yaml.load(stream)
            metric_list_nodal = metric_dict_nodal['metric_list_nodal']
            print("%s%s%s" % ('\n\nCalculating nodal measures:\n', metric_list_nodal, '\n\n'))
        except FileNotFoundError:
            print('Failed to parse nodal_graph_measures.yaml')

    # Note the use of bare excepts in preceding blocks. Typically, this is considered bad practice in python. Here,
    # we are exploiting it intentionally to facilitate uninterrupted, automated graph analysis even when algorithms are
    # undefined. In those instances, solutions are assigned NaN's.

    # Iteratively run functions from above metric list that generate single scalar output
    num_mets = len(metric_list_glob)
    net_met_arr = np.zeros([num_mets, 2], dtype='object')
    j = 0
    for i in metric_list_glob:
        met_name = str(i).split('<function ')[1].split(' at')[0]
        net_met = met_name
        try:
            try:
                net_met_val = raw_mets(G, i, custom_weight)
            except:
                print("%s%s%s" % ('WARNING: ', net_met, ' timeout for graph G. Most likely this is because the graph '
                                                        'is either disconnected or because it is fully saturated. See '
                                                        'thresholding and pruning options in pynets_run.py -h.'))
                net_met_val = np.nan
        except:
            print("%s%s%s" % ('WARNING: ', str(i), ' is undefined for graph G'))
            net_met_val = np.nan
        net_met_arr[j, 0] = net_met
        net_met_arr[j, 1] = net_met_val
        print(net_met)
        print(str(net_met_val))
        print('\n')
        j = j + 1
    net_met_val_list = list(net_met_arr[:, 1])

    # Create a list of metric names for scalar metrics
    metric_list_names = []
    net_met_val_list_final = net_met_val_list
    for i in net_met_arr[:, 0]:
        metric_list_names.append(i)

    # Run miscellaneous functions that generate multiple outputs
    # Calculate modularity using the Louvain algorithm
    if 'louvain_modularity' in metric_list_comm:
        try:
            ci = community.best_partition(G)
            modularity = community.community_louvain.modularity(ci, G)
            metric_list_names.append('modularity')
            net_met_val_list_final.append(modularity)
        except:
            print('Louvain modularity calculation is undefined for graph G')
            pass

    # Participation Coefficient by louvain community
    if 'participation_coefficient' in metric_list_nodal:
        try:
            if ci is None:
                raise KeyError('Participation coefficient cannot be calculated for graph G in the absence of a '
                               'community affiliation vector')
            if len(in_mat[in_mat < 0.0]) > 0:
                pc_vector = participation_coef_sign(in_mat, ci)
            else:
                pc_vector = participation_coef(in_mat, ci)
            print('\nExtracting Participation Coefficient vector for all network nodes...')
            pc_vals = list(pc_vector)
            pc_edges = list(range(len(pc_vector)))
            num_edges = len(pc_edges)
            pc_arr = np.zeros([num_edges + 1, 2], dtype='object')
            j = 0
            for i in range(num_edges):
                pc_arr[j, 0] = "%s%s" % (str(pc_edges[j]), '_partic_coef')
                try:
                    pc_arr[j, 1] = pc_vals[j]
                except:
                    print("%s%s%s" % ('Participation coefficient is undefined for node ', str(j), ' of graph G'))
                    pc_arr[j, 1] = np.nan
                j = j + 1
            # Add mean
            pc_arr[num_edges, 0] = 'average_participation_coefficient'
            nonzero_arr_partic_coef = np.delete(pc_arr[:, 1], [0])
            pc_arr[num_edges, 1] = np.mean(nonzero_arr_partic_coef)
            print("%s%s" % ('Mean Participation Coefficient across edges: ', str(pc_arr[num_edges, 1])))
            for i in pc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(pc_arr[:, 1])
        except:
            print('Participation coefficient cannot be calculated for graph G')
            pass

    # Diversity Coefficient by louvain community
    if 'diversity_coefficient' in metric_list_nodal:
        try:
            if ci is None:
                raise KeyError('Diversity coefficient cannot be calculated for graph G in the absence of a community '
                               'affiliation vector')
            [dc_vector, _] = diversity_coef_sign(in_mat, ci)
            print('\nExtracting Diversity Coefficient vector for all network nodes...')
            dc_vals = list(dc_vector)
            dc_edges = list(range(len(dc_vector)))
            num_edges = len(dc_edges)
            dc_arr = np.zeros([num_edges + 1, 2], dtype='object')
            j = 0
            for i in range(num_edges):
                dc_arr[j, 0] = "%s%s" % (str(dc_edges[j]), '_diversity_coef')
                try:
                    dc_arr[j, 1] = dc_vals[j]
                except:
                    print("%s%s%s" % ('Diversity coefficient is undefined for node ', str(j), ' of graph G'))
                    dc_arr[j, 1] = np.nan
                j = j + 1
            # Add mean
            dc_arr[num_edges, 0] = 'average_diversity_coefficient'
            nonzero_arr_diversity_coef = np.delete(dc_arr[:, 1], [0])
            dc_arr[num_edges, 1] = np.mean(nonzero_arr_diversity_coef)
            print("%s%s" % ('Mean Diversity Coefficient across edges: ', str(dc_arr[num_edges, 1])))
            for i in dc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(dc_arr[:, 1])
        except:
            print('Diversity coefficient cannot be calculated for graph G')
            pass

    # Local Efficiency
    if 'local_efficiency' in metric_list_nodal:
        try:
            le_vector = local_efficiency(G)
            print('\nExtracting Local Efficiency vector for all network nodes...')
            le_vals = list(le_vector.values())
            le_nodes = list(le_vector.keys())
            num_nodes = len(le_nodes)
            le_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                le_arr[j, 0] = "%s%s" % (str(le_nodes[j]), '_local_efficiency')
                try:
                    le_arr[j, 1] = le_vals[j]
                except:
                    print("%s%s%s" % ('Local efficiency is undefined for node ', str(j), ' of graph G'))
                    le_arr[j, 1] = np.nan
                j = j + 1
            le_arr[num_nodes, 0] = 'average_local_efficiency_nodewise'
            nonzero_arr_le = np.delete(le_arr[:, 1], [0])
            le_arr[num_nodes, 1] = np.mean(nonzero_arr_le)
            print("%s%s" % ('Mean Local Efficiency across nodes: ', str(le_arr[num_nodes, 1])))
            for i in le_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(le_arr[:, 1])
        except:
            print('Local efficiency cannot be calculated for graph G')
            pass

    # Local Clustering
    if 'local_clustering' in metric_list_nodal:
        try:
            cl_vector = clustering(G)
            print('\nExtracting Local Clustering vector for all network nodes...')
            cl_vals = list(cl_vector.values())
            cl_nodes = list(cl_vector.keys())
            num_nodes = len(cl_nodes)
            cl_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                cl_arr[j, 0] = "%s%s" % (str(cl_nodes[j]), '_local_clustering')
                try:
                    cl_arr[j, 1] = cl_vals[j]
                except:
                    print("%s%s%s" % ('Local clustering is undefined for node ', str(j), ' of graph G'))
                    cl_arr[j, 1] = np.nan
                j = j + 1
            cl_arr[num_nodes, 0] = 'average_local_efficiency_nodewise'
            nonzero_arr_cl = np.delete(cl_arr[:, 1], [0])
            cl_arr[num_nodes, 1] = np.mean(nonzero_arr_cl)
            print("%s%s" % ('Mean Local Clustering across nodes: ', str(cl_arr[num_nodes, 1])))
            for i in cl_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(cl_arr[:, 1])
        except:
            print('Local clustering cannot be calculated for graph G')
            pass

    # Degree centrality
    if 'degree_centrality' in metric_list_nodal:
        try:
            dc_vector = degree_centrality(G)
            print('\nExtracting Degree Centrality vector for all network nodes...')
            dc_vals = list(dc_vector.values())
            dc_nodes = list(dc_vector.keys())
            num_nodes = len(dc_nodes)
            dc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                dc_arr[j, 0] = "%s%s" % (str(dc_nodes[j]), '_degree_centrality')
                try:
                    dc_arr[j, 1] = dc_vals[j]
                except:
                    print("%s%s%s" % ('Degree centrality is undefined for node ', str(j), ' of graph G'))
                    dc_arr[j, 1] = np.nan
                j = j + 1
            dc_arr[num_nodes, 0] = 'average_degree_cent'
            nonzero_arr_dc = np.delete(dc_arr[:, 1], [0])
            dc_arr[num_nodes, 1] = np.mean(nonzero_arr_dc)
            print("%s%s" % ('Mean Degree Centrality across nodes: ', str(dc_arr[num_nodes, 1])))
            for i in dc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(dc_arr[:, 1])
        except:
            print('Degree centrality cannot be calculated for graph G')
            pass

    # Betweenness Centrality
    if 'betweenness_centrality' in metric_list_nodal:
        try:
            bc_vector = betweenness_centrality(G_len, normalized=True)
            print('\nExtracting Betweeness Centrality vector for all network nodes...')
            bc_vals = list(bc_vector.values())
            bc_nodes = list(bc_vector.keys())
            num_nodes = len(bc_nodes)
            bc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                bc_arr[j, 0] = "%s%s" % (str(bc_nodes[j]), '_betweenness_centrality')
                try:
                    bc_arr[j, 1] = bc_vals[j]
                except:
                    print("%s%s%s" % ('Betweeness centrality is undefined for node ', str(j), ' of graph G'))
                    bc_arr[j, 1] = np.nan
                j = j + 1
            bc_arr[num_nodes, 0] = 'average_betweenness_centrality'
            nonzero_arr_betw_cent = np.delete(bc_arr[:, 1], [0])
            bc_arr[num_nodes, 1] = np.mean(nonzero_arr_betw_cent)
            print("%s%s" % ('Mean Betweenness Centrality across nodes: ', str(bc_arr[num_nodes, 1])))
            for i in bc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(bc_arr[:, 1])
        except:
            print('Betweenness centrality cannot be calculated for graph G')
            pass

    # Eigenvector Centrality
    if 'eigenvector_centrality' in metric_list_nodal:
        try:
            ec_vector = eigenvector_centrality(G, max_iter=1000)
            print('\nExtracting Eigenvector Centrality vector for all network nodes...')
            ec_vals = list(ec_vector.values())
            ec_nodes = list(ec_vector.keys())
            num_nodes = len(ec_nodes)
            ec_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                ec_arr[j, 0] = "%s%s" % (str(ec_nodes[j]), '_eigenvector_centrality')
                try:
                    ec_arr[j, 1] = ec_vals[j]
                except:
                    print("%s%s%s" % ('Eigenvector centrality is undefined for node ', str(j), ' of graph G'))
                    ec_arr[j, 1] = np.nan
                j = j + 1
            ec_arr[num_nodes, 0] = 'average_eigenvector_centrality'
            nonzero_arr_eig_cent = np.delete(ec_arr[:, 1], [0])
            ec_arr[num_nodes, 1] = np.mean(nonzero_arr_eig_cent)
            print("%s%s" % ('Mean Eigenvector Centrality across nodes: ', str(ec_arr[num_nodes, 1])))
            for i in ec_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(ec_arr[:, 1])
        except:
            print('Eigenvector centrality cannot be calculated for graph G')
            pass

    # Communicability Centrality
    if 'communicability_centrality' in metric_list_nodal:
        try:
            cc_vector = communicability_betweenness_centrality(G, normalized=True)
            print('\nExtracting Communicability Centrality vector for all network nodes...')
            cc_vals = list(cc_vector.values())
            cc_nodes = list(cc_vector.keys())
            num_nodes = len(cc_nodes)
            cc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                cc_arr[j, 0] = "%s%s" % (str(cc_nodes[j]), '_communicability_centrality')
                try:
                    cc_arr[j, 1] = cc_vals[j]
                except:
                    print("%s%s%s" % ('Communicability centrality is undefined for node ', str(j), ' of graph G'))
                    cc_arr[j, 1] = np.nan
                j = j + 1
            cc_arr[num_nodes, 0] = 'average_communicability_centrality'
            nonzero_arr_comm_cent = np.delete(cc_arr[:, 1], [0])
            cc_arr[num_nodes, 1] = np.mean(nonzero_arr_comm_cent)
            print("%s%s" % ('Mean Communicability Centrality across nodes: ', str(cc_arr[num_nodes, 1])))
            for i in cc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(cc_arr[:, 1])
        except:
            print('Communicability centrality cannot be calculated for graph G')
            pass

    # Rich club coefficient
    if 'rich_club_coefficient' in metric_list_nodal:
        try:
            rc_vector = rich_club_coefficient(G, normalized=True)
            print('\nExtracting Rich Club Coefficient vector for all network nodes...')
            rc_vals = list(rc_vector.values())
            rc_edges = list(rc_vector.keys())
            num_edges = len(rc_edges)
            rc_arr = np.zeros([num_edges + 1, 2], dtype='object')
            j = 0
            for i in range(num_edges):
                rc_arr[j, 0] = "%s%s" % (str(rc_edges[j]), '_rich_club')
                try:
                    rc_arr[j, 1] = rc_vals[j]
                except:
                    print("%s%s%s" % ('Rich club coefficient is undefined for node ', str(j), ' of graph G'))
                    rc_arr[j, 1] = np.nan
                j = j + 1
            # Add mean
            rc_arr[num_edges, 0] = 'average_rich_club_coefficient'
            nonzero_arr_rich_club = np.delete(rc_arr[:, 1], [0])
            rc_arr[num_edges, 1] = np.mean(nonzero_arr_rich_club)
            print("%s%s" % ('Mean Rich Club Coefficient across edges: ', str(rc_arr[num_edges, 1])))
            for i in rc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(rc_arr[:, 1])
        except:
            print('Rich club coefficient cannot be calculated for graph G')
            pass

    if roi:
        met_list_picke_path = "%s%s%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list', "%s" %
                                            ("%s%s%s" % ('_', network, '_') if network else "_"),
                                            os.path.basename(roi).split('.')[0])
    else:
        if network:
            met_list_picke_path = "%s%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list_', network)
        else:
            met_list_picke_path = "%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list')
    pickle.dump(metric_list_names, open(met_list_picke_path, 'wb'), protocol=2)

    # And save results to csv
    out_path = utils.create_csv_path(ID, network, conn_model, thr, roi, dir_path, node_size)
    np.savetxt(out_path, net_met_val_list_final, delimiter='\t')

    if frag is True:
        out_path_neat = "%s%s" % (out_path.split('.csv')[0], '_frag_neat.csv')
    else:
        out_path_neat = "%s%s" % (out_path.split('.csv')[0], '_neat.csv')
    df = pd.DataFrame.from_dict(dict(zip(metric_list_names, net_met_val_list_final)), orient='index').transpose()
    df.to_csv(out_path_neat, index=False)

    return out_path
