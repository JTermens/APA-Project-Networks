#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import cProfile
from networktools import choose_detection_method, compute_features, make_kd_tree

def compute_features_loop():
    community_list = []
    for comm_num in dict_nodes.keys():
        set_nodes = dict_nodes[comm_num]
        community_list.append(compute_features(set_nodes,df_filtered,'gene1','gene2','edge'))
    return community_list

def distance_loop():
    for comm in community_list:
        community_list[0].distance(comm)

def group_bench():
    kd_tree.group(comm_pivot, dist_to_neigh, num_neighbours)

print('main program profiling \n')
print('The tested functions will be:')
print('\t*choose_detection_method with community_binning\n\t*choose_detection_method with louvain\n\t*compute_features\n\t*distance\n\t*\n\t*make_kd_tree\n\t*group')
# Reading the data, building a network, choosing community detection algorithm and obtaining communities

print('\n choose_detection_method with community_binning (k = 10)')
cProfile.runctx('choose_detection_method(filename,k)', {'filename':"4_coexpr_geo_v2.txt", 'k':10, 'choose_detection_method':choose_detection_method}, {})
dict_nodes, df_filtered = choose_detection_method(filename = "4_coexpr_geo_v2.txt", k = 10)

print('\n choose_detection_method with louvain (k = 6)')
cProfile.runctx('choose_detection_method(filename,k)', {'filename':"4_coexpr_geo_v2.txt", 'k':6, 'choose_detection_method':choose_detection_method}, {})

print('\n Using the results from the community binning algorithm (2000 communities):')

print('\n compute_features for each community')
cProfile.run('compute_features_loop()')
community_list = compute_features_loop()

print('\n distance for 2000 pairs of communities')
cProfile.run('distance_loop()')

print('\n make_kd_tree with 10 arguments')
cProfile.runctx('make_kd_tree(community_list)', {'community_list':community_list, 'make_kd_tree':make_kd_tree}, {})
kd_tree = make_kd_tree(community_list)

comm_pivot = community_list[1]
num_neighbours = 100
dist_to_neigh = 10

print('\n group for k = 100 and d = 10')
cProfile.run('group_bench()')
