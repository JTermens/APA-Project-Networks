#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------------------
# APA PROJECT MAIN
#
# This code is degined to generate and filter graphs, search communities on it with the Louvain or
# bining algorithm, characterize the communities with 10 features and represent them in a class.
# It will build k-d trees and find k-nearest neighbour communities closer than a distance
# d from a given pivot.
#
#
# Alda Sabalić, Beatriz Urda and Joan Térmens                  Last revision: 24th of November 2019
#--------------------------------------------------------------------------------------------------

from networktools import choose_detection_method, compute_features, make_kd_tree

           
if __name__=='__main__':

    # Reading the data, building a network, choosing community detection algorithm and obtaining communities
    dict_nodes, df_filtered = choose_detection_method(filename = "4_coexpr_geo_v2.txt", k = 8)

    # Computing the features of each community and building a list of Community instances
    community_list = []
    for comm_num in dict_nodes.keys():
        set_nodes = dict_nodes[comm_num]
        community_list.append(compute_features(set_nodes,df_filtered,'gene1','gene2','edge'))
    print('Feature extraction completed...')
    
    # Generating a k-d tree, in this case, using the maximum number of features to achieve optimal performance
    print("Generating a k-dim tree...")
    kd_tree = make_kd_tree(community_list)

    # Selecting a pivot
    comm_pivot = community_list[1]
    dim = len(kd_tree.axis_key)
    
    # Initializing arguments for the group function
    num_neighbours = 12
    dist_to_neigh = 1.6

    # Finding the k nearest neighbours using the previously considered features, they are contained in the 'final' list
    final = kd_tree.group(comm_pivot, dist_to_neigh, num_neighbours, 'dens', 'size', 'rel_dens', 'max_btw', 'avg_btw', 'max_centr', 'avg_centr', 'max_load')
    print ('The found neigbours are: ' )
    for f in range(0,len(final)):
        print("Community: dens = {}, size = {}, max btw = {}, ...)".format(final[f].dens,final[f].size,final[f].max_btw))
    	

