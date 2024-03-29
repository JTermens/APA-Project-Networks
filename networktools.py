#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------------------
# APA PROJECT LIBRARY NETWORKTOOLS
#
# This code is degined to generate and filter graphs, search communities on it with the Louvain or
# bining algorithm, characterize the communities with 10 features and represent them in a class.
# It will build k-d trees and find k-nearest neighbour communities closer than a distance
# d from a given pivot.
#
#
# Alda Sabalić, Beatriz Urda and Joan Térmens                  Last revision: 24th of November 2019
#--------------------------------------------------------------------------------------------------

# Importing functions
import pandas as pd
import numpy as np
from pandas import DataFrame
import networkx as nx
from networkx.algorithms import community
import community
from math import sqrt
from functools import reduce

def filtered(filename,threshold = 0.38):
    """Generates a data frame from a given file (filename) and filters it removing the
    edges with a weight lower than a given threshold.

    Input:  * filename: a tab separated file containing 3 columns (source node, target node and
            weight) and a row for each edge
            * threshold: minimum weight that an edge has to have ho not being filtered.
    Output: * df_filtered: filtered dataframe, has 3 columns, keyed 'gene1', 'gene2' and 'edge' and
    a row for each edge.
    """
    df = pd.read_csv(filename, sep="\t")
    df_filtered=df.loc[ (df['edge'] > threshold) & (df['gene1'] != df['gene2']) ]
    return df_filtered

def network(df):
    """Creates and returns a grap (G) from a dataframe (df) using NetworkX library.

    Input:  * df: dataframe with 3 columns, keyed 'gene1', 'gene2' and 'edge' and a row for each
            edge.
    Output: * G: Network graph
    """
    G=nx.from_pandas_edgelist(df, 'gene1', 'gene2')
    return G

def louvain(G):
    """Implements the Louvain community search algorithm in a graph (G) and returns a
    dictionary of the partitions.
    Input:  * G: a graph generated with NetworkX
    Output: * dic_nodes: a dictionary like {community_number:{'node_1 , ... , node_N'}}
    """
    # Starting with an initial partition of the graph and running the Louvain algorithm for Community Detection
    partition=community.best_partition(G, weight='MsgCount')

    values=[partition.get(node) for node in G.nodes()]
    list_com=partition.values()

    # Creating a dictionary like {community_number:list_of_participants}
    dict_nodes={}

    # Populating the dictionary with items
    for each_item in partition.items():
        v= set()
        community_num=each_item[1]
        community_node=each_item[0]
        if community_num in dict_nodes:

            dict_nodes.get(community_num).add(community_node)

        else:
            #print('entered else')
            v.add(community_node)
            dict_nodes.update({community_num:v})

    # Creating a new graph to represent the communities created by the Louvain algorithm
    G_comm=nx.Graph()

    # Populating the data from the node dictionary created earlier
    G_comm.add_nodes_from(dict_nodes)

    # Calculating modularity and the total number of communities
    mod=community.modularity(partition,G)
    print("Total number of Communities=", len(G_comm.nodes()))
    print("Modularity: ", mod)

    return dict_nodes


def community_binning(filename):
    """In the case the spacified number of features k is >= 8, this function is called for generating communities
    and returns a dictionary of the partitions.
    Input:  * filename: the name of the file with the adjacency matrix
    Output: * dic_nodes: a dictionary like {community_number:{'node_1 , ... , node_N'}}
    		* dfr: dataframe containing the information about the extracted rows from which the communities are computed
    """
    from random import sample
    dict_nodes = {}
    df = pd.read_csv(filename, sep="\t")
    
    #create random index
    rindex =  np.array(sample(range(len(df)), 4000))

    # get 4000 random rows from df
    dfr = df.loc[rindex]
    dataframes = np.array_split(dfr, 2000) ###array of 2000 dataframes each containing 2 rows
    i = 0
    for dataframe in dataframes:
        #print(len(list(dataframe.gene1)))
        l1 = set(dataframe.gene1)
        l2 = set(dataframe.gene2)
        l_of_features = l1.union(l2)
        #print(l_of_features)
        value = set(l_of_features)
        #print(value)
        dict_nodes.update({i:value})
        i += 1
    #print(len(dict_nodes))
    return dict_nodes, dfr


def normalize_list(list_normal):
    max_value = max(list_normal)
    min_value = min(list_normal)
    for i in range(len(list_normal)):
        if (max_value == min_value):
           return list_normal
        else:
            list_normal[i] = (list_normal[i] - min_value) / (max_value - min_value)
    return list_normal

def euclidean_distance(a,b):
    """Returns euclidean distance for vector of dimension n>=2"""
    if(len(a) != len(b)):
        raise IndexError("The vectors must have the same number of elements")
    for k in a:
        if type(k) not in [int,float]:
            raise TypeError("The vectors must contain numbers")
    for k in b:
        if type(k) not in [int,float]:
            raise TypeError("The vectors must contain numbers")
    else:
        dim = len(a)
        return sqrt(reduce(lambda i,j: i + ((a[j] - b[j]) ** 2), range(dim), 0))

def choose_detection_method(filename,k):
    """Function that given an input parameter k chooses the optimal community detectiom method.
    If k <= 8, community detection will be performed by using Louvain's algorithm with function.
    If k > 8, community detecion will be done by using a randomization binning implemented in function community_binning.  

    Input: * filename: a tab separated file containing 3 columns (source node, target node and
            weight) and a row for each edge representing the adjacency matrix of the network
           * k: the number of features of each community that the user wants to use in the furhter analysis.

    Output: * dic_nodes: a dictionary like {community_number:{'node_1 , ... , node_N'}}
    		* df_fil: a dataframe of considered.  it has 3 columns, keyed 'gene1', 'gene2' and 'edge' and a row for each edge. 
    """
    if k > 8:
        print("'Binning community detection in course...'")
        dict_nodes, df_fil = community_binning(filename)
    elif k <= 8:
        df_fil = filtered(filename)
        net = network(df_fil)
        print("'Louvain's community detection in course...'")
        dict_nodes = louvain(net)
    
    return dict_nodes, df_fil

# Community class, each comunity (parition) found with Louvain algorithm is characterized by 10
# features and stored as an instance of this class.
class Community(object):
    """The community class represents a given network community by 10 of its features.
    Each feature is an attribute of the class. It also contains a function (distance) 
    that computes the distance from self to another community instance based
    on a given list of features.
    Attributes:
        density (dens)
        size (size)
        relative centrality (rel_dens)
        maximum betweenness centrality (max_btw)
        average betweenness centrality (avg_btw)
        maximum degree centrality (max_centr)
        average degree centrality (avg_centr)
        maximum load centrality (max_load)
        average load centrality (avg_load)
        community modularity (mod)
    Funtions:
        get_features()
        distance(other,*argv) 
    """
    def __init__(self,dens,size,rel_dens,max_btw,avg_btw,max_centr,avg_centr,max_load,avg_load,mod):
        self.dens = dens
        self.size = size
        self.rel_dens = rel_dens 
        self.max_btw = max_btw
        self.avg_btw = avg_btw 
        self.max_centr = max_centr 
        self.avg_centr = avg_centr
        self.max_load = max_load 
        self.avg_load = avg_load
        self.mod = mod

    def get_features(self):
        """Returns a list with the features of the community"""
        feat_list = list()
        for feat in list(self.__dict__.keys()):
            feat_list.append(self.__getattribute__(feat))
            
        return (feat_list)


    def distance(self,other,*argv):
        """Returns the distance between the current community and other instance of
        the class using the attributes specified in argv. If no arg is provided,
        it uses all the attributes"""
        if(other.__class__.__name__ != 'Community'):
            raise TypeError("The first argument must be an instance of the Community class")
        if (len(argv) == 0):
            features1 = self.get_features()
            features2 = other.get_features()
            features1 = tuple(normalize_list(features1))
            features2 = tuple(normalize_list(features2))
            return (euclidean_distance(features1,features2))
        
        elif (len(argv) == 1):
            features1 = list()
            features2 = list()
            if(argv[0] in dir(self)):
                features1.append(self.__getattribute__(argv[0]))
                features2.append(other.__getattribute__(argv[0]))
                return (euclidean_distance(features1,features2))
            else:
                raise AttributeError("%s is not an attribute of the community class" %(argv[0]))
        else:
            features1 = list()
            features2 = list()
            for arg in argv:
                if(arg in dir(self)):
                    features1.append(self.__getattribute__(arg))
                    features2.append(other.__getattribute__(arg))
                else:
                    raise AttributeError("%s is not an attribute of the community class" %(arg))
            features1 = tuple(normalize_list(features1))
            features2 = tuple(normalize_list(features2))
            return (euclidean_distance(features1,features2))

def compute_features(set_nodes,df_network,key1='node1',key2='node2',weight = None):
    """Computes the following features of a given community in a graph: (1) density,
    (2) size, (3) relative density,  (4) maximum betweenness centrality,  (5) average betweenness
    centrality, (6) maximum degree centrality, (7) average degree centrality, (8) maximum load
    centrality, (9) average load centrality and (10) modularity.
    Input:  * set_nodes: iterable containing the community nodes
        	* df_network: network dataframe with a column for the source node (key1), a column for
        	the target node (key2) and an optional column for the edge weight. Has a row for each
        	edge.
        	* key1 and key2: dataframe source and targe node column keys, by default; 'node1' and
        	'node2'.
        	* weight: key of the node weight in the dataframe. By default, None (unweighted graph)
    Output: * comm: instance of the Community class containing the 10 specified features.
    """

    # Argument parsing for errors
    if (len(set_nodes) == 0):
        raise IndexError('set_nodes should not be empty')

    if df_network.empty:
        raise TypeError('df_network should not be empty')
    if key1 not in df_network.keys():
        raise KeyError('{} is not a key of df_network'.format(key1))
    if key2 not in df_network.keys():
        raise KeyError('{} is not a key of df_network'.format(key2))
    if (weight is not None) and (weight not in df_network.keys()):
        raise KeyError('{} is not a key of df_network'.format(key2))

    df_comm = df_network[(df_network[key1].isin(set_nodes)) & (df_network[key2].isin(set_nodes))] # community data frame

    if df_comm.empty:
        raise TypeError('df_comm should not be empty, look at the elements of set_nodes')

    g_comm = nx.from_pandas_edgelist(df_comm,key1,key2,edge_attr = weight) # community graph
    g_network = nx.from_pandas_edgelist(df_network,key1,key2,edge_attr = weight) # network graph

    dict_comm = {node:0 if node in set_nodes else 1 for node in g_network.nodes()} # necessary to calculate the modularity

    # Computing features:

    dens = nx.density(g_comm) # 1. density
    size = len(set_nodes) # 2. size
    rel_dens = dens/size # 3. relative density

    btw_centr = nx.betweenness_centrality(g_comm, normalized=True, weight=weight) # dict with nodes' betweenness centrality
    max_btw = max(btw_centr.values()) # 4. max betweenness centrality
    avg_btw = sum(btw_centr.values())/size # 5. average betweenness centrality

    degree_centr = nx.degree_centrality(g_comm) # dict with nodes' degree centrality
    max_centr = max(degree_centr.values()) # 6. max degree centrality
    avg_centr = sum(degree_centr.values())/size # 7. average degree centrality

    load_centr = nx.load_centrality(g_comm, normalized=True, weight=weight) # dict with nodes' load centrality
    max_load = max(load_centr.values()) # 8. max load centrality
    avg_load = sum(load_centr.values())/size # 9. average load centrality

    mod = community.modularity(dict_comm,g_network) # 10. community modularity within the network

    comm = Community(dens,size,rel_dens,max_btw,avg_btw,max_centr,avg_centr,max_load,avg_load,mod)

    return comm

# Tree class, implements an object with 4 atributes: node is the value of the tree node, left is the left branch, right,
# the right one and axis_key is a tuple containing the keys of the attributes that are used as k-d tree's axis. It is set as
# None in all nodes except for the root
class Tree(object):
    """Binary Tree structured defined by a root node (node) and its left and
    right children (left, rigth, respectively). It also contains the attribute
    axis, that is the list of features of the nodes used to build the tree
    Attributes:
        node,left,right,axis
    Functions:
        group(x, d, k, *argv)
    """
    def __init__(self,node,left=None,right=None, axis_key= None):
        self.node = node
        self.left = left
        self.right = right
        self.axis = axis_key

    def group(self, x, d, k, *argv):
        """Returns a list with the k nearest neighbours of a given pivot
        community (x) that are closer than a distance d using the features
        specified in *argv"""
        
        # Checking if the arguments are of the correct type
        if(x.__class__.__name__ != 'Community'):
            raise TypeError("The first argument must be an instance of the Community class")
        
        if type(d) not in [int,float]:
            raise TypeError("The distance must be a non-negative real number")
        
        if d < 0:
            raise ValueError("The distance must be a positive number")
        
        if type(k) is not int: 
            raise TypeError("The number of nearest neighbours must be a non-negative integer")
        
        if k < 1:
            raise ValueError("The number of nearest neighbours must be >= 1")
        
        for arg in argv:
            if arg not in x.__dict__.keys():
                raise AttributeError('{} is not an attribute of the Tree class'.format(arg))
                
        axis_key = self.axis_key
        dim = len(self.axis_key)

        comm_pivot = x
        num_neighbours = k
        distance = d 

        if len(axis_key) != len(argv):
            raise IndexError('Number of dimensions to build the tree and to perform neighbour-search must match.')
        
        else:
            print('Finding the %s nearest neighbours...' %(num_neighbours))
        	# creates a list of tuples nn that has the form: [(dist_to_neighbour1, neighbour1),..,(dist_to_neighbourN, neighbourN)]
            nn = get_k_neighbours(comm_pivot,self,num_neighbours,dim,axis_key)
            #print(nn)
            counter = 0
            l = []
            for n in range (0,len(nn)):
                #print(nn[n][0])
                #print(distance)
                if nn[n][0] <= distance:
                    #print('this is counter',counter)
                    counter += 1
                    l.append(nn[n][1])
                    #return ("dist = {}; Community: dens = {}, size = {}, max btw = {}, ...)".format(nn[n][0],nn[n][1].dens,nn[n][1].size,nn[n][1].max_btw))
            #print('this is counter')
            #print(counter)
            if counter < num_neighbours:
               raise ValueError('There are no %s nearest neighbours. The maximum number of neighbours found at distance %s is %s. Redefine your search by selecting num_neighbours = %s or less. ' \
               %(num_neighbours, distance, counter, counter))
            return l 


def recursive_kd_tree(list_inst,axis_key,i=0):
    """Creates a K-d tree recursively from a list of instances (list_inst) and axis_key,
    a tuple containing the keys (as strings) of the class attributes that will be used as axis.
    Input:  *list_inst: a list of instances of a class
            *axis_key: tuple containing the keys (as strings) of the class attributes that will be
            used as axis.
            *i: axis number in which the list_inst is sorted and paritioned. By default, 0.
    Output: *instance of Tree class where the node is the median point, left calls to create a left
            subtree and right, a right one.
    """

    # Argument parsing for errors
    if (i < 0) or (i > (len(axis_key)-1)):
        raise IndexError('i should be between 0 and {}'.format(len(axis_key)-1))
    if not isinstance(i, int):
        raise TypeError('i should be an integer')

    if len(list_inst) > 1:
        dim = len(axis_key)
        list_inst.sort(key=lambda x: x.__getattribute__(axis_key[i])) # pyhton list.sort has complexity O(nlog(n))
        i = (i + 1) % dim
        half = len(list_inst) >>1 # just division by 2 moving the bits

        return Tree(list_inst[half], recursive_kd_tree(list_inst[:half],axis_key,i),recursive_kd_tree(list_inst[half+1:],axis_key,i))

    elif len(list_inst) == 1:
        return Tree(list_inst[0])

def get_axis_key(list_inst,list_attr = None):
    """Checks if the given instances are from the same class and creates axis_key, a
    tuple containing the keys (as strings) of the class attributes that will be used as axis.
    This could be done by two ways:
        *If the user gives a list of attributes as list_attr, it will check if they are attributes
        of the given instances and create axis_key from it.
        *By default: list_attr = None; the function computes the optimal number of attributes to
        ensure efficiency (N > 2^k) and generates axis_key with the firsts ones (ordered
        alphabetically by the attribute key)
    Input:  *list_inst: list of instances of a certain class
            *list_attr: list of attributes of the given instances, by default, None
    Output: *axis_key: tuple containing the keys (as strings) of the class attributes that will be
            used as axis to make a k-d tree.
    """
    if(len(list_inst) == 0):
        raise IndexError('list_inst should not be empty')

    # chek if all instances belong to the same class
    if(len(list_inst) > 0):
        for inst in list_inst:
            # if an instance isn't from the same class as the 1st one, raise an error
            if(inst.__class__.__name__ != list_inst[0].__class__.__name__):
                raise TypeError('instance {} do not belong to the same class as the first one'.format(inst))

    if (list_attr == None): # create axis_key from scrach
        axis_key = tuple(list_inst[0].__dict__.keys()) # get the keys of all possible attributes
        if (len(list_inst) < 2**(len(axis_key))): # ensures N > 2^k, so the algorithm remains efficient
            optimal_num_attr = len(list_inst).bit_length()-1 # binary form of log2(len(list_inst))
            axis_key = axis_key[:optimal_num_attr]
    else: # creates axis_key from a non-empty list_attr and checks if the attributes are corrct
        axis_key = list()
        for attr in list_attr:
            if (attr in tuple(list_inst[0].__dict__.keys())):
                axis_key.append(attr)
            else:
                raise AttributeError('{} is not an attribute of the given instances'.format(attr))
        axis_key = tuple(axis_key)

    return axis_key

def make_kd_tree(list_inst,list_attr = None):
    """This function recursively takes a list of instances and a list of attributes and generates a
    k-dim tree using the specified features as the different axis. Firstly calls to GetKeyAxis to
    generate the axis_key that is a tuple containing the keys (as strings) of the class attributes
    that will be used as axis. Then, calls to RecursiveKdTree which recursively generates the tree
    Input:  *list_inst: a list of instances of a class
            *list_attr: list of attributes of the given instances, by default, None
    Output: *kd_tree: k-dim binary tree, instance of class Tree
    """

    axis_key = get_axis_key(list_inst,list_attr)
    kd_tree = recursive_kd_tree(list_inst,axis_key)

    kd_tree.axis_key = axis_key

    return kd_tree

def get_nearest_neighbour(pivot,kd_tree,dim,dist_func,axis_key,i=0,best=None):
    """Traverses a given kd-tree to find the closest instance to the pivot. First, it
    checks that the kd-tree's root node and the pivot are instances of the same class and then
    recursively tranveses the kd-tree looking for the closest instance and saving it as best.
    Input:  *pivot: instance for wich the function searches for the nearest neighbour.
            *kd_tree: a kd-tree of instances of the same class than pivot.
            *dim: dimension of axis_key, number of attributes considered to generate the kd-tree
            *dist_func: distance function.
            *axis_key: tuple containing the keys (as strings) of the class attributes that will be
            used as axis to make a k-d tree.
            *axis number in which the compairsons are done. By default, 0.
            *best: [minimum distance, closer instance].
    Output: *best: tuple(minimum distance, closer instance).
    """

    # Argument parsing for errors

    if not isinstance(dim, int):
        raise TypeError('dim should be an integer')
    if (len(axis_key) != dim):
        raise IndexError('axis_key lenght and dim sould be equal')

    if (i < 0) or (i > (len(axis_key)-1)):
        raise IndexError('i should be between 0 and {}'.format(len(axis_key)-1))
    if not isinstance(i, int):
        raise TypeError('i should be an integer')

    if kd_tree is not None:

        # Argument parsing for errors
        if (pivot.__class__.__name__ != kd_tree.node.__class__.__name__):
            raise TypeError('pivot and node tree {} should belong to the same class'.format(kd_tree.node))
        if (axis_key[i] not in dir(pivot)):
            raise AttributeError('{} is not an attribute of the class of the pivot'.format(axis_key[i]))

        # dist is the distance from the pivot to the root node of the actual kd-tree
        # dist_x is the distance from the kd-tree partition line and the pivot
        dist = dist_func(kd_tree.node,*axis_key)
        dist_x = kd_tree.node.__getattribute__(axis_key[i]) - pivot.__getattribute__(axis_key[i])

        if not best: # is there is no best result, save the actual one
            best = [dist,kd_tree.node]
        elif dist < best[0]: # if there is a best result, uptade if the actual is better
            best[0], best[1] = dist, kd_tree.node

        if (dist_x < 0): # if dist_x < 0 <-> pivot is at the right side
            next_branch = kd_tree.right
            opp_branch = kd_tree.left
        else: # if dist_x > 0 <-> pivot is at the left side
            next_branch = kd_tree.left
            opp_branch = kd_tree.right

        i = (i+1) % dim # next axis number

        # follow searching at the actual side of the partition (branch)
        get_nearest_neighbour(pivot,next_branch,dim,dist_func,axis_key,i,best)

        # if pivot is closer to the partition line than to the pivot there could be closer points
        # at the other branch so the function looks for them
        if (dist > np.absolute(dist_x)):
            get_nearest_neighbour(pivot,opp_branch,dim,dist_func,axis_key,i,best)
    return tuple(best)

def get_k_neighbours(pivot,kd_tree,k,dim,axis_key,i=0,best_k = None):
	"""Traverses a given kd-tree to find the k closest instances to the pivot. To do so, it
	recursively traverses the kd-tree looking for the closest instances and saving it in best_k.
	Input:  *pivot: instance for wich the function searches for the k nearest neighbours.
			*kd_tree: a kd-tree of instances of the same class than pivot.
			*k: desired number of neighbours
			*dim: dimension of axis_key, number of attributes considered to generate the kd-tree
			*dist_func: distance function.
			*axis_key: tuple containing the keys (as strings) of the class attributes that will be
			used as axis to make a k-d tree.
			*axis number in which the compairsons are done. By default, 0.
			*best_k: list containing the k (or less) best elements. By default, None.
	Output: *neighbours: list of k tuples as (distance(node,pivot),node) which contains the k
			closest instances and its distances to the pivot.
	"""

    # Error check

	if (best_k is not None) and (len(best_k) > k):
		raise IndexError('best_k should not have more than {} elements'.format(k))

	if not isinstance(dim, int):
		raise TypeError('dim should be an integer')
	if (len(axis_key) != dim):
		raise IndexError('axis_key lenght and dim sould be equal')

	if (i < 0) or (i > (len(axis_key)-1)):
		raise IndexError('i should be between 0 and {}'.format(len(axis_key)-1))
	if not isinstance(i, int):
		raise TypeError('i should be an integer')

	is_root = not best_k
	if is_root: # if there is no best_k, create one
		best_k = []

	if kd_tree is not None:
		# Error check
		if (pivot.__class__.__name__ != kd_tree.node.__class__.__name__):
			raise TypeError('pivot and node tree {} should belong to the same class'.format(kd_tree.node))
		if (axis_key[i] not in dir(pivot)):
			raise AttributeError('{} is not an attribute of the class of the pivot'.format(axis_key[i]))

		# dist is the distance from the pivot to the root node of the actual kd-tree
		# dist_x is the distance from the kd-tree partition line and the pivot
		dist = pivot.distance(kd_tree.node,*axis_key)
		dist_x = kd_tree.node.__getattribute__(axis_key[i]) - pivot.__getattribute__(axis_key[i])

		if len(best_k) < k: # if best_k has less than k instances, add the actual one
			best_k.append((dist, kd_tree.node))
			best_k.sort(key= lambda x: x[0])
		elif dist < best_k[k-1][0]: # if the actual instance is better than the worst in best_k
			best_k[k-1] = (dist, kd_tree.node)
			best_k.sort(key= lambda x: x[0])

		if (dist_x < 0): # if dist_x < 0 <-> pivot is at the right side
			next_branch = kd_tree.right
			opp_branch = kd_tree.left
		else: # if dist_x > 0 <-> pivot is at the left side
			next_branch = kd_tree.left
			opp_branch = kd_tree.right

		i = (i+1) % dim # next axis number

		# follow searching at the actual side of the partition (branch)
		get_k_neighbours(pivot,next_branch,k,dim,axis_key,i,best_k)

		# if pivot is closer to the partition line than to the pivot there could be closer points
		# at the other branch so the function looks for them
		if (dist > np.absolute(dist_x)):
			get_k_neighbours(pivot,opp_branch,k,dim,axis_key,i,best_k)
		if is_root:
			return best_k

