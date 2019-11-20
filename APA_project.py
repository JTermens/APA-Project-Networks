#--------------------------------------------------------------------------------------------------
# APA PROJECT: COMMUNITY SEARCH AND SORT
#
# This code is degined to generate and filter graphs, search communities on it with the Louvain
# algorithm, characterize the communities with 10 features and computes a distance between each
# pair. Features used:
#  
# To come: implementation of a ball-tree structure to perform fast sorting.	
# 
# Alda Sabalić, Beatriz Urda and Joan Térmens                  Last revision: 15th of November 2019
#--------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from pandas import DataFrame
import networkx as nx
from networkx.algorithms import community
import community
from math import sqrt
from functools import reduce

def filtered(filename,threshold = 0.55):
	'''
	Function: filtered
	This function generates a data frame from a given file (filename) and filters it removing the 
	edges with a weight lower than a given threshold.
	
	Input:  * filename: a tab separated file containing 3 columns (source node, target node and 
		weight) and a row for each edge
		* threshold: minimum weight that an edge has to have ho not being filtered.
	Output: * df_filtered: filtered dataframe, has 3 columns, keyed 'gene1', 'gene2' and 'edge' and
	 	a row for each edge.
	'''
	df = pd.read_csv(filename, sep="\t")
	df_filtered=df.loc[ (df['edge'] > threshold) & (df['gene1'] != df['gene2']) ]
	return df_filtered

def network(df):
	'''
	Function: network
	This function creates a grap, G, from a dataframe, df, using NetworkX library.
	
	Input:  * df: dataframe with 3 columns, keyed 'gene1', 'gene2' and 'edge' and a row for each 
		edge.
	Output: * G: Network graph
	'''
	G=nx.from_pandas_edgelist(df_filtered, 'gene1', 'gene2')
	return G

def louvain(G):
	'''
	Function: louvain
	This function implements the Louvain community search algorithm in a graph, G, and returns a 
	dictionary of the partitions.
	Input:  * G: a graph generated with NetworkX
	Output: * dic_nodes: a dictionary like {community_number:'node_1 | ... | node_N'}
	'''
	# Starting with an initial partition of the graph and running the Louvain algorithm for Community Detection
	partition=community.best_partition(G, weight='MsgCount')

	values=[partition.get(node) for node in G.nodes()]
	list_com=partition.values()

	# Creating a dictionary like {community_number:list_of_participants}
	dict_nodes={}

	# Populating the dictionary with items
	for each_item in partition.items():
	    community_num=each_item[1]
	    community_node=each_item[0]
	    if community_num in dict_nodes:
	        value=str(dict_nodes.get(community_num)) + ' | ' + str(community_node)
	        dict_nodes.update({community_num:value})
	    else:
	        dict_nodes.update({community_num:community_node})

	# Creating a new graph to represent the communities created by the Louvain algorithm
	G_comm=nx.Graph()

	# Populating the data from the node dictionary created earlier
	G_comm.add_nodes_from(dict_nodes)

	# Calculating modularity and the total number of communities
	mod=community.modularity(partition,G)
	print("Total number of Communities=", len(G_comm.nodes()))
	print("Modularity: ", mod)

	return dict_nodes


def euclidean_distance(a,b):
    '''Returns euclidean distance for vector of dimension n>=2'''
    if(len(a) != len(b)):
        return "error"
    else:
        dim = len(a)
        return sqrt(reduce(lambda i,j: i + ((a[j] - b[j]) ** 2), range(dim), 0))

# Community class, each comunity (parition) found with Louvain algorithm is characterized by 10
# features and stored as an instance of this class.
class Community(object):
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
        
    
        
    def distance(self,other,*argv):
        '''Returns the distance between the current community and other instance of
        the class using the attributes specified in argv. If no arg is provided,
        it uses all the attributes'''
        if (len(argv) == 0):
            features1 = (self.dens,self.size,self.rel_dens,self.max_btw, self.avg_btw, self.max_centr, self.avg_centr, self.max_load, self.avg_load, self.mod)
            features2 = (self.dens,self.size,self.rel_dens,self.max_btw, self.avg_btw, self.max_centr, self.avg_centr, self.max_load, self.avg_load, self.mod)
            return (euclidean_distance(features1,features2))
        else:
            features1 = list()
            features2 = list()
            for arg in argv:
                if(arg in dir(self)):
                    features1.append(self.__getattribute__(arg))
                    features2.append(other.__getattribute__(arg))
                else:
                    print("%s is not an attribute of the communities" %(arg))
            features1 = tuple(features1)
            features2 = tuple(features2)
            return (euclidean_distance(features1,features2))


def ComputeFeatures(set_nodes,df_network,key1='node1',key2='node2',weight = None):
    '''
    Function: ComputeFeatures
    This function computes the following features of a given community in a graph: (1) density, 
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
     '''

    df_comm = df_network[(df_network[key1].isin(set_nodes)) & (df_network[key2].isin(set_nodes))] # community data frame
    g_comm = nx.from_pandas_edgelist(df_comm,key1,key2,edge_attr = weight) # community graph
    g_network = nx.from_pandas_edgelist(df_network,key1,key2,edge_attr = weight) # network graph

    dict_comm = {node:0 if node in set_nodes else 1 for node in g_network.nodes()} # necessary to calculate the modularity

    # Features:

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

# Tree class, implements an object with 3 atributes: node is the value of the tree node, left is the left branch and right, 
# the right one
class Tree(object):
	def __init__(self,node,left=None,right=None):
		self.node = node
		self.left = left
		self.right = right

def MakeKdTree(list_inst,i=0,list_features = None):
    '''
    Function: MakeKdTree
    This function recursively takes a list of instances and a list of attributes (or features) and generates a k-d tree using the 
    specified features as the different axis. By default the function considers list_inst a list of Community class instances and 
    considers much instances as possible (with max 10) to ensure that N > 2^k.
    
    Input:  *list_inst: a list of instances of a class
            *i: axis number in which the list_inst is sorted and paritioned. by default, 0
	    *list_features: list of the features/attributes that will be used as axis to generate the K-d tree. By default, as
	    much instances as possible (with max 10) to ensure that N > 2^k
    Output: *tree: binary tree, instance of class Tree
    '''
    if len(list_inst) > 1:
        if (list_features == None):
            axis_key = ('dens','size','rel_dens','max_btw', 'avg_btw', 'max_centr', 'avg_centr', 'max_load', 'avg_load', 'mod')
	    if (len(list_inst) < 2**(10)): # ensures N > 2^k, so the algorithm remains efficient
	        optimal_num_features = int(np.log2(len(list_inst)))
		axis_key = axis_key[:optimal_num_features]
        elif(i == 0): # ensures the attribute checking its just done at the first iteration
            axis_key = list()
            for feature in list_features:
                if (feature in dir(list_inst[0])):
                    axis_key.append(feature)
                else:
                    print("%s is not an attribute of the instances given" %(feature))
                axis_key = tuple(axis_key)

        dim = len(axis_key)
        list_inst.sort(key=lambda x: x.__getattribute__(axis_key[i])) # pyhton list.sort has complexity O(nlog(n))
        i = (i+1) % dim
        half = len(list_inst) >>1 # just division by 2
        return Tree(list_inst[half], MakeKdTree(list_inst[:half],i,list_features),MakeKdTree(list_inst[half+1:],i,list_features))

    elif len(list_inst) == 1:
        return Tree(list_inst[0])

if __name__=='__main__': 

	#df = pd.read_csv("4_coexpr_geo_v2.txt", sep="\t")
    df_filtered = filtered("4_coexpr_geo_v2.txt")
    print('Building network...')
    network = network(df_filtered)
    print('Community detection in course...')
    dict_nodes = louvain(network)
    print('Extracting features...')
    community_list = []
    for comm_num in dict_nodes.keys():
	#for comm_num in range(1,len(G_comm.nodes())):
        set_nodes = dict_nodes[comm_num].split(' | ')
        community_list.append(ComputeFeatures(set_nodes,df_filtered,'gene1','gene2','edge'))
    print('Feature extraction completed...')
    x1 = community_list[1].distance(community_list[1])
    x2 = community_list[1].distance(community_list[3])
    x3 = community_list[1].distance(community_list[2],"size")
    print("The first community has %s distance from itself" %(x1))
    print("The first community has %s distance from the second one" %(x2))
    print("The first community has %s distance from the second one" %(x3))
    print("Generating a k-dim tree...")
    tree = MakeKdTree(community_list)
    
    
    
    
#    a = (1,1,1)
#    b = (2,2,2)
#    x = euclidean_distance(a,b)
#    print(x)
    
    
