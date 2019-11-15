#--------------------------------------------------------------------------------------------------
# APA PROJECT: COMMUNITY SEARCH AND SORT
#
# This code is degined to generate and filter graphs, search communities on it with the Louvain
# algorithm, characterize the communities with 10 features and computes a distance between each
# pair. Features used:
#  
# To come: implementation of a ball-tree structure to perform fast sorting.	
# 
# Alda Sabalić, Bea Urda and Joan Térmens                      Last revision: 15th of November 2019
#--------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from pandas import DataFrame
import networkx as nx
from networkx.algorithms import community
import community



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
	#print(df_filtered)
	return df_filtered

def network(df):
	'''
	Function: network
	This function creates a grap, G, from a dataframe, df, using NetworkX library.
	
	Input:  * df: dataframe with 3 columns, keyed 'gene1', 'gene2' and 'edge' and a row for each 
		edge.
	Output: * G: Network graph
	'''
	G=nx.from_pandas_edgelist(df_filtered, 'gene1', 'gene2', edge_attr='edge')
	#print(len(G.nodes()))
	return G

#--------------------------------------------------------------------------------------------------

def louvain(G):
	'''
	Function: louvain
	This function implements the Louvain community search algorithm in a graph, G, and returns a 
	dictionary of the partitions.

	Input:  * G: a graph generated with NetworkX
	Output: * dic_nodes: a dictionary like {community_number:'node_1 | ... | node_N'}
	'''
	# Starting with an initial partition of the graph and running the Louvain algorithm 
	# for Community Detection
	partition=community.best_partition(G, weight='MsgCount')
	#print('Completed Louvain algorithm .. . . ' )
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
	# matplotlib.rcParams['figure.figsize']= [12, 8]
	G_comm=nx.Graph()
	# print('im here')


	# Populating the data from the node dictionary created earlier
	G_comm.add_nodes_from(dict_nodes)
	#print(G_comm.nodes())
	# Calculating modularity and the total number of communities
	mod=community.modularity(partition,G)
	print("Total number of Communities=", len(G_comm.nodes()))
	print("Modularity: ", mod)

	#print(dict_nodes.keys())
	return dict_nodes
	# Creating the Graph and also calculating Modularity
	#pos_louvain=nx.spring_layout(G_comm)

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


if __name__=='__main__':

	#df = pd.read_csv("4_coexpr_geo_v2.txt", sep="\t")
	df_filtered = filtered("4_coexpr_geo_v2.txt")
	print('Building network...')
	network = network(df_filtered)
	print('Community detection in course...')
	dict_nodes = louvain(network)
	print('Extracting features...')
	for comm_num in dict_nodes.keys():
	#for comm_num in range(1,len(G_comm.nodes())):
	    set_nodes = dict_nodes[comm_num].split(' | ')
	    ComputeFeatures(set_nodes,df_filtered,'gene1','gene2','edge')
	print('Feature extraction completed...')
