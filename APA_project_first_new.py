import pandas as pd
import numpy as np
from pandas import DataFrame
import networkx as nx
from networkx.algorithms import community
import community
from math import sqrt
from functools import reduce



def filtered(filename,threshold = 0.55):
	df = pd.read_csv(filename, sep="\t")
	df_filtered=df.loc[ (df['edge'] > threshold) & (df['gene1'] != df['gene2']) ]
	#print(df_filtered)
	return df_filtered

def network(df):
	G=nx.from_pandas_edgelist(df_filtered, 'gene1', 'gene2')
	#print(len(G.nodes()))
	return G
#-----------------------------------------
def louvain(G):
	# Starting with an initial partition of the graph and running the Louvain algorithm for Community Detection
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
	#matplotlib.rcParams['figure.figsize']= [12, 8]
	G_comm=nx.Graph()
	#print('im here')
	

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

def euclidean_distance(a,b):
    """Returns euclidean distance for vector of dimension n>=2"""
    if(len(a) != len(b)):
        return "error"
    else:
        dim = len(a)
        return sqrt(reduce(lambda i,j: i + ((a[j] - b[j]) ** 2), range(dim), 0))

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
        """Returns the distance between the current community and other instance of
        the class using the attributes specified in argv. If no arg is provided,
        it uses all the attributes"""
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
    
    
    
    
#    a = (1,1,1)
#    b = (2,2,2)
#    x = euclidean_distance(a,b)
#    print(x)
    
    
