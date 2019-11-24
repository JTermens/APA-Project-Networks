# K-nearest neighbour search of network communities using K-d trees

The aim of this project is to implement k-nearest neighbour search algorithm based on the K-d tree representation of network communities extracted from a co-expression network. 

## Application details
The co-expression network used in this project consists of nodes and edges. The nodes are represented by the genes while the edges are computed by using liner correlation and represent correlation values between each pair of genes. In order for the script to work correctly, a file containing an adjacency matrix of the network must be provided. 

Notice that you can choose the threshold of the value of correlation when constructing the network. In the case you don't specify it differently, the threshold is set to 0.38. However, it is very important to bear in mind that the number of generated communities depends directly on the threshold and that changing the threshold can affect your neighbour search.
Here we give a figure representing the density distribution of the correlation values that might be used as an orientative guideline in the case the threshold needs to be changed.

[![INSERT YOUR GRAPHIC HERE](https://github.com/JTermens/APA-Project-Networks/blob/master/distribution.png)]()


## Getting Started
The code is divided in 3 sections: the main program, a library and a collection of tests. The main script is the file `main.py`; it executes the implemented functions and structures to perform a k-nearest neighbour search based on the co-expression matrix. It uses the library `networktools.py`, that contains the collection of classes and functions developed during the project (i.e. function to build a K-d tree, the Community instance with the distance function, and the Tree instance with its group function). Take a look at the tests and profiling section below to get more information about the tests. 

!! mention also the coverage folder that will be generated for the tests!!!

### Prerequisites

There are some Python 3 libraries that you will need to install before running our code. 
We list them here:

```
Numpy
Pandas
Networkx
Community
Math
Functools
```

### Tests and Profiling

The Unittest library was used to test the performance and exception handling of the main implemented functions (`euclidean_distance`, `distance` and `group`). Following the standard specifications, there is one file per tested function and we have used the filename convention `test_<function_name>.py` for conventional functions and `test_<container_name>.py` for the ones that belong to a given class. To perform the tests you have to run:

```$ python -m unittest test_<function/container_name>.py```

Specifically,

Euclidean function: ```$ python -m unittest test_euclidean_distance.py```

Distance function: ```$ python -m unittest test_Community.py```

Group function: ```$ python -m unittest test_Tree.py```

On the other hand, a profiling was done using cProfiler for the functions `make_kd_tree`, `get_nearest_neighbour`, and both implementations of k-neighbour search (`get_k_neighbours_heap` and `get_k_neighbours_eq`) benchmarking. Its implementation and results could be found at the [test folder](https://github.com/JTermens/APA-Project-Networks/blob/master/tests/). Furthermore, the functions found in the main code have been profiled too with cProfiler, results could be found at `main_profiling.txt`.

## Objectives and followed steps

### 1. Obtain communities from the co-expression network
This step consists of extracting communities from the co-expression network. In order to extract the communities from a given file, the algorithm calls the function `choose_detection_method(filename,k)` where the argument `filename` specifies the file from which the communities need to be extracted and the argument `k` controls the number of features the user wants to perform the k-neigbour search. 

In the case the user plans to perform the k-neighbour search by considering more than 8  features of the communities (i.e. the arguement `k` of the function is larger than 8), the algorithm calls the function `community_binning(filename)` that will randomly select 2000 communities from the desired file. 

However if the user wants to perform the k-neighbour search by considering 8 or less features of the communities (i.e. the arguement `k` of the function is smaller or equal to 8), the communities are found by filtering the chosen file with the `filtered` function , building the newtwork with the `network` function and finally, the communities are computed by using the `louvain(G)` function that implements the standard Louvain algorithm of community detection.

### 2. Choose and compute ten features for each community
After generating communities, we extracted 10 features for each of them. The extracted features are: density, size, relative density, maximum betweenness centrality, average betweenness centrality, maximum degree centrality, average degree centrality, maximum load centrality, average load centrality and community modularity within the network.

### 3. Build a 'Community' class with features as attributes and a defined distance method
We defined a community class that contains 10 attributes that correspond to the previously described features. Moreover, it contains a distance function that computes the distance from the current instance to another one based on the provided features and using the euclidean distance.  

### 4. Define a tree class and implement a function that builds a k-d tree
To fast neighbour search, we opted for implementing a k-d tree structure for which a class `Tree` is defined with attributes `node`, `left`, for the left subtree,`right`, for the right one and `axis_key`, which is a tuple containing the keys (as strings) of the class (`Community` in this case) attributes that have been used as axis to make a k-d tree. K-d trees are created as instances of the `Tree` class and using the function `make_kd_tree`. This function firstly generates `axis_key` from the given arguments or, by defauly, computes the optimal number of attributes to ensure efficiency (N > 2^k) and generates `axis_key` with the first ones. Then, recursively generates the k-d tree by successively splitting the given group of instances according to the different axis.

### 5. Implement a function that finds the k-nearest neighbours on the K-d tree
The function `get_k_neighbours` traverses a given kd-tree to find the k closest instances to the given one (called `pivot`). To do it, it successively traverses the kd-tree branckes looking for the closest instances and saving it the list `best_k`. This list contains tuples of the form (distance(pivot, instance), instance) sorted from the minimal to the maximal distance. This implementation using lists has been chosen, instead of the implementation using heaps, due to being able to compare equal distances and for being faster. Both implementations where benchamarked using one million 2-dim random points generated in the unit circle and looking for the thousand closest point to the origin. Its profiling showed that the list implementation was nearly 0.7 seconds faster than the heap one (23.513 and 24.229 seconds, respectively).

## Limitations

---------------rewrite, this is a template, mention optimality and k features------
The current version of Code Maat processes all its content in memory. Thus, it may not scale to large input files (however, it depends a lot on the combination of parser and analysis). The recommendation is to limit the input by specifying a sensible start date (as discussed initially, you want to do that anyway to avoid confounds in the analysis).

## Authors

* **Alda Sabalić** 
* **Joan Termens** 
* **Beatriz Urda** 




