# K-nearest neighbour search of network communities in their K-d tree representation

The aim of this project is to implement k-nearest neighbour search algorithm based on the K-d tree representation of network communities extracted from a co-expression network. 

## Application details
In this sc to notice the network consists of nodes and edges. The nodes are represented by the genes while the edges are computed using liner correlation and represent correlation values between each pair of genes. In order for the script to work correctly, a file containing an adjacency matrix of the network must be provided. 

Notice that you can choose the threshold of the value of correlation when constructing the network. In the case you don't specify it differently, the threshold is set to 0.38. 
Here we give a figure representing the density distribution of the correlation values 
[![INSERT YOUR GRAPHIC HERE](https://github.com/JTermens/APA-Project-Networks/blob/master/distribution.png)]()


## Getting Started
!! Here we should say that the objects are in one file named 'X', the functions in the file named '', the main here. 


### Prerequisites

There are some Python 3 libraries that you will need to install before running our code. 
We list them here:

```
Numpy
Pandas
Networkx
Community
```

### Tests and Profiling

The Unittest library was used to test the performance and exception handling of the main implemented functions (`euclidean_distance`, `distance` and `group`). Following the standard specifications, there is one file per tested function and we have used the filename convention: `test_<function_name>.py`. To see the result of these tests run:

```$ python -m unittest test_<function_name>.py```

example:

```$ python -m unittest test_euclidean_distance.py```

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

## Code limitations

## Authors

* **Alda Sabalić** 
* **Joan Termens** 
* **Beatriz Urda** 



# WHAT  WE HAD BEFORE

## Community detection in co-expression networks

Program that from the selected file contructs a network and detects present communities by applying Louvain's algorithm of community detection. Also, it extracts some main features of each community and computes the distance between them.
The extracted features are: density, size, relative density, betweenness centrality, maximum betweenness centrality, average betweenness centrality, degree centrality, maximum degree centrality, average degree centrality, load centrality, maximum load centrality, average load centrality and community modularity within the network.


## Getting Started

It is important to notice the network consists of nodes and edges. The nodes are represented by the genes while the edges are computed using liner correlation and represent correlation values. Notice that you can choose the threshold of the value of correlation when constructing the network. In the case you don't specify it differently, the threshold is set to 0.55.
Here we give a figure representing the density distribution of the correlation values that might help you decide your threshold. Bear in mind that the lower the selected threshold the longer the time of execution of the program.

[![INSERT YOUR GRAPHIC HERE](https://github.com/JTermens/APA-Project-Networks/blob/master/distribution.png)]()

### Prerequisites

There are some Python 3 libraries that you will need to install before running our code. 
We list them here:

```
Numpy
Pandas
Networkx
Community
```

## Authors

* **Alda Sabalić** 
* **Joan Termens** 
* **Beatriz Urda** 

