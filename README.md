# k-nearest Neighbour search of network communities in their K-d tree representation

The aim of this project is to implement k-nearest neighbour search algorithm based on the K-d tree representation of network communities extracted from a co-expression network. 

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

### Tests

The Unittest library was used to test the performance and exception handling of the main implemented functions (euclidean_distance, distance and group). Following the standard specifications, there is one file per tested function and we have used the filename convention: 'test_<function_name>.py'. To see the result of these tests run:

```$ python -m unittest test_<function_name>.py```
example:

```$ python -m unittest test_euclidean_distance.py```

## Steps

### 1. Obtain communities from the co-expression network
This step consists of extracting communities from the co-expression network. !! Explain what happens when the number of communities is small etc

### 2. Choose and compute ten features for each community
After generating communities, we extracted 10 features for each of them. The extracted features are: density, size, relative density, betweenness centrality, maximum betweenness centrality, average betweenness centrality, degree centrality, maximum degree centrality, average degree centrality, load centrality, maximum load centrality, average load centrality and community modularity within the network.

### 3. Build a 'Community' class with features as attributes and a defined distance method
We defined a community class that contains 10 attributes that correspond to the previously described features. Moreover, it contains a distance function that computes the distance from the current instance to another one based on the provided features and using the euclidean distance.  

### 4. Define a Kd tree class and implement a function that builds it


### 5. Implement a function that finds the k-nearest neighbours search on the Kd tree with specified arguments



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

