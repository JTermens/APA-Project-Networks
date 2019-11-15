# Community detection in co-expression networks

Program that from the selected file contructs a network and detects present communities by applying Louvain's algorithm of community detection. 


## Getting Started

It is important to notice the network consists of nodes and edges. The nodes are represented by the genes while the edges are computed using liner correlation and represent correlation values. Notice that you can choose the threshold of the value of correlation when constructing the network. In the case you don't specify it differently, the threshold is set to 0.55.
Here we give a figure representing the density distribution of the correlation values that might help you decide your threshold.Bear in mind that the lower the selected threshold the longer the time of execution of the program.

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

