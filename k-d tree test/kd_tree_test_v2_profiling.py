import numpy as np
from math import sqrt
from functools import reduce
import random as rd
import cProfile

class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def get_x(self):
        return self.x

    def get_y(self,x):
        return self.y,x

    def position(self,func1,func2,data):
        x = func1()
        return x,func2(data)

    def distance(self,other,*argv):
        attr_self = []
        attr_other = []
        for arg in argv:
            if(arg in dir(self)):
                attr_self.append(self.__getattribute__(arg))
                attr_other.append(other.__getattribute__(arg))
            else:
                print("%s is not an attribute of the communities"%(arg))
        attr_self = tuple(attr_self)
        attr_other = tuple(attr_other)

        dim = len(attr_self)
        return sqrt(reduce(lambda i,j: i + ((attr_self[j] - attr_other[j]) ** 2), range(dim), 0))

def make_inst_list(point_list):
    list_inst = []
    for point in point_list:
        inst_point = Point(point[0],point[1])
        list_inst.append(inst_point)
    return list_inst

def cartesian_points(N):
    """
    This function returns a list with N random 2-dim points inside the unit circle. The random
    coordinates are x and y
    """
    n = 0
    points = []
    while n < N:
        x = rd.choice([1,-1])*rd.random() # random generation of points
        y = rd.choice([1,-1])*rd.random()
        if (x**2 + y**2)**(1/2) <= 1:
            points.append([x,y])
            n = n+1
        else:
            continue

    return points


# Tree class, implements an object with 3 atributes: node is the value of the tree node, left is the left branch and right,
# the right one
class Tree(object):
	def __init__(self,node,left=None,right=None):
		self.node = node
		self.left = left
		self.right = right

def recursive_kd_tree(list_inst,axis_key,i=0):
	'''
	Function: RecursiveKdTree
	This function creates a K-d tree recursively from a list of instances (list_inst) and axis_key,
	a tuple containing the keys (as strings) of the class attributes that will be used as axis.
	Input:  *list_inst: a list of instances of a class
			*axis_key: tuple containing the keys (as strings) of the class attributes that will be
			used as axis.
			*i: axis number in which the list_inst is sorted and paritioned. By default, 0.
	Output: *instance of Tree class where the node is the median point, left calls to create a left
			subtree and right, a right one.
	'''
	if len(list_inst) > 1:
		dim = len(axis_key)
		list_inst.sort(key=lambda x: x.__getattribute__(axis_key[i])) # pyhton list.sort has complexity O(nlog(n))
		i = (i + 1) % dim
		half = len(list_inst) >>1 # just division by 2 moving the bits

		return Tree(list_inst[half], recursive_kd_tree(list_inst[:half],axis_key,i),recursive_kd_tree(list_inst[half+1:],axis_key,i))

	elif len(list_inst) == 1:
		return Tree(list_inst[0])

def get_axis_key(list_inst,list_attr = None):
	'''
	Function: get_Axis_key
	This function checks if the given instances are from the same class and creates axis_key, a
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
	'''
	# chek if all instances belong to the same class
	if(len(list_inst) > 0):
		inst_class = list_inst[0].__class__ # class of the first instance
		for inst in list_inst:
			# if an instance isn't from the same class as the 1st one, raise an error
			if(not isinstance(inst,inst_class)):
				print("Error: Not all instances belong to the same class")

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
				print("%s is not an attribute of the instances given" %(attr))
		axis_key = tuple(axis_key)

	return axis_key

def make_kd_tree(list_inst,list_attr = None):
	'''
	Function: make_kd_tree
	This function recursively takes a list of instances and a list of attributes and generates a
	k-dim tree using the specified features as the different axis. Firstly calls to GetKeyAxis to
	generate the axis_key that is a tuple containing the keys (as strings) of the class attributes
	that will be used as axis. Then, calls to recursive_kd_tree which recursively generates the tree.
	Input:  *list_inst: a list of instances of a class
			*list_attr: list of attributes of the given instances, by default, None
	Output: *kd_tree: k-dim binary tree, instance of class Tree
            *axis_key: tuple containing the keys (as strings) of the class attributes that will be
        	used as axis to make a k-d tree.
	'''

	axis_key= get_axis_key(list_inst,list_attr)
	kd_tree = recursive_kd_tree(list_inst,axis_key)

	return kd_tree,axis_key


def get_nearest_neighbour(pivot,kd_tree,dim,axis_key,i=0,best=None):
    '''
    Function: get_nearest_neighbour
    This function traverses a given kd-tree to find the closest instance to the pivot. To do it,
    recursively traverses the kd-tree looking for the closest instance and saving it as best.
    Input:  *pivot: instance for wich the function searches for the nearest neighbour.
            *kd_tree: a kd-tree of instances of the same class than pivot.
            *dim: dimension of axis_key, number of attributes considered to generate the kd-tree
            *axis_key: tuple containing the keys (as strings) of the class attributes that will be
        	used as axis to make a k-d tree.
            *axis number in which the compairsons are done. By default, 0.
            *best: [minimum distance, closer instance].
    Output: *best: tuple(minimum distance, closer instance).
    '''

    if kd_tree is not None:

        # dist is the distance from the pivot to the root node of the actual kd-tree
        # dist_x is the distance from the kd-tree partition line and the pivot
        dist = pivot.distance(kd_tree.node,*axis_key)
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
        get_nearest_neighbour(pivot,next_branch,dim,axis_key,i,best)

        # if pivot is closer to the partition line than to the pivot there could be closer points
        # at the other branch so the function looks for them
        if (dist > np.absolute(dist_x)):
            get_nearest_neighbour(pivot,opp_branch,dim,axis_key,i,best)
    return tuple(best)

def get_k_neighbours_heap(pivot,kd_tree,k,dim,axis_key,i=0,heap=None):
    '''
    Function: get_k_neighbours_heap *CANNOT HANDLE COMPAIRSONS BETWEEN EQUAL INSTANCES*
    This function traverses a given kd-tree to find the k closest instances to the pivot. Firstly
    checks that the kd-tree's root node and the pivot are instances of the same class and then
    recursively tranveses the kd-tree looking for the closest instances and saving it in a heap.
    Input:  *pivot: instance for wich the function searches for the k nearest neighbours.
            *kd_tree: a kd-tree of instances of the same class than pivot.
            *k: desired number of neighbours
            *dim: dimension of axis_key, number of attributes considered to generate the kd-tree
            *axis_key: tuple containing the keys (as strings) of the class attributes that will be
        	used as axis to make a k-d tree.
            *axis number in which the compairsons are done. By default, 0.
            *heap: max heap containing the k closest instances. By default, None.
    Output: *neighbours: list of k tuples as (distance(node,pivot),node) which contains the k
            closest instances and its distances to the pivot.
    '''
    import heapq

    is_root = not heap
    if is_root: # if there is no heap, create one
        heap = []

    if kd_tree is not None:

        pivot_class = pivot.__class__
        if not isinstance(kd_tree.node,pivot_class): #checks if pivot and node are from the same class
            print("Error: the pivot and the elements of the k-d tree must be instances of the same class")

        # dist is the distance from the pivot to the root node of the actual kd-tree
        # dist_x is the distance from the kd-tree partition line and the pivot
        dist = pivot.distance(kd_tree.node,*axis_key)
        dist_x = kd_tree.node.__getattribute__(axis_key[i]) - pivot.__getattribute__(axis_key[i])

        if len(heap) < k: # if the heap has less than k instances, add the actual one
            # distances in the heap are negative so that the farests instances will leave first
            heapq.heappush(heap, (-dist, kd_tree.node)) # add to the heap
        elif dist < -heap[0][0]: # if the actual instance is better than the worst in the heap
            heapq.heappushpop(heap, (-dist, kd_tree.node)) # change the worst for the actual one

        if (dist_x < 0): # if dist_x < 0 <-> pivot is at the right side
            next_branch = kd_tree.right
            opp_branch = kd_tree.left
        else: # if dist_x > 0 <-> pivot is at the left side
            next_branch = kd_tree.left
            opp_branch = kd_tree.right

        i = (i+1) % dim # next axis number

        # follow searching at the actual side of the partition (branch)
        get_k_neighbours_heap(pivot,next_branch,k,dim,axis_key,i,heap)

        # if pivot is closer to the partition line than to the pivot there could be closer points
        # at the other branch so the function looks for them
        if (dist > np.absolute(dist_x)):
            get_k_neighbours_heap(pivot,opp_branch,k,dim,axis_key,i,heap)
    if is_root:
        neighbours = [(-h[0], h[1]) for h in heap] # dump the heap onto a list
        neighbours.sort(key = lambda x: x[0])
        return neighbours

def get_k_neighbours_eq(pivot,kd_tree,k,dim,axis_key,i=0,best_k=None):
    '''
    Function: get_k_neighbours_eq *CAN HANDLE COMPAIRSONS BETWEEN EQUAL INSTANCES*
    This function traverses a given kd-tree to find the k closest instances to the pivot. to do it
    recursively traverses the kd-tree looking for the closest instances and saving it in best_k.
    Input:  *pivot: instance for wich the function searches for the k nearest neighbours.
            *kd_tree: a kd-tree of instances of the same class than pivot.
            *k: desired number of neighbours
            *dim: dimension of axis_key, number of attributes considered to generate the kd-tree
            *axis_key: tuple containing the keys (as strings) of the class attributes that will be
        	used as axis to make a k-d tree.
            *axis number in which the compairsons are done. By default, 0.
            *best_k: list containing the k (or less) best elements. By default, None.
    Output: *best_k
    '''

    is_root = not best_k
    if is_root: # if there is no best_k, create one
        best_k = []

    if kd_tree is not None:
        # dist is the distance from the pivot to the root node of the actual kd-tree
        # dist_x is the distance from the kd-tree partition line and the pivot
        dist = pivot.distance(kd_tree.node,*axis_key)
        dist_x = kd_tree.node.__getattribute__(axis_key[i]) - pivot.__getattribute__(axis_key[i])

        if len(best_k) < k: # if best_k has less than k instances, add the actual one
            best_k.append((dist, kd_tree.node))
            best_k.sort(key = lambda x: x[0])
        elif dist < best_k[k-1][0]: # if the actual instance is better than the worst in best_k
            best_k[k-1] = (dist, kd_tree.node)  # change the worst for the actual one
            best_k.sort(key = lambda x: x[0])

        if (dist_x < 0): # if dist_x < 0 <-> pivot is at the right side
            next_branch = kd_tree.right
            opp_branch = kd_tree.left
        else: # if dist_x > 0 <-> pivot is at the left side
            next_branch = kd_tree.left
            opp_branch = kd_tree.right

        i = (i+1) % dim # next axis number

        # follow searching at the actual side of the partition (branch)
        get_k_neighbours_eq(pivot,next_branch,k,dim,axis_key,i,best_k)

        # if pivot is closer to the partition line than to the pivot there could be closer points
        # at the other branch so the function looks for them
        if (dist > np.absolute(dist_x)):
            get_k_neighbours_eq(pivot,opp_branch,k,dim,axis_key,i,best_k)
    if is_root:
        return best_k

def display_kd_tree(kd_tree,i=0):
    print(i*'\t','({},{})'.format(kd_tree.node.x,kd_tree.node.y))
    i += 1
    if (kd_tree.left is not None):
        display_kd_tree(kd_tree.left,i)
    if (kd_tree.right is not None):
        display_kd_tree(kd_tree.right,i)

if __name__ == '__main__':

    num_points = 10**6
    list_inst = make_inst_list(cartesian_points(num_points))
    pivot = Point(0,0)
    num_neighbours = 1000
    dim = 2

    print('Prolifling with {} points and {} neighbours'.format(num_points,num_neighbours))
    print('\n make_kd_tree')
    cProfile.runctx('make_kd_tree(list_inst)', {'list_inst':list_inst, 'make_kd_tree':make_kd_tree}, {})
    kd_tree,axis_key = make_kd_tree(list_inst)
    #display_kd_tree(kd_tree_1)

    print('\n get_nearest_neighbour')
    cProfile.runctx('get_nearest_neighbour(pivot,kd_tree,dim,axis_key)', {'pivot':pivot, 'kd_tree': kd_tree, 'dim':dim, 'axis_key':axis_key, 'get_nearest_neighbour':get_nearest_neighbour}, {})
    nn = get_nearest_neighbour(pivot,kd_tree,dim,axis_key)

    print('\n get_k_neighbours_heap')
    cProfile.runctx('get_k_neighbours_heap(pivot,kd_tree,k,dim,axis_key)', {'pivot':pivot, 'kd_tree': kd_tree, 'k':num_neighbours, 'dim':dim, 'axis_key':axis_key, 'get_k_neighbours_heap':get_k_neighbours_heap}, {})
    nn_k_heap = get_k_neighbours_heap(pivot,kd_tree,num_neighbours,dim,axis_key)

    print('\n get_k_neighbours_eq')
    cProfile.runctx('get_k_neighbours_eq(pivot,kd_tree,k,dim,axis_key)', {'pivot':pivot, 'kd_tree': kd_tree, 'k':num_neighbours, 'dim':dim, 'axis_key':axis_key, 'get_k_neighbours_eq':get_k_neighbours_eq}, {})
    nn_k_eq = get_k_neighbours_eq(pivot,kd_tree,num_neighbours,dim,axis_key)

    print('\n Results:')

    print('\n Nearest Neighbour')
    print('dist = {}; point =({},{})'.format(nn[0],nn[1].x,nn[1].y))

    print('\n {} Nearest Neighbours with heap'.format(num_neighbours))
    print('Display the first {}'.format(int(num_neighbours/10)))
    for n in range(0,int(num_neighbours/100)):
        print('dist = {}; point =({},{})'.format(nn_k_heap[n][0],nn_k_heap[n][1].x,nn_k_heap[n][1].y))

    print('\n {} Nearest Neighbours without heap'.format(num_neighbours))
    print('Display the first {}'.format(int(num_neighbours/10)))
    for n in range(0,int(num_neighbours/100)):
        print('dist = {}; point =({},{})'.format(nn_k_eq[n][0],nn_k_eq[n][1].x,nn_k_eq[n][1].y))
