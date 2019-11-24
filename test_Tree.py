#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the group function of the Tree class
"""

import unittest
from networktools import Community,euclidean_distance, Tree, make_kd_tree
import numpy as np


class TestTree(unittest.TestCase):
    
    
    def setUp(self):
        self.t1 = Tree(1,1,1)
        self.t2 = Tree(1,2,2)
        
        self.c = Community(1,1,1,1,1,1,1,1,1,1)
        self.c2 = Community(2,2,2,2,2,2,2,2,2,2)
        self.tree = make_kd_tree([self.c2,self.c2,self.c])
        self.tree2 = make_kd_tree([self.c2,self.c2,self.c2])
    
    def test_group(self):
        
        # Testing that the output is correct for several valid inputs
        self.assertEqual(self.tree.group(self.c,0,1,'dens'),[self.c])
        
        # Checking if x is not a Community
        self.assertRaises(TypeError,self.t1.group,2,1,3)
        self.assertRaises(TypeError,self.t1.group,False,1,3)
        self.assertRaises(TypeError,self.t1.group,"string",1,3)
        
        # Checking if d is not a real number
        self.assertRaises(TypeError,self.t1.group,self.c,False,3)
        self.assertRaises(TypeError,self.t1.group,self.c,"string",3)
        
        # Checking if k is not an integer number
        self.assertRaises(TypeError,self.t1.group,self.c,1,False)
        self.assertRaises(TypeError,self.t1.group,self.c,1,"string")
        self.assertRaises(TypeError,self.t1.group,self.c,1,3.3)
        
        # Checking if d is non-negative
        self.assertRaises(ValueError,self.t1.group,self.c,-7,3)
        
        # Checking if k is non-negative
        self.assertRaises(ValueError,self.t1.group,self.c,1,-3)
        
        # If args aren't attributes of the community instance
        self.assertRaises(AttributeError,self.t1.group,self.c,1,3,"invented_attribute")
        self.assertRaises(AttributeError,self.t1.group,self.c,1,3,"invented_attribute","also_invented")
        
        # Number of dimensions to build the tree and to perform neighbour-search must match
        self.assertRaises(IndexError,self.tree.group,self.c,0,1)
        
        # There aren't enough nearest neighbours for that distance
        self.assertRaises(ValueError,self.tree2.group,self.c,0,1,'dens')
        

if __name__ == 'main':
    unittest.main()