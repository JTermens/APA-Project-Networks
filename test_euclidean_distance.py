#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the euclidean_distance function
"""

import unittest
from APA_project_v5 import euclidean_distance
import numpy as np

class TestEuclidean_distance(unittest.TestCase):
    
    def setUp(self):
        self.p1 = [1,1,1]
        self.p2 = [2,2,2]
        self.p3 = [3,3,3,3]
        
    def test_distance(self):
        self.assertEqual(euclidean_distance(self.p1,self.p1),0)
        self.assertAlmostEqual(euclidean_distance(self.p1,self.p2),1.7320508075688772)
        
#    def test_values (self):
#        # If other is not a community instance
#        self.assertRaises(TypeError,euclidean_distance,False,self.p2)
#        self.assertRaises(TypeError,euclidean_distance,self.p1,False)
#        self.assertRaises(TypeError,euclidean_distance,"string",self.p2)
#        self.assertRaises(TypeError,euclidean_distance,self.p1,"string")
    
    def test_equal_length(self):
        self.assertRaises(IndexError,euclidean_distance,self.p1,self.p3)

if __name__ == 'main':
    unittest.main()