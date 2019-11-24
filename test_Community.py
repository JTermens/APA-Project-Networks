#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the distance function of the Community class
"""

import unittest
from networktools import Community,euclidean_distance
import numpy as np


class TestCommunity(unittest.TestCase):
    
    
    def setUp(self):
        self.c = Community(1,1,1,1,1,1,1,1,1,1)
        self.other = Community(2,2,2,2,2,2,2,2,2,2)
        
    
    def tearDown(self):
        pass
    
    def test_distance(self):
        
        # Testing that the output is correct for several valid inputs
        self.assertEqual(self.c.distance(self.c),0)
        self.assertEqual(self.c.distance(self.other),3.1622776601683795)
        self.assertEqual(self.c.distance(self.other,"size"),1.0)
        
        # If other is not a community instance
        self.assertRaises(TypeError,self.c.distance,3)
        self.assertRaises(TypeError,self.c.distance,False)
        self.assertRaises(TypeError,self.c.distance,"string")
        
        # If args aren't attributes of the community instance
        self.assertRaises(AttributeError,self.c.distance,self.other,"invented_attribute")
        self.assertRaises(AttributeError,self.c.distance,self.other,0)
        self.assertRaises(AttributeError,self.c.distance,self.other,False)
        self.assertRaises(AttributeError,self.c.distance,self.other,5.5)
        
        

if __name__ == 'main':
    unittest.main()