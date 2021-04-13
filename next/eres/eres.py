#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:54:36 2021

@author: hernando
"""

#import numpy as np

def dz_energy_correction(energy, dz, alpha = 2.76e-4):
    """ Apply Josh's energy correction by delta-z effect
    """
    return energy/(1 - alpha * dz)


#def energy_resolution(energy, p0, nbins, erange = None):
    
    