#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:54:36 2021

@author: hernando
"""

import numpy as np

import matplotlib.pyplot as plt

import hipy.cfit   as cfit
import hipy.histos as histos
import hipy.pltext as pltext


#import numpy as np

def dz_energy_correction(energy, dz, alpha = 2.76e-4):
    """ Apply Josh's energy correction by delta-z effect
    """
    return energy/(1 - alpha * dz)


#def energy_resolution(energy, p0, nbins, erange = None):
    
def dz_effect(dz, ene, nbins = 10, p0s = None, plot = False):
    
    bins  = np.percentile(dz, np.linspace(0., 100., nbins))
    counts, xedges, ipos = histos._profile(dz, ene, 'count', bins)
    nbins = len(bins)
    ys = [ene[ipos == i] for i in range(1, nbins)]

    epars, eupars = [], []
    #p0 = (10., 0.71, 0.02, 70., -70.)
    subplot = pltext.canvas(nbins, 2) if plot else None
    for i in range(nbins - 1):
        hfit = pltext.hfit if plot else histos.hfit
        if (plot): subplot(i + 1)
        p0 = None if p0s is None else p0s[i]
        _, _, _, pars, upars, _ = hfit(ys[i], 60, 'gaus+poly.1', p0 = p0);
        epars .append(pars)
        eupars.append(upars)
            
    _, xmed, xstd, _, _ = histos.hprofile(dz, ene, bins)

    mus  = [p[1] for p in epars]
    sigs = [p[2] for p in epars] 
    kpars, kupars, ffun = cfit.curve_fit(xmed, mus, sigma = sigs, fun = 'poly.1')

    if (plot):
        pltext.canvas(1)
        alpha = 1./ np.sqrt(len(ene))
        plt.scatter(dz, ene, alpha = alpha);
        label = pltext.label_parameters(kpars, kupars, ('a', 'b'), formate = '6.5f')
        #plt.errorbar(xmed, mus, ffun, sigs, label = label)
        pltext.hfun(xmed, mus, ffun, sigs, label = label, mode = 'plot')
        plt.legend();
        
    return kpars, kupars, mus, sigs