#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:54:36 2021

@author: hernando
"""

import numpy as np

import matplotlib.pyplot as plt

import hipy.utils  as ut
import hipy.cfit   as cfit
import hipy.histos as histos
import hipy.pltext as pltext


#import numpy as np

def dz_energy_correction(energy, dz, alpha = 2.76e-4):
    """ Apply Josh's energy correction by delta-z effect
    """
    return energy/(1 - alpha * dz)


def efficiencies(sels, names = None, plot = False):
    xeffs = []
    ksel = sels[0]
    for isel in sels:
        ksel = ksel & isel
        xeffs.append(ut.efficiency(ksel))
    effs  = [100. * eff[0] for eff in xeffs]
    ueffs = [100. * eff[1] for eff in xeffs]
    
    if (plot):
        names = np.range(len(sels)) if names is None else names
        plt.errorbar(names, effs, ueffs, ls = '--', marker = 'o');
        plt.grid(); plt.ylabel('efficiency (%)')
        
    return effs, ueffs

#def energy_resolution(energy, p0, nbins, erange = None):
    
def energy_fit(ene, bins, range = None, p0 = None, plot = False):
    
    hfit = pltext.hfit if plot else histos.hfit
    
    kargs = {'residuals' : True, 'formate' : '6.5f'} if plot else {}
    _, _, _, pars, upars, _ =  \
        hfit(ene, bins, range = range, fun = 'gaus+poly.1', p0 = p0, **kargs) 
    
    ene0, sigma, usigma = pars[1], pars[2], upars[2]
    r, ur               = 235. * sigma/ene0, 235. * usigma/ene0
    
    return ene0, sigma, usigma, r, ur
    
    
def dz_effect(dz, ene, nbins = 10, p0s = None, plot = False, mbins = 60):
    
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
        _, _, _, pars, upars, _ = hfit(ys[i], mbins, 'gaus+poly.1', p0 = p0);
        epars .append(pars)
        eupars.append(upars)
            
    _, xmed, xstd, _, _ = histos.hprofile(dz, ene, bins)

    mus  = np.array([p[1] for p in epars])
    sigs = np.array([p[2] for p in epars])
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