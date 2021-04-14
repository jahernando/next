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

def dz_energy_correction(energy : float or np.array,
                         dz     : float or np.array,
                         alpha  : float = 2.76e-4,
                         scale  : float  = 1.):
    """ Apply Josh's energy correction by delta-z effect
    """
    return scale * energy/(1 - alpha * dz)


def efficiencies(sels  : tuple,
                 names : tuple = None,
                 plot  : bool = False):
    """
    
    compute the efficiencies applying then in serie
    
    Parameters
    ----------
    sels  : tuple(np.array(bool)), list of selections
    names : tuple(names), list of the selection names.
            The default is None.
    plot  : bool, plot the selection efficiencies
            The default is False.

    Returns
    -------
    effs  : np.array(float), efficiencies, i.e sel0, sel0 & sel1, ...
    ueffs : np.array(float), uncertainties of the efficiencies

    """
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
    
def energy_fit(ene   : np.array, 
               bins  : int,
               range : tuple = None,
               p0    : np.array = None,
               plot  : bool = False):
    """
    
    fit the energy distribution inside the range to gaus+poly.1 using 
    initial parameters p0

    Parameters
    ----------
    ene   : np.array, energy values
    bins  : int or np.array, bins
    range : tuple(float), optional, energy range.
            The default is None.
    p0    : np.array(float), (Ngaus, mu, sigma, a0, b) optional. Initial parameters.
            The default is None.
    plot  : bool, optional. Plot the enerfy fit.
            The default is False.

    Returns
    -------
    ene0   : float, mu- of the gausian
    sigma  : float, sigma of the gausian
    usigma : float, uncertainty of sigma
    r      : float, FWHM in %
    ur     : float, unvertainty of FWHM %

    """

    
    hfit = pltext.hfit if plot else histos.hfit
    
    kargs = {'residuals' : True, 'formate' : '6.4f'} if plot else {}
    _, _, _, pars, upars, _ =  \
        hfit(ene, bins, range = range, fun = 'gaus+poly.1', p0 = p0, **kargs) 
    
    ene0, sigma, usigma = pars[1], pars[2], upars[2]
    r, ur               = 235. * sigma/ene0, 235. * usigma/ene0
    
    return ene0, sigma, usigma, r, ur
    
    
def dz_effect_profile(dz    : np.array,
                      ene   : np.array,
                      nbins : int or np.array = 10, 
                      p0s   : tuple = None,
                      mbins : int = 60,
                      plot  : bool = False):
    """
    
    Compute the profile energy vs dz fitting each slice to a gaus+poly.1 

    Parameters
    ----------
    dz    : np.array, dz-values
    ene   : np.array, energy values
    nbins : int or np.array, optional. Number of bins of the profile.
            The default is 10.
    p0s   : tuple(np.array(float)), optional. Each parameter must be (N, mu, sigma, a, b)
            The default is None.
    mbins : int : TYPE, optional. Number of bins of the energy fit per slice.
            The default is 60.
    plot  : bool, optional. Plot the different energy fits of the profile
            The default is False.

    Returns
    -------
    xmed : np.array(float), dz-values of the profile
    mus  : np.array(float), mu-values of the gaussians
    sigs : np.array(float), sigma-values of the gaussian fit

    """
                          
    
    bins  = np.percentile(dz, np.linspace(0., 100., nbins))
    counts, xedges, ipos = histos._profile(dz, ene, 'count', bins)
    nbins = len(bins)
    ys = [ene[ipos == i] for i in range(1, nbins)]

    epars, eupars = [], []
    #p0 = (10., 0.71, 0.02, 70., -70.)
    subplot = pltext.canvas(nbins, 3) if plot else None
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
    
    return xmed, mus, sigs



def dz_effect(xmed : np.array,
              mus  : np.array,
              sigs : np.array,
              fun  : float = 'poly.1',
              plot : bool = False):
    """

    Fit the energy vs dz profile to a 1D-polynomial.
    
    Parameters
    ----------
    xmed : np.array(float), dz-values
    mus  : np.array(float), energy values
    sigs : np.array(float), uncertainties of the energy values
    fun  : str, optional. Name of the function to fit.
           The Default is 'poly.1' 
    plot : bool, optional. plot the fit
           The default is False.

    Returns
    -------
    kpars  : np.array(float), the fitted parameters
    kupars : np.array(float), the errors of the fitted parameters

    """

    kpars, kupars, ffun = cfit.curve_fit(xmed, mus, sigma = sigs, fun = 'poly.1')
    
    if (plot):
        label = pltext.label_parameters(kpars, kupars, ('a', 'b'), formate = '6.5f')
        #plt.errorbar(xmed, mus, ffun, sigs, label = label)
        pltext.hfun(xmed, mus, ffun, sigs, label = label, mode = 'plot')
        plt.legend();
        
    return kpars, kupars