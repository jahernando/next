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



def dz_effect_profile_fit(xmed : np.array,
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


def dz_effect(dz    : np.array,
              ene   : np.array,
              nbins : int or np.array = 10, 
              p0s   : tuple = None,
              mbins : int = 60,
              fun   : str = 'poly.1',
              plot  : bool = False):
    """
    
    Compute the profile energy vs dz fitting each slice to a gaus+poly.1 
    
    Fit the means of the guassians vs dz to poly.1

    returns the (x, y, ysigma) del profile and the parameters of the fit to poly.1


    Parameters
    ----------
    dz    : np.array, dz-values
    ene   : np.array, energy values
    nbins : int or np.array, optional. Number of bins of the profile.
            The default is 10.
    p0s   : tuple(np.array(float)), optional. Each parameter must be (N, mu, sigma, a, b)
            The default is None.
    mbins : int, optional. Number of bins of the energy fit per slice.
            The default is 60.
    fun   : str, option. Name of the function to fit the profile.
            Default is 'poly.1' (1 degree polynomial)
    plot  : bool, optional. Plot the different energy fits of the profile
            The default is False.
    fun   

    Returns
    -------
    xmed  : np.array(float), dz-values of the profile
    mus   : np.array(float), mu-values of the gaussians
    sigs  : np.array(float), sigma-values of the gaussian fit,
    pars  : np.array(float), parameters of the function of the profile fit
    upars : np.array(float), uncertainties of the parameters of the profile fit
    
    """
    
    xmed, mus, sigs =  \
        dz_effect_profile(dz, ene, nbins, p0s = p0s, mbins = mbins, plot = plot)
        
    if (plot): 
        pltext.canvas(1)
        plt.hist2d(dz, ene, (nbins, nbins))
    pars, upars = \
        dz_effect_profile_fit(xmed, mus, sigs, fun = fun, plot = plot) 
    
    return xmed, mus, sigs, pars, upars



def ds_eblob2scan(ene    : np.array, 
                  eblob2 : np.array, 
                  nscan  : int = 10,
                  mbins  : int = 60,
                  p0     : np.array = None,
                  plot   : bool = False):
    """
    
    dot a scan in eblob2 energy, fit the energy distribution to gaus+poly.1d
    compute the number of signal and bkg events for each cut

    Parameters
    ----------
    ene    : np.array, energy values
    eblob2 : np.array, blob2 energy values
    nscan  : int, optional, number of eblob2 cuts.
             The default is 10.
    mbins  : int, optional, number of bins in the energy hfit
             The default is 60.
    p0     : np.array, optional, initial parameters of the hfit.
             The default is None.
    plot   : bool, optional, plot the hfit for each value of the scan
             The default is False.

    Returns
    -------
    eblob2_scan : np.array, values of the eblob2 threshold
    nsigs       : np.array, signal events
    nbkgs       : np.array, bkg events

    """
    
    eblob2_scan = np.percentile(eblob2, np.linspace(0., 100., nscan))

    hfit = pltext.hfit if plot else histos.hfit

    if (plot):
        subplot = pltext.canvas(nscan)
    
    nbkgs, nsigs = [], []
    for i, eblob2_thr in enumerate(eblob2_scan[:-1]):
        isel = ut.in_range(eblob2, (eblob2_thr, np.inf))
        if (plot): subplot(i + 1)
        ys, xs, eys, pars, upars, ffun = \
            hfit(ene[isel], mbins, fun = 'gaus+poly.1', p0 = p0)
        n, mu, sigma, a0, a1 = pars
        pol = lambda x: a1 * x + a0
        xcs = ut.centers(xs)
        if (plot):
            plt.plot(xcs, pol(xcs), color = 'red');
        nitots = np.sum(ys)
        nibkgs = np.sum(pol(xcs))
        nbkgs.append(nibkgs)
        nsigs.append(nitots - nibkgs)
    nsigs, nbkgs = np.array(nsigs), np.array(nbkgs)
    
    return eblob2_scan[:-1], nsigs, nbkgs
    

def plt_eblob2scan(eblob2_scan, nsigs, nbkgs):
    
    esigs = 100. * nsigs/nsigs[0]
    ebkgs = 100. * nbkgs/nbkgs[0]

    subplot = pltext.canvas(4)

    subplot(1)
    plt.plot(eblob2_scan, nbkgs, marker = 'o', ls = '--', label = 'nkgs')
    plt.plot(eblob2_scan, nsigs, marker = 'o', ls = '--', label = 'signal')
    plt.xlabel('eblob2 threshold (MeV)'); plt.ylabel('number of events'); 
    plt.grid(); plt.legend();

    subplot(2)
    plt.plot(eblob2_scan, ebkgs, marker = 'o', ls = '--', label = 'bkg eff')
    plt.plot(eblob2_scan, esigs, marker = 'o', ls = '--', label = 'sig eff')
    plt.xlabel('eblob2 threshold (MeV)'); plt.ylabel('efficiency (%)'); 
    plt.grid(); plt.legend();

    subplot(3)
    plt.plot(100. - ebkgs, esigs, marker = 'o', ls = '--');
    plt.xlabel('bkg rejection (%)'); plt.ylabel('signal efficiency (%)')
    vs = np.linspace(0., 100., 10)
    plt.plot(100. - vs, vs)
    plt.grid(); 

    subplot(4)
    fom = 0.01* esigs/np.sqrt(0.01 * ebkgs)
    #plt.plot(eblob2_scan[:-1], fom, marker = 'o', ls = '--', label = 'events sigma/sqrt(bkg)')
    plt.plot(eblob2_scan, fom, marker = 'o', ls = '--', label = 'fom'); plt.grid();
    plt.xlabel('eblob2 threshold (MeV)'); 
    plt.ylabel('fom = $\epsilon_s/\sqrt{\epsilon_b}$'); 
    
    
    return