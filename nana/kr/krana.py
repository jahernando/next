#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:22:49 2022

@author: hernando
"""


import numpy  as np
import pandas as pd

from   collections import namedtuple
from   scipy       import stats
from   scipy       import optimize


import matplotlib.pyplot as plt
import hipy.pltext       as pltext
import hipy.profile      as prof

# parameters
size             = 100
e0, tau0, length = 41.5, 10, 1.
wi               = 41.5/2000.

def generate_kr_toy(size = 100000, 
                    length = length,
                    e0 = e0,
                    tau = tau0,
                    beta = 0.2,
                    sigma = 0.05,
                    x0 = 0.,
                    y0 = 0.):
    """
    Generate Kr data: x, y, z, enegy
    """
    
    ts = stats.uniform.rvs(0, length, size = size)
    xs = stats.uniform.rvs(0, length, size = size) - 0.5 * length
    ys = stats.uniform.rvs(0, length, size = size) - 0.5 * length
    es = (1. - tau * ts/length) * stats.norm.rvs(loc = e0, scale = e0 * sigma, size = size)
    rs = np.sqrt((xs - x0)** 2 + (ys - y0)** 2)
    er = es * (1 - beta * (2 * rs / length) ** 2)
    
    sel = rs < length/2
    df = {'dtime' : ts[sel], 'x': xs[sel], 'y': ys[sel], 'energy': er[sel]}
    return pd.DataFrame(df)
                               

#   Testing generation of LT and Fitting
#-------------------------------------------

def generate_toy(size, e0 = e0, tau0 = tau0, length = length, 
                 sigma_ref = 0.04):
    """ simple LT """
    
    ts = length * stats.uniform.rvs(size = size)
    es  = e0 * ((1 - ts/tau0) + sigma_ref * stats.norm.rvs(size = size))
    return ts, es

@np.vectorize
def attachment(t, lifetime, n_ie):
    return np.count_nonzero(-lifetime * np.log(np.random.uniform(size=int(n_ie))) > t)


def generate_toy_att(size = size, e0 = e0, tau0 = tau0, length = length, wi = wi):
    """ Binomial LT """
    nie0     = e0/wi
    tis      = length * stats.uniform.rvs(size = size)
    nies     = stats.poisson.rvs(nie0, size = size)
    nis      = attachment(tis, tau0, nies)
    eis      = e0 * nis/nie0
    return tis, eis


def generate_toy_bin(size = size, e0 = e0, tau0 = tau0, length = length, wi = wi):
    """ Binomial LT """
    nie0     = e0/wi
    tis      = length * stats.uniform.rvs(size = size)
    nies     = stats.poisson.rvs(nie0, size = size)
    ps       = 1 - tis/tau0
    nks      = stats.binom.rvs(nies, ps) * wi 
    return tis, nks


st_ipar = lambda t, a, b : a - b * t
st_cpar = lambda t, a, b : a * (1 - t / b) 
st_imed = lambda t, a, b : a - b * (t - length/2)


def experiments(generate, fun, e0 = e0, tau0 = tau0, mexps = 1000, size = 100):
    """ generate experiments fit and return result of the fit """
    rs    = []
    for i in range(mexps):
        ts, es = generate(size = size, e0 = e0, tau0 = tau0)
        r      = optimize.curve_fit(fun, ts, es)
        #chi2   = chisq(fun, ts, es, r[0])
        rs.append((r[0], r[1])) 
    return rs

#  Kr maps from fitting
#-----------------------------------

def residuals_(ts, es, par, cov):
    
    xv    = np.ones(shape = (2, len(ts)))
    xv[1] = -ts
    res   = np.dot(par, xv) - es

    var   = np.sum(xv * np.matmul(cov, xv), axis = 0)
    sig   = np.sqrt(var)
    
    sigma = np.sqrt(np.sum(res * res)/ (len(ts) - 2))
    return res, sig, sigma 


KrMap = namedtuple('KrMap', 
                   ('counts', 'eref', 'dedt', 'dtref', 'ueref', 'udedt', 'cov',
                    'chi2', 'pvalue',
                    'bin_centers', 'bin_edges', 'bin_indices', 'residuals'))

def krmap(coors, dtime, energy, bins = (36, 36), counts_min = 40, dt0 = None):
    
    
    counts, ebins, ibins = stats.binned_statistic_dd(coors, energy, 
                                                     bins = bins, statistic = 'count',
                                                     expand_binnumbers = True)    
    ibins = [b-1 for b in ibins]
    cbins = [0.5 * (x[1:] + x[:-1]) for x in ebins]

    ref     = 1000
    indices =  ref * ibins[1] + ibins[0]
    #indices0 = np.array([ref * i1 + i0 for i0 in range(bins[0]) for i1 in range(bins[1])], int)

    eref  = np.zeros(shape = counts.shape)
    dedt  = np.zeros(shape = counts.shape)
    dtref = np.zeros(shape = counts.shape)
    ueref = np.zeros(shape = counts.shape)
    udedt = np.zeros(shape = counts.shape)
    cov   = np.zeros(shape = counts.shape)
    chi2  = np.zeros(shape = counts.shape)
    pval  = np.zeros(shape = counts.shape)
    
    mask = counts > counts_min
      
    residuals = -99999 * np.ones(len(energy))
    for i0, i1 in np.argwhere(mask == True):
        ijsel = indices == int(ref * i1 + i0)
        ts, enes = dtime[ijsel], energy[ijsel]
        #print(len(ts), len(enes), counts[i0, i1], np.sum(ijsel))
        tij = np.median(ts) if dt0 is None else dt0
        st_fun = lambda ts, a, b : a - b * (ts - tij)
        par, var = optimize.curve_fit(st_fun, ts, enes)
        eref [i0, i1] = par[0]
        dedt [i0, i1] = par[1]
        dtref[i0, i1] = tij
        ueref[i0, i1] = np.sqrt(var[0, 0])
        udedt[i0, i1] = np.sqrt(var[1, 1])
        cov  [i0, i1] = var[0, 1]
        
        res, _ , sig = residuals_(ts - tij, enes, par, var)
        residuals[ijsel] = res/sig
        ijchi2 = np.sum(res * res)/(len(res) - 2)
        ijpval = stats.shapiro(res)[1] if (len(res) > 3) else 0.
        chi2  [i0, i1] = ijchi2
        pval  [i0, i1] = ijpval
        
    return KrMap(counts, eref, dedt, dtref, ueref, udedt, cov, chi2, pval,
                 cbins, ebins, ibins, 
                 residuals)



def krmap_scale(coors, dtime, energy, krmap, scale = 1., mask = None):
    
    bins = krmap.bin_edges
    _, _, ibins = stats.binned_statistic_dd(coors, energy, 
                                            bins = bins, statistic = 'count',
                                            expand_binnumbers = True)   
    ibins = [b-1 for b in ibins]
    ref     = 1000
    indices =  ref * ibins[1] + ibins[0]


    counts = krmap.counts
    eref   = krmap.eref
    dtref  = krmap.dtref
    dedt   = krmap.dedt
    
    mask   = counts > 3 if mask is None else mask
    
    corr_energy = np.zeros(len(energy))
    
    for i0, i1 in np.argwhere(mask == True):
        ijsel = indices == int(ref * i1 + i0)
        ts, enes = dtime[ijsel], energy[ijsel]
        par = eref[i0, i1], dedt[i0, i1]
        tij = dtref[i0, i1]
        st_fun = lambda ts, a, b : a - b * (ts - tij)
        cenes = st_fun(ts, *par)
        
        corr_energy[ijsel] = scale * enes / cenes

    return corr_energy, ibins    
    

#--- Plotting

def plot_data(df, bins):
    """
    Plot Kr Data
    """
    canvas = pltext.canvas(6, 2)
    canvas(1)
    pltext.hist(df.dtime, 100);
    plt.xlabel('drift time (ms)')
    canvas(2)
    pltext.hist(df.x, 100);
    plt.xlabel('x (mm)')
    canvas(3)
    pltext.hist(df.y, 100);
    plt.xlabel('y (mm)')
    canvas(4)
    pltext.hist(df.energy, 100);
    plt.xlabel('energy (keV)')
    canvas(5)
    plt.hist2d(df.dtime, df.energy, bins) 
    plt.xlabel('drift time (ms)'); plt.ylabel('energy (keV)')
    plt.colorbar();
    canvas(6)
    mean, ebins, _  = stats.binned_statistic_dd((df.x, df.y), df.energy, bins = bins , statistic = 'mean')
    cbins = [0.5 * (b[1:] + b[:-1]) for b in ebins]
    mesh = np.meshgrid(*cbins)
    plt.hist2d(mesh[0].ravel(), mesh[1].ravel(), bins = ebins, weights = mean.T.ravel())
    plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('energy (keV)')
    plt.colorbar();
    plt.tight_layout();
    
    
def plot_res(res, a0, b0, alabel = '', blabel = ''):
    """ plotting Result of the fit res is a list of manu experiment and have par, cov """
    
    alphas  = np.array([r[0][0] for r in res])
    betas   = np.array([r[0][1] for r in res])
    ualphas = np.array([np.sqrt(np.diag(r[1]))[0] for r in res])
    ubetas  = np.array([np.sqrt(np.diag(r[1]))[1] for r in res])
    cov     = np.array([r[1][0, 1]                for r in res])
    #chi2    = np.array([r[2] for r in res])

    canvas = pltext.canvas(8, 3)
    canvas(1)
    pltext.hist(alphas, 100);
    plt.xlabel(alabel);
    canvas(2)
    pltext.hist(ualphas, 100);
    plt.xlabel(alabel + ' uncertainty');
    canvas(3)
    pltext.hist((alphas - a0)/ualphas, 100);
    plt.xlabel(alabel + ' pool');

    canvas(4)
    pltext.hist(betas, 100);
    plt.xlabel(blabel);
    canvas(5)
    pltext.hist(ubetas, 100);
    plt.xlabel(blabel + ' uncertainty');
    canvas(6)
    pltext.hist((betas - b0)/ubetas, 100);
    plt.xlabel(blabel + ' pool');

    canvas(7)
    plt.hist2d(alphas, betas, (20, 20));
    plt.xlabel(alabel); plt.ylabel(blabel); plt.colorbar();
    canvas(8)
    pltext.hist(cov, 100);
    plt.xlabel('cov')
    
    plt.tight_layout()
    return

def plot_xyvar(var, bins = None, title = '', mask = None, nbins = 100):
    
    mask   = var != np.nan if mask is None else mask 
    nx, ny = var.shape
    bins   = (np.arange(nx+1), np.arange(ny+1)) if bins == None else bins
    cbins  = [0.5 * (x[1:] + x[:-1]) for x in bins]
    mesh   = np.meshgrid(cbins[0], cbins[1])
    canvas = pltext.canvas(2, 2)
    canvas(1)
    plt.hist2d(mesh[0][mask].ravel(), mesh[1][mask].ravel(), bins = bins,
               weights = var[mask].T.ravel());
    plt.xlabel('x'); plt.ylabel('y'); plt.title(title);
    plt.colorbar();
    canvas(2)
    #xsel = var != np.nan
    pltext.hist(var[mask].ravel(), nbins);
    plt.xlabel(title)
    plt.tight_layout();
    return


def plot_xydt_energy_profiles(xdf, nbins = 100):
    zprof  = prof.profile((xdf.dtime,), xdf.energy, nbins)
    prof.plot_profile(zprof, nbins = nbins, stats = ('mean',), coornames = ('dtime',))
    xprof  = prof.profile((xdf.x,), xdf.energy, nbins)
    prof.plot_profile(xprof, nbins = nbins, stats = ('mean',), coornames = ('x',))
    yprof  = prof.profile((xdf.y,), xdf.energy, nbins)
    prof.plot_profile(yprof, nbins = nbins, stats = ('mean',), coornames = ('y',))
    return
    