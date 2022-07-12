#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:23:20 2022

@author: hernando
"""


#from collections.abc import Iterable

import numpy             as np
#import pandas            as pd
#import tables            as tb


#import hipy.utils        as ut
import hipy.pltext       as pltext
#import hipy.hfit         as hfit

import clouds.clouds    as cl
#import clouds.utils     as clut

import networkx         as nx

#import clouds.utils     as clut
#import clouds.clouds    as cl
#import clouds.graphs    as graphs
import clouds.pltclouds as pltclouds
#import networkx         as nx

#import clouds.mc_clouds as mcclouds


import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors        
#from mpl_toolkits.mplot3d import axes3d

plt.rcParams['image.cmap'] = 'rainbow'

#from utils.plotting_utils import plot_cloud_voxels,\
#    plot_cloud_voxels_and_hits, plot_3d_hits

pltext.style()

#-----
# Draw
#-----

def draw_event(clouds, cells, bins, ehits, draw_pina = False):
    #%matplotlib notebook
    cells_select = cl.cells_select
    plt.figure(figsize = (8, 8));

    # cloud
    pltclouds.draw_cloud(cells, bins, clouds);
    
    # MC hits
    hits = [ehits[name].values for name in ['x', 'y', 'z']]    
    ax = plt.gca()
    ax.scatter(*hits, c = 'white', alpha = 0.5);
    # start of the track
    start = [x[5] for x in hits]
    ax.scatter(*start, c = 'green', alpha = 1.)
    # other hits
    sel = ehits.segclass == 1
    if (np.sum(sel) > 0):
        ohits = [ehits[sel][name].values for name in ['x', 'y', 'z']]
        ax.scatter(*ohits, c = 'grey', alpha = 0.5);
    
    # blob voxels
    bsel = clouds.segclass == 3
    pltclouds.voxels(cells_select(cells, bsel), bins, alpha = 0.5, color = 'green' , label = 'true');
    
    # blob voxels
    if (draw_pina):
        _sel = clouds.pinablob > 0
        ax.scatter(*cells_select(cells, _sel), marker = 'D', c = 'yellow', s = 100)
        #pltclouds.voxels(cells_select(cells, psel), bins, alpha = 0.5, color = 'red' , label = 'true');
    
    # Paulina extremes
    
    
    return

# def draw_graph(graph, glink, enes = None):
#     #%matplotlib inline
#     g = nx.Graph()
#     g.add_nodes_from(graph)
#     g.add_edges_from(glink)
#     enes = enes if enes is not None else np.ones(len(graph))
#     diclabels = {}
#     for i, g in enumerate(graph): diclabels[g] = '{:3.1f}.'.format(1e3 * enes[i])
#     print(diclabels)
#     enes = enes if enes is not None else np.ones(len(graph))
#     nx.draw(g, with_labels = True, node_size = 1e3 * enes, node_color = 'yellow');
#     return

def draw_graph(graph, glink, enes = None):
    #%matplotlib inline
    g = nx.Graph()
    g.add_nodes_from(graph)
    g.add_edges_from(glink)
    enes = enes if enes is not None else np.ones(len(graph))
    diclabels = {}
    for i, k in enumerate(graph): diclabels[k] = '{:3.1f}'.format(1e3 * enes[i])
    print(diclabels)
    nx.draw(g, labels = diclabels, with_labels = True, node_size = 1e3 * enes, node_color = 'yellow');
    return


#----
#  Draw clouds
#-----

cells_select = cl.cells_select

def draw_cloud_mc(clouds, cells, bins, ehits):
    #%matplotlib notebook

    plt.figure(figsize = (8, 8));
    pltclouds.voxels(cells, bins, alpha = 0.05, color = 'grey', label = 'cloud');
    
    osel = clouds.segclass == 1
    tsel = clouds.segclass == 2
    bsel = clouds.segclass == 3
    
    pltclouds.voxels(cells_select(cells, osel), bins, alpha = 0.3, color = 'orange', label = 'other');
    pltclouds.voxels(cells_select(cells, tsel), bins, alpha = 0.1, color = 'blue', label = 'track');
    pltclouds.voxels(cells_select(cells, bsel), bins, alpha = 0.3, color = 'red' , label = 'blob');

    # MC hits
    hits = [ehits[name].values for name in ['x', 'y', 'z']]    
    ax = plt.gca()
    ax.scatter(*hits, c = 'white', alpha = 0.5);
    
    plt.xlabel('x (mm)'); plt.ylabel('y (mm)');plt.title('cloud - label');
    return

def draw_cloud_gradient(clouds, cells, bins, ehits):
    #%matplotlib notebook
    plt.figure(figsize = (8, 8));
        # cloud
    #pltclouds.draw_cloud(cells, bins, clouds);
    pltclouds.voxels(cells, bins, alpha = 0.05, color = 'grey', label = 'cloud');
    pltclouds.grads(cells, bins, clouds);
    #plt.gca().scatter(*hits, c = 'gray', alpha = 1., s = 3);
    
    # MC hits
    hits = [ehits[name].values for name in ['x', 'y', 'z']]    
    ax = plt.gca()
    ax.scatter(*hits, c = 'white', alpha = 0.5);

    plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('cloud gradient');
    return

def draw_cloud_nodes(clouds, cells, bins, ehits):
    #%matplotlib notebook
    plt.figure(figsize = (8, 8));
    pltclouds.voxels(cells, bins, alpha = 0.0, color = 'grey', label = 'node');
    nodes  = clouds.enode.unique()
    enodes = [1e3*np.sum(clouds.energy[clouds.enode == node]) for node in nodes]
    #print(enodes)
    norm   = pltcolors.Normalize(vmin=np.min(enodes), vmax=np.max(enodes))
    cmap   = plt.cm.rainbow
    for i, node in enumerate(nodes):
        _nsel = clouds.enode == node
        pltclouds.voxels(cells_select(cells, _nsel), bins, alpha = 0.15,
                         color = cmap(norm(enodes[i])));
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # only needed for matplotlib < 3.1
    cbar = plt.colorbar(sm)
    cbar.set_label('energy (keV)')
    
    ax = plt.gca()
    _sel = clouds.eisnode == True
    ax.scatter(*cells_select(cells, _sel), marker = 'x', c = 'black')
    #pltclouds.voxels(cells_select(cells, nsel), bins, alpha = 0.2, color = 'blue', marker = 'X');
    
    hits = [ehits[name].values for name in ['x', 'y', 'z']]    
    ax = plt.gca()
    ax.scatter(*hits, c = 'white', alpha = 1., s = 3);
    plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('cloud nodes');

#---
# RoC, test in histo2D and hbar
#---

def roc(var1, var2, nbins, range, **kargs):
    counts1, bins = np.histogram(var1, nbins, range, density = True)
    ccounts1 = np.cumsum(counts1 * np.diff(bins))
    counts2,  _   = np.histogram(var2, nbins, range, density = True)
    ccounts2 = np.cumsum(counts2 * np.diff(bins))
    xs = 0.4 * (bins[1:] + bins[:-1])
    plt.plot(ccounts1, 1.-ccounts2, **kargs) # sig eff (y) vs bkg rej (x)
    #plt.plot(ccounts2, 1.-ccounts1, **kargs) #  bkg-acc (y) vs eff (x)
    return counts1, counts2, xs

def hist2d_text(xs, ys, bins, ranges,
                  factor = 100, text_format = "{:4.1f} %", offset = 0.3,
                  **kargs):
    h, hxs, hys, im = plt.hist2d(xs, ys, bins, ranges, **kargs);
    ax = plt.gca()
    for i in range(len(hys)-1):
        for j in range(len(hxs)-1):
            ax.text(hxs[j] + offset, hys[i] + offset, 
                    text_format.format(factor * h.T[i,j]))
    return

def hbar(var, vals = ['blob', 'other', 'track', 'unknown'], **kargs):
    counts = [np.sum(var == val)/len(var) for val in vals]
    plt.bar(vals, counts, **kargs);
    plt.gca().set_xticklabels(vals)
    return