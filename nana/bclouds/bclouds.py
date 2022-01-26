#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:50:22 2022

@author: hernando
"""

import numpy             as np
import pandas            as pd
#import tables            as tb


#import hipy.utils        as ut
#import hipy.pltext       as pltext
#import hipy.hfit         as hfit

import clouds.clouds    as clouds
#import clouds.graphs    as graphs
#import clouds.pltclouds as pltclouds

#import clouds.mc_clouds as mcclouds

import invisible_cities.io.dst_io as dio



datadir   = "/Users/hernando/work/investigacion/NEXT/data/MC/NEW/ds/"
filenames = ("Tl208_NEW_v1_03_01_nexus_v5_03_04_cut24.beersheba_label_4mm_fid_v0.h5", 
             "Tl208_NEW_v1_03_01_nexus_v5_03_04_cut25.beersheba_label_4mm_fid_v0.h5")
filenames = ("prueba_cut24_isaura.h5",) 

filenames = [datadir+file for file in filenames]


seg_blob = 3

def get_dfs(filename):
    
    dfs = {}
    
    dfs['rcvoxels'] = dio.load_dst(filename, 'DATASET', 'BeershebaVoxels')
    dfs['mcvoxels'] = dio.load_dst(filename, 'DATASET', 'MCVoxels')
    dfs['mchits']   = dio.load_dst(filename, 'DATASET', 'MCHits')
    dfs['events']   = dio.load_dst(filename, 'DATASET', 'EventsInfo')
    dfs['bins']     = dio.load_dst(filename, 'DATASET', 'BinsInfo')
    dfs['isaura']   = dio.load_dst(filename, 'DATASET', 'IsauraInfo')
    
    
    return dfs

def get_voxel_size(dfs):
    
     voxel_size = [float(dfs['bins'][var].unique()) 
                   for var in ('size_x', 'size_y', 'size_z')]
     return voxel_size
    

def get_event(dfs, evt):
    
    rcvoxels = dfs['rcvoxels']
    mchits   = dfs['mchits']
    isaura   = dfs['isaura']
    
    evoxels    = rcvoxels[rcvoxels.dataset_id == evt]
    ehits      = mchits  [mchits  .dataset_id == evt]
    etracks    = isaura  [isaura  .dataset_id == evt]
    eblobs     = get_eblobs(etracks)

    return evoxels, ehits, etracks, eblobs


def get_cloud(evoxels, voxel_size, eblobs):
    
    coors  = [evoxels[var].values for var in ('xbin', 'ybin', 'zbin')]
    coors  = [size/2 + size * coor for size, coor in zip(voxel_size, coors)]
    ene    = evoxels['energy'].values
    
    # creat cloud
    bins, mask, cells, cloud = clouds.clouds(coors, voxel_size, ene)

    # extend with segclass of the voxel    
    cloud['segclass'] = evoxels['segclass'].values
    
    # extend with energy of the paolina blob
    xposblobs, eneblobs = (eblobs['x'], eblobs['y'], eblobs['z']), eblobs['energy']
    bcell = clouds.cells_value(bins, mask, xposblobs, eneblobs)
    cloud['pinablob'] = bcell
    
    return bins, mask, cells, cloud


def get_eblobs(etracks):
    isablobs = {}
    sel = etracks.trackID == 0
    names = (('x', 'blobi_x'), ('y', 'blobi_y'), ('z', 'blobi_z'), ('energy', 'eblobi'))
    for key, varname in names:
        varnames = [varname.replace('i', i) for i in ['1', '2']]
        var = [float(etracks[varname][sel].values) for varname in varnames]
        isablobs[key] = np.array(var, float)
    eblobs = pd.DataFrame(isablobs)
    #print(eblobs)
    return eblobs


def ana_nodes(cloud):
        
    sel           = cloud['segclass'] == seg_blob
    blob_nodes    = cloud['enode'][sel].unique()

    # create a DF with the nodes information
    nodes         = cloud['enode'].unique()
    nodes_seg     = np.array([int(cloud['segclass'][node]) for node in nodes])
    nodes_isblob  = np.isin(nodes, blob_nodes)
    nodes_size    = np.array([np.sum(cloud['enode'] == node) for node in nodes])
    nodes_energy  = np.array([np.sum(cloud[cloud['enode'] == node].energy) for node in nodes])
    nodes_enecell = np.array([float(cloud['energy'][node]) for node in nodes])
    nodes_nlinks = [np.sum(cloud[cloud.enode == node]['eispass'] == True) for node in nodes]

    def blob_order(vals, nodes):
        vals, pos = clouds.ut_sort(vals, nodes)
        ipos = [int(np.where(pos == node)[0]) for node in nodes]
        return ipos 
    
    nodes_osize    = blob_order(nodes_size   , nodes)
    nodes_oenergy  = blob_order(nodes_energy , nodes)
    nodes_oenecell = blob_order(nodes_enecell, nodes)

    nnodes  = len(nodes)
    dfnodes = pd.DataFrame()
    dfnodes['blobs']      = np.ones(nnodes, int) * len(blob_nodes)
    dfnodes['nodes']     = np.arange(nnodes)
    dfnodes['segclass']  = nodes_seg
    dfnodes['isblob']    = nodes_isblob
    dfnodes['size']      = nodes_size
    dfnodes['energy']    = nodes_energy 
    dfnodes['enecell']   = nodes_enecell
    dfnodes['osize']     = nodes_osize
    dfnodes['oenergy']   = nodes_oenergy 
    dfnodes['oenecell']  = nodes_oenecell
    dfnodes['nlinks']    = nodes_nlinks

    return dfnodes