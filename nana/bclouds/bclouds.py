#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:50:22 2022

@author: hernando
"""

from collections.abc import Iterable

import numpy             as np
import pandas            as pd
#import tables            as tb


#import hipy.utils        as ut
#import hipy.pltext       as pltext
#import hipy.hfit         as hfit

import clouds.clouds    as cl
import clouds.utils     as clut

import networkx         as nx
#import clouds.graphs    as graphs
#import clouds.pltclouds as pltclouds

#import clouds.mc_clouds as mcclouds

import invisible_cities.io.dst_io as dio



datadir   = "/Users/hernando/work/investigacion/NEXT/data/MC/NEW/ds/"
filenames = ("Tl208_NEW_v1_03_01_nexus_v5_03_04_cut24.beersheba_label_4mm_fid_v0.h5", 
             "Tl208_NEW_v1_03_01_nexus_v5_03_04_cut25.beersheba_label_4mm_fid_v0.h5")
filenames = ("prueba_cut24_isaura.h5",) 
filenames = ("Tl208_NEW_v1_03_01_nexus_v5_03_04_cut50.beersheba_label_4mm.h5",
             "Tl208_NEW_v1_03_01_nexus_v5_03_04_cut51.beersheba_label_4mm.h5",
             "Tl208_NEW_v1_03_01_nexus_v5_03_04_cut52.beersheba_label_4mm.h5",
             "Tl208_NEW_v1_03_01_nexus_v5_03_04_cut53.beersheba_label_4mm.h5",
             "Tl208_NEW_v1_03_01_nexus_v5_03_04_cut54.beersheba_label_4mm.h5")
             

filenames = [datadir+file for file in filenames]


seg_blob = 3

#-----------------------------
#    CLOUDS
#-----------------------------


def get_dfs(filename):
    """
    
    get the DataFrames of Beershaba Voxles & Isaura

    Parameters
    ----------
    filename : str, filename

    Returns
    -------
    dfs      : list(DF), list of DataFrames

    """
    
    dfs = {}
    
    dfs['rcvoxels'] = dio.load_dst(filename, 'DATASET', 'BeershebaVoxels')
    dfs['mcvoxels'] = dio.load_dst(filename, 'DATASET', 'MCVoxels')
    dfs['mchits']   = dio.load_dst(filename, 'DATASET', 'MCHits')
    dfs['events']   = dio.load_dst(filename, 'DATASET', 'EventsInfo')
    dfs['bins']     = dio.load_dst(filename, 'DATASET', 'BinsInfo')
    dfs['isaura']   = dio.load_dst(filename, 'DATASET', 'IsauraInfo')    
    
    
    for key in dfs.keys():
        dfs[key].fillna(-999)
    
    return dfs


def get_dimensions(dfs):
    """
    
    returns dimensions: frame origin and vixel size

    Parameters
    ----------
    dfs : DataFrame, bins dataFrame

    Returns
    -------
    x0         : np.array(float), origin of the frame
    voxel_size : np.array(float), voxel size

    """
    
    df = dfs['bins']
    
    voxel_size = [float(df[var].unique()) for var in ('size_x', 'size_y', 'size_z')]
    x0         = [float(df[var].unique()) for var in ('min_x', 'min_y', 'min_z')]
     
    return np.array(x0), np.array(voxel_size)



def get_events(dfs):
    """
    
    return the event numbers

    Parameters
    ----------
    dfs : DataFrame

    Returns
    -------
    events : np.array(int), list of event IDs

    """
    
   
    events = dfs['rcvoxels'].dataset_id.unique()
    
    return events


def get_isaura_eblobs(etracks):
    """
    
    return the isaura-paulin energy blobs

    Parameters
    ----------
    etracks : isaura DF

    Returns
    -------
    eblobs : eblob filtered DataFrame

    """
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


def get_event(dfs, evt):
    """
    
    return the DataFrames of a given event ID

    Parameters
    ----------
    dfs : list(DataFrame)
    evt : int, event ID

    Returns
    -------
    evoxels : DF, Beersheba voxels
    ehits   : DF, MC hits
    etracks : DF, Isaura tracks
    eblobs :  DF, eblob info (TODO)

    """
    
    rcvoxels = dfs['rcvoxels']
    mchits   = dfs['mchits']
    isaura   = dfs['isaura']
    
    evoxels    = rcvoxels[rcvoxels.dataset_id == evt]
    ehits      = mchits  [mchits  .dataset_id == evt]
    etracks    = isaura  [isaura  .dataset_id == evt]
    eblobs     = get_isaura_eblobs(etracks)

    return evoxels, ehits, etracks, eblobs



def get_clouds(evoxels, origin, voxel_size):
    """
    
    Construct Clouds from the reconstructrd voxels 

    Parameters
    ----------
    evoxels    : DataFrame, Reconstructred voxels
    origin     : np.array, origin position of the clouds frame
    voxel_size : np.array, size of the voxels

    Returns
    -------
    bins       : np.array, bins in the different coordinates
    mask       : np.array(bool), indicating the active cells in the frame
    cells      : np.array, acitve cells
    clouds     : DatFrame, Clouds DataFrame

    """
    
    offsets = 0.5, 0.5, 0.5, # 0.5
    coors  = [evoxels[var].values for var in ('xbin', 'ybin', 'zbin')]
    coors  = [x + size * (coor + offset) for x, size, coor, offset 
              in zip(origin, voxel_size, coors, offsets)]
    ene    = evoxels['energy'].values
    
    # creat cloud
    bins, mask, cells, clouds = cl.clouds(coors, voxel_size, ene)


    # extend with segclass of the voxel    
    #cloud['dataclass'] = np.ones(len(cells[0]))
    #cloud['segclass'] = evoxels['segclass'].values
    
    # extend with energy of the paolina blob
    #xposblobs, eneblobs = (eblobs['x'], eblobs['y'], eblobs['z']), eblobs['energy']
    #bcell = clouds.cells_value(bins, mask, xposblobs, eneblobs)
    #cloud['pinablob'] = bcell
    
    return bins, mask, cells, clouds


def clouds_extend_label(clouds, evoxels, mchits, bins, mask):
    """
    
    Extend the clouds with the labeling and MChits:
        'segclass'  : segementation or label of the cell
        'segblob'   : 1, 2 for blob1, blob2
        'mcenergy'  : mcenergy 
        'mcextreme' : 1, 2, -1 for blob1, blob2 start of the electron

    Parameters
    ----------
    clouds  : DataFrame, Clouds
    evoxels : DataFrame, RC voxels, with labeling
    mchits  : DataFrame, MC hits
    bins    : np.array, bins of the clouds
    mask    : np.array, mask of the clouds (active cells)

    Returns
    -------
    clouds  : DataFrame, Extended Clouds 
        DESCRIPTION.

    """
    
    # segment
    clouds['segclass']  = evoxels['segclass'].values
    
    # mc-energy
    
    
    # blobs
    isblob0 = evoxels['elem'] == '3_0' # first blob
    isblob1 = evoxels['elem'] == '3_1' # second blob
    
    segblob  = np.zeros(len(clouds), int)
    segblob[isblob0] = 1
    segblob[isblob1] = 2
    clouds['segblob'] = segblob
    
    # mcenergy
    mcpos, mcene = [mchits[x] for x in ['x', 'y', 'z']], mchits['energy']
    mccells      = cl.cells_value(bins, mask, mcpos, mcene)
    #print(len(mccells), len(clouds))
    clouds['mcenergy'] = mccells
    #print(np.sum(clouds.mcenergy), np.sum(mcene))

    # extreme of the blobs    
    mcextreme = np.zeros(len(clouds), int)
    for i in (1, 2):
        selblob = (clouds.segblob == i) & (clouds.segclass == 3)
        if (np.sum(selblob) > 0):
            emax = np.max(clouds.mcenergy[selblob])
            mcextreme[clouds.mcenergy == emax] = i
            #print(emax, i, np.unique(mcextreme))
   
    
    # extreme of the initial track, only for electrons?
    if (np.sum(mchits.binclass == 0) > 0):
        i, ok = 0, False
        while ( (not ok) & (i < 5)):
            i += 1
            ihit = min(i * 10, len(mchits))
            sel  = (mchits.segclass == 2) & (mchits.hit_id == ihit)
            if (np.sum(sel) == 1):
                first_hit = [(float(mchits[sel][i]),) for i in ('x', 'y', 'z')]
                icell     = cl.cells_value(bins, mask, first_hit, [1.,])
                if (np.sum(icell >0) > 0): ok = True
                mcextreme[icell > 0] = -1
                #print('start electron')
    clouds['mcextreme'] = mcextreme
    
    
    #print(clouds[['segclass', 'segblob', 'mcextreme', 'mcenergy']])
    
    return clouds


def clouds_extend_isaura(clouds, eblobs, bins, mask):
    """
    
    extend clouds DF with the energy and index of the Paulina/Isauara blobs:
        'epinablob'  : isaura-paulina blob energy asociated to the best isaura cell of the blob
        'pinablob'   : 1, 2 in the best cell of isarua-paulina blob.

    Parameters
    ----------
    clouds : DF, clouds
    eblobs : DF, labelled voxels
    bins   : np.array, cloud bins
    mask   : np.array, cloud mask

    Returns
    -------
    clouds : DF, extended cloud with 'epinablob' (, 'pinablob' (0, 1, 2)
        energy and index the blob in the cell asociated to position of paulina/isaura blob)
        DESCRIPTION.

    """
    
    xposblobs, eneblobs = (eblobs['x'], eblobs['y'], eblobs['z']), eblobs['energy']
    bcell = cl.cells_value(bins, mask, xposblobs, eneblobs)
    clouds['epinablob'] = bcell
    clouds['pinablob']  = np.zeros(len(bcell), int)
    clouds['pinablob'][bcell > 0.] = 2
    clouds['pinablob'][bcell == np.max(bcell)] = 1
    
    return clouds


def get_clouds_from_event(dfs, evt):
    
    origin, voxel_size = get_dimensions(dfs)
    evoxels, ehits, etracks, eblobs = get_event(dfs, evt)
    edataclass                      = int(evoxels.binclass.unique())

    bins, mask, cells, clouds   = get_clouds(evoxels, origin, voxel_size)  
    clouds                      = clouds_extend_label(clouds, evoxels, ehits, bins, mask) 
    clouds                      = clouds_extend_isaura(clouds, eblobs, bins, mask)
    clouds['dataclass']         = edataclass
    clouds['evtntracks']        = int(etracks.numb_of_tracks.unique())
    
    return clouds


#-----------------------------
#---- NODES FRAME
#------------------------------

def nodes_frame(clouds):
    """
    
    Construct a DataFrame for the nodes of the Clouds. Indexed by node ID.
        'node': node ID,
        'energy': energy of the node,
        'nsize': number of cells in the node

    Parameters
    ----------
    clouds : DataFrame, Clouds DF

    Returns
    -------
    df    : DataFrame, with nodes info

    """
    
    nodes  = clouds.enode[clouds.eisnode == True]
    
    enes  = [np.sum(clouds.energy[clouds.enode == node]) for node in nodes]
    sizes = [np.sum(clouds.enode == node) for node in nodes]

    df = {}
    df['nnodes'] = len(nodes)
    df['node']   = nodes
    df['energy'] = enes
    df['nsize']  = sizes
    
    nnodes = len(nodes)
    df['evtene']   = np.sum(enes) * np.ones(nnodes, float)
    df['evtnodes'] = nnodes       * np.ones(nnodes, int)
    
    df = pd.DataFrame(df)
    df.set_index('node')
    return df
        
    
def nodes_frame_extend_label(fnodes, clouds):
    """
    
    Extend Nodes Frame DataFrame with labeling and MC information
        'segclass': blob, other, track of unknown depending on the segment of the cells
        'segextremne': blob1, blob2, start, for blob1, blob2 or extreme of the electron
            one per event, associated to the node where the best cell of each tipe lay
    
    Parameters
    ----------
    fnodes : DataFrame, of the nodes, 
    clouds : DataFrame, Clouds 

    Returns
    -------
    fnodes : DataFrame, extended nodes DF 

    """
    
    def _seg(vals):
        uval = np.unique(vals)
        if 3 in uval: return 'blob'
        if 1 in uval: return 'other'
        if 2 in uval: return 'track'
        return 'unkown'
    
    def _extreme(vals):
        uval = np.unique(vals)
        s = ''
        if  1 in uval: s += 'blob1'
        if  2 in uval: s += 'blob2'
        if -1 in uval: s += 'start'
        if (s == ''):  s =  'node'
        #print(uval, s)
        return s
    
    nodes = fnodes.node
    scell = [         clouds.segclass[clouds.enode == node]  for node in nodes]
    sseg  = [_seg    (clouds.segclass[clouds.enode == node])  for node in nodes]
    sext  = [_extreme(clouds.mcextreme[clouds.enode == node]) for node in nodes]
    
    fnodes['segcell']     = scell
    fnodes['segclass']    = sseg
    fnodes['segextreme']  = sext
    
    return fnodes
    
def nodes_frame_extend_isaura(fnodes, clouds):
    """
    
    Extend nodes Frame DataFrame with information of isaura-paulina blobs
        'segpaulina' : blob1, blod2, segment
        'epaulihna'  : energy of the paulina blob

    Parameters
    ----------
    fnodes : DataFrame, Nodes DataFrame,
    clouds : DataFrame, clouds DataFrame

    Returns
    -------
    fnodes : DataFrame, extended nodes DF

    """

    nodes    = fnodes.node.values

    ntracks  = int(np.unique(clouds.evtntracks))
    fnodes['evtntracks'] = ntracks * np.ones(len(nodes), int)

    epaulina = np.full(len(nodes), 0.)
    paulina  = np.full(len(nodes), 'segment')
    fnodes['segpaulina'] = paulina
    fnodes['epaulina']   = epaulina
    for i in (1, 2):
        sel = clouds.pinablob == i
        if (np.sum(sel) == 1):
            node = int(clouds.enode[sel])
            fnodes.segpaulina[node] = 'blob'+str(i)
            fnodes.epaulina[node] = float(clouds.epinablob[sel])
    #print(fnodes.segpaulina)
    return fnodes
        

def nodes_frame_extend_graph(fnodes, graphs, glinks):
    """
    
    Extend the nodes Frame DF with information about the graphs:
        'nlinks': number of links of the node
        'links': list((int, int)) list of the links (pairs of node-id, node-id)
        'graphid': id of the graph which this node belongs to (graphs ordered by energy)
        'eccentricity': eccentricity of the node
        'eccdistance' : distance to the eccentricity
        'eneorder'    : order position on energy in the graph
        'ecc0distance' : eccentricity to the most energetic among the most eccentric nodes
        
        'graphid'   : id of the graph which this node belongs to (graphs ordered by energy)
        'graphsize' : number of nodes in the graph
        'graphecc'  : eccentricity of the graph
        'graphnecc' : number of eccentric nodes in the graph
        'graphcycles': number of cycles in the graph

    Parameters
    ----------
    fnodes : DataFrame
        DESCRIPTION.
    graphs : list(list), list of nodes in graphs
    glinks : list (list (int, int)), list of links for each graph

    Returns
    -------
    fnodes : DataFrame, Extended Nodes DF

    """
    
    nodes = fnodes.node.values
    #print(fnodes.node.values)
    egraphs = [np.sum(fnodes.energy[graph]) for graph in graphs]
    _, xzip = clut.ut_sort(egraphs, zip(graphs, glinks))
    graphs, glinks = [x[0] for x in xzip], [x[1] for x in xzip]
    
    igraph  = [i for node in nodes for i, graph in enumerate(graphs) if node in graph]
    #print('igraph ', igraph)
     
    #print('graphs ', graphs)
    #print('glinks ', glinks)
    try:
        nlinks     =  []
        for glink in glinks: nlinks += list(glink)
    except:
        print('glinks ', glinks)
        print('nlinks ', nlinks)
    #print('nlinks ', nlinks)
     
    nodes_links = [tuple([link for link in nlinks if node in link]) for node in nodes]
    #print([len(x) for x in nodes_links])

    fnodes['links']  = nodes_links
    fnodes['nlinks'] = [len(link) for link in nodes_links]
 
    decc   = {}
    ddis   = {}
    dior   = {}
    decc0  = {}
    dsize  = {}
    dene   = {}
    dloops = {}
    dnecc  = {}
    ddiam  = {}
    for graph, glink in zip(graphs, glinks):
        g = nx.Graph()
        g.add_nodes_from(graph)
        g.add_edges_from(glink)
    
        shortest_path = dict(nx.all_pairs_shortest_path_length(g))
    
        iecc = nx.eccentricity(g)
        decc.update(iecc)
        #print(iecc)
        diameter = nx.diameter(g)
        eccs     = [node for node in graph if iecc[node] == diameter]
        necc     = len(eccs)
        
        _, eccs_ineneorder = clut.ut_sort(fnodes.energy[eccs].values, eccs)
        ecc0               = eccs_ineneorder[0]
        #print('eccs > ', ecc0, eccs, fnodes.energy[eccs].values)
        
        egraph = np.sum(fnodes.energy[graph])
        loops  = nx.cycle_basis(g)
        for node in graph:
            ddis[node]   = diameter - iecc[node]
            dsize[node]  = len(graph)
            dene[node]   = egraph
            dloops[node] = len(loops)
            ddiam[node]  = diameter
            dnecc[node]  = necc
            decc0[node]  = shortest_path[node][ecc0]
            
        
        _, ograph = clut.ut_sort(fnodes.energy[graph].values, graph)
        for node in graph: 
            dior[node] = [i for i, onode in enumerate(ograph) if onode == node][0]
        #print('graph ', graph)
        #print('graphene', egraph)
        #print('graphsize', len(graph))
        #print('graphecc' , diameter)
        #print('graphnecc', necc)
        #print('graphcycles ', len(loops))              
        #print('ecce    ', decc)
        #print('eccdis  ', ddis)
        #print('enedis  ', dior)
        #print('ecc0dis ', decc0)
        
    
    fnodes['eccentricity'] = [decc[node]  for node in nodes]
    fnodes['eccdistance']  = [ddis[node]  for node in nodes]
    fnodes['ecc0distance'] = [decc0[node] for node in nodes]
    fnodes['eneorder']     = [dior[node]  for node in nodes]
    fnodes['graphid']      = igraph
    fnodes['graphene']     = [dene[node]  for node in nodes]
    fnodes['graphsize']    = [dsize[node] for node in nodes]
    fnodes['graphecc']     = [ddiam[node] for node in nodes]
    fnodes['graphnecc']    = [dnecc[node] for node in nodes]
    fnodes['graphcycles']  = [dloops[node] for node in nodes]
    
    return fnodes


#---------------------
#--- Graphs
#---------------------


def get_graphs(clouds, type = 'eisborder'):
    """
    
    compute the number of clouds, assign a cloud-id to the clouds.

    Parameters
    ----------
    clouds : DF, clouds dataFrame
    type   : str, name of the edge: eispass, eisborder

    Returns
    -------

    nodes   : np.array(int), list of nodes IDs
    nlinks  : list( (int, int)), list of pairs of connected nodes
    graphs  : list( list(int)), list of nodes in separated graphs
    glinks  : list( list (int, int)) list of list of paris between nodes in separated graphs
    idgraph : np.array(int), size the number of cells, each cell has the id od the graph
                            it correspond to the node-id with the largest energy

    """

    
    nodes = clouds.kid[clouds.eisnode == True].values
    enes  = [np.sum(clouds.energy[clouds.enode == node].values) for node in nodes]

    enes, nodes = clut.ut_sort(enes, nodes)
    #print('energies ', enes)
    #print('nodes ', nodes)

    cborder    = clouds.kid[clouds[type] == True].values
    #print('passes ', cpass)
    nlinks = []
    for kid in cborder:
        #weight = np.max(clouds.energy[kid], clouds.energy[elink[kid]])
        pair   = (clouds.enode[kid], clouds.enode[clouds.elink[kid]])
        if (pair in nlinks)              : continue
        if ((pair[1], pair[0]) in nlinks): continue
        nlinks.append(pair)


    graphs, glinks = _get_graphs(nodes, nlinks)
    
    idgraph = np.zeros(len(clouds), int)
    for graph in graphs:
        node = graph[0]
        for inode in graph:
            idgraph[clouds.enode == inode] = int(node)
    

    return nodes, nlinks, graphs, glinks, idgraph



def _get_graphs(nodes, nlinks):
    
    def _graph_new_nodes(node, graph):
        newnodes = []
        for link in nlinks:
            if (node in link):
                node1, node2 = link
                if (node1 not in graph): newnodes.append(node1)
                if (node2 not in graph): newnodes.append(node2)
        return newnodes


    def _graph_list(node, graph):
        newnodes = _graph_new_nodes(node, graph)
        if (len(newnodes) > 0):
            for unode in newnodes:
                if unode not in graph: graph.append(unode)
            #print('graph ', graph)
        for unode in newnodes: _graph_list(unode, graph)
    
    graphs = []
    _nodes  = list(nodes)
    while (len(_nodes)> 0):
        graph = [_nodes[0]]
        _graph_list(_nodes[0], graph)
        for n in graph: _nodes.remove(n)
        graphs.append(graph)

    graph_links = []
    for graph in graphs:
        glink = []
        for node in graph:
            for link in nlinks:
                if (node in link):
                    ok = (link not in glink) and ((link[1], link[0]) not in glink)
                    #print(link, glink, ok)
                    if (ok): glink.append(link)
        graph_links.append(glink)


    return graphs, graph_links


#-------------------------
#   CLOUDS TO NODES
#-------------------------


def ana_nodes_frame(clouds):
    
    fnodes = nodes_frame(clouds)
    fnodes = nodes_frame_extend_label(fnodes, clouds)
    fnodes = nodes_frame_extend_isaura(fnodes, clouds)
    _, _, graphs, glinks, idgraph = get_graphs(clouds, 'eisborder')
    fnodes = nodes_frame_extend_graph(fnodes, graphs, glinks)
    
    return fnodes
    
#-- Run    


def clouds_run(filename, ana, nevt = -1, axis = 0):
    
    print('Filename : ', filename)
    dfs        = get_dfs(filename)

    evts       = get_events(dfs)
    print('Events  :', len(evts))
    
    df = None
    nevt = len(evts) if nevt == -1 else nevt
    mevt      = 0 
    nfailures = 0
    
    for evt in evts[: nevt]:
        mevt += 1
        
        if (evt % 500 == 0): print('event ', evt)
        
        # clouds = get_clouds_from_event(dfs, evt)
        
        # idf              = ana(clouds)
        # idf['dataclass'] = np.unique(clouds.dataclass)[0]
        # idf['event']     = evt
        
        # df = idf if df is None else \
        # pd.concat((df, idf), ignore_index = True, axis = axis)

        
        try:
            clouds = get_clouds_from_event(dfs, evt)
        except:
            nfailures += 1
            continue
        
        idf              = ana(clouds)
        idf['dataclass'] = np.unique(clouds.dataclass)[0]
        idf['event']     = evt
        
        df = idf if df is None else \
            pd.concat((df, idf), ignore_index = True, axis = axis)
         
    print('processed number of events ', mevt)
    print('clouds failures            ', nfailures)

    df = df if axis != 1 else df.transpose()
    return df
        
        
#------------------
#   NODES to BLOBS
#------------------


def nodes_to_graph(fnodes, graphid = 0):
    sel    = fnodes.graphid == graphid
    graph  = fnodes.node  [sel].values
    _links = [list(link) for link in fnodes.links[sel].values]
    glinks = []
    for link in _links: glinks += link
    return graph, glinks
    

def graph_extremes(graph, glinks, depth = 0):
    
    g = nx.Graph()
    g.add_nodes_from(graph)
    g.add_edges_from(glinks)
 
    shortest_path = dict(nx.all_pairs_shortest_path_length(g))
    eccentricity = nx.eccentricity(g)
    diameter     = nx.diameter(g)
    
    ecc_nodes = [node for node in graph if eccentricity[node] >= diameter - depth]
    
    extremes = []
    for i, n0 in enumerate(ecc_nodes):
        for n1 in ecc_nodes[i+1 :]:
            if (shortest_path[n0][n1] >= diameter - depth):
                extremes.append((n0, n1))

    def _accept(nu, nv, list_extremes):
        if ((nu[0] in nv) & (nu[1] in nv)): return False
        return (nu, nv) not in list_extremes

    ext_extremes = []
    for n0, n1 in extremes:
        n0s = [(n0, ni) for ni in graph if shortest_path[n0][ni] == 1]
        n1s = [(n1, ni) for ni in graph if shortest_path[n1][ni] == 1]
        ext_extremes += [(nu, nv) for nu in n0s for nv in n1s 
                         if _accept(nu, nv, ext_extremes)]

    return diameter, ecc_nodes, extremes, ext_extremes 


def _nodes_energy(nodes, fnodes):
    _nodes = [nodes,] if isinstance(nodes, Iterable) == False else nodes
    return np.sum([float(fnodes.energy[fnodes.node == node]) for node in _nodes])
    
def _nodes_size(nodes, fnodes):
    _nodes = [nodes,] if isinstance(nodes, Iterable) == False else nodes
    return np.sum([int(fnodes.nsize[fnodes.node == node]) for node in _nodes])
    
def _nodes_ismcextr(nodes, fnodes, names = ('blob1', 'blob2', 'start')):
    _nodes = [nodes,] if isinstance(nodes, Iterable) == False else nodes
    isin   = np.sum([str(fnodes.segextreme[fnodes.node == node].values[0]) in names for node in _nodes])
    return isin   

def _nodes_segclass(nodes, fnodes):
    _nodes   = [nodes,] if isinstance(nodes, Iterable) == False else nodes
    segments = [str(fnodes.segclass[fnodes.node == node].values[0]) for node in _nodes]
    segment  = 'track' if 'track' in segments else 'unknown'
    segment  = 'other' if 'other' in segments else segment
    segment  = 'blob'  if 'blob'  in segments else segment
    return segment   


def _order_by_energy(items, fnodes):
    enes        = [_nodes_energy(item, fnodes) for item in items]
    oenes, oitems = clut.ut_sort(enes, items)
    return oitems, oenes


def ana_extremes(fnodes):
    
    nodes                                  = fnodes.node.values
    graph, glinks                          = nodes_to_graph(fnodes)
    #print('graph  ', graph)
    #print('glinks ', glinks)
    ecc, ecc_nodes, extremes, ext_extremes = graph_extremes(graph, glinks)
    #print('ecc          ', ecc)
    #print('ecc nodes    ', ecc_nodes)
    #print('extremes     ', extremes)
    #print('ext_extremes ', ext_extremes)
    
    mc_nodes = [node for node in graph if _nodes_ismcextr(node, fnodes) == 1]
    #print('mc nodes     ', mc_nodes)
    

    def _mc():
        blobs = [mc_nodes,] if len(mc_nodes) == 2 else []
        return blobs

    def _paulina():
        names     = ('blob1', 'blob2')
        paunodes  = [node for node in nodes # graph 
                   if str(fnodes.segpaulina[fnodes.node == node].values[0]) in names]
        #print('pau nodes' , paunodes)
        blobs     = [paunodes,] if len(paunodes) == 2 else []
        return blobs

    def _isapaulina():
        return _paulina()

    def _extremes():
        if (len(extremes) <= 1): return extremes
        oextremes, _ = _order_by_energy(extremes, fnodes)
        return oextremes


    def _extremes_sum():
        if (len(ext_extremes) <= 1): return ext_extremes
        enes = [_nodes_energy(p0, fnodes) + _nodes_energy(p1, fnodes) 
                for p0, p1 in ext_extremes]
        _, oext_extremes = clut.ut_sort(enes, ext_extremes)
        return oext_extremes

    def _extremes_max():
        oext_extremes = _extremes_sum()
        oexts = []
        for extr in oext_extremes:
            extr0, _ = _order_by_energy(extr[0], fnodes)
            extr1, _ = _order_by_energy(extr[1], fnodes)
            oexts.append((extr0[0], extr1[0]))
        return oexts

    
    def _iorder(candis, method):
        if (method == 'extremes_sum'):
            for i, item in enumerate(candis):
                #print('iorder ', i, item[0], item[1])
                if (_nodes_ismcextr(list(item[0]) + list(item[1]), fnodes) == 2):
                    return i
            return -1
        for i, item in enumerate(candis):
            if (_nodes_ismcextr(item, fnodes) == 2):
                return i
        return -1
        
            
    dfg = {'evtenergy'   : np.sum(fnodes.energy.values),
           'evtsize'     : len(fnodes),
           'evtntracks'  : int(np.unique(fnodes.evtntracks)),
           'evtngraphs'  : np.max(fnodes.graphid.values) + 1,
           'evtismcextr' : _nodes_ismcextr(nodes, fnodes),
           'g0size'      : len(graph),
           'g0energy'    : _nodes_energy(graph, fnodes),
           'g0ecc'       : ecc,
           'g0necc'      : len(ecc_nodes),
           'g0ismcextr'  : _nodes_ismcextr(graph, fnodes)
           }
    #print('graph info ', dfg)

    df0 = {'bmethod'   : 'unknown',
           'bncandis'  : 0,
           'bmcorder'  : -1,
           'bismcextr' : 0,
           'b1node'    : -1,
           'b1energy'  : 0.,
           'b1size'    : 0,
           'b1ismcextr': 0,
           'b1segclass': 'unknown',
           'b2node'    : -1,
           'b2energy'  : 0.,
           'b2size'    : 0,
           'b2ismcextr': 0,
           'b2segclass': 'unknown'}

    methods = {'mc'          : _mc,
               'paulina'     : _paulina,
               'isapaulina'  : _isapaulina,
               'extremes'    : _extremes,
               'extremes_sum': _extremes_sum,
               'extremes_max': _extremes_max}

    def _data_method(method):
        candis = methods[method]()
        #print('method ', method)
        #print('candis ', candis)
        idf = dict(df0)
        if (len(candis) <= 0):
            idf['bmethod'] = method
            return idf
        
        idf['bmethod']  = method
        idf['bncandis'] = len(candis)
        idf['bmcorder'] = _iorder(candis, method)
        
        
        blobs = candis[0]
        if (method != 'extremes_sum'):
            idf['bismcextr'] = _nodes_ismcextr(blobs, fnodes)
        else:
            idf['bismcextr'] = _nodes_ismcextr(list(blobs[0]) + list(blobs[1]), fnodes)


        blobs, _ = _order_by_energy(blobs, fnodes)

        for i in (0, 1):
            b = blobs[i]
            idf['b'+str(i+1)+'node']     = b
            idf['b'+str(i+1)+'energy']   = _nodes_energy  (b, fnodes)
            idf['b'+str(i+1)+'size']     = _nodes_size    (b, fnodes)
            idf['b'+str(i+1)+'ismcextr'] = _nodes_ismcextr(b, fnodes)
            idf['b'+str(i+1)+'segclass'] = _nodes_segclass(b, fnodes)
            
            if (method == 'isapaulina'):
                idf['b'+str(i+1)+'energy'] = float(fnodes.epaulina[fnodes.node == b])
        
        return idf
        

    df = {}
    for key in dfg.keys(): df[key] = []
    for key in df0.keys(): df[key] = []
    
    
    for method in methods.keys():
        try:
            idf = _data_method(method)
        except:
            print(method, extremes, ext_extremes)
            print(df)
        #print('data method ', idf)
        for key in dfg.keys(): df[key].append(dfg[key])
        for key in df0.keys(): df[key].append(idf[key])

        
    df = pd.DataFrame(df)
    return df
       

def nodes_run(dfnodes, ana, nevt = -1, axis = 1):

    group     = ['fileindex', 'event']
    dfgroups  = dfnodes.groupby(group)
    indices   = list(dfgroups.groups.keys())
    print('Number of events ', len(indices))
    
    df   = None
    mevt = 0
    nfailures = 0
    for indice in indices[ : nevt]:
        
        nevt = dfgroups.get_group(indice)
        if (mevt % 500 == 0): print('event ', mevt)
        mevt += 1
        idf              = ana(nevt)
        idf['dataclass'] = np.unique(nevt.dataclass)[0]
        for g in group:
            idf[g]       = np.unique(nevt[g])[0]
        df = idf if df is None else \
            pd.concat((df, idf), ignore_index = True)
  
        #try :
        #    idf              = ana(nevt)
        #    idf['dataclass'] = np.unique(nevt.dataclass)[0]
        #    for g in group:
        #        idf[g]       = np.unique(nevt[g])[0]
        #    df = idf if df is None else \
        #        pd.concat((df, idf), ignore_index = True)
        #except:
        #    nfailures += 1
        #    continue
    
    #df = df if axis != 1 else df.transpose()        
    print('processed number of events ', mevt)
    print('failures                   ', nfailures)
    return df 

# def nodes_mcblobs(fnodes):
    
#     sel = fnodes.segextreme == 'blob1'
#     b1   = int(fnodes.node[sel]) if np.sum(sel) == 1 else -1
#     seg2 = 'blob2'
#     idataclass = int(np.max(fnodes.dataclass))
#     if (idataclass == 0): 
#         seg2 = 'start'
#     sel = fnodes.segextreme == seg2
#     b2  = int(fnodes.node[sel]) if np.sum(sel) == 1 else -1
#     mcblobs = (b1, b2)
    
#     #print('mcblobs ', mcblobs)
#     return mcblobs



# def nodes_blobs_candidates(fnodes, depth = 0, graphid = 0):
#     sel      = fnodes.graphid == 0
#     nodes    = fnodes.node[sel].values
#     #enes     = fnodes.energy[sel].values
#     links    = fnodes.links[sel].values
#     eccnodes = fnodes.node[sel & (fnodes.eccdistance <= depth)].values

#     nlinks = []
#     for link in links: nlinks += link

#     g = nx.Graph()
#     g.add_nodes_from(nodes)
#     g.add_edges_from(nlinks)
#     shortest_path = dict(nx.all_pairs_shortest_path_length(g))
#     ecc = nx.diameter(g)

#     extremes = []
#     for i, n0 in enumerate(eccnodes):
#         for n1 in eccnodes[i+1 :]:
#             if (shortest_path[n0][n1] >= ecc - depth):
#                 extremes.append((n0, n1))
                
#     def _ene(n):
#         ene = float(fnodes[fnodes.node == n].energy.values)
#         return ene
            
#     extremes_enes = [_ene(n0) + _ene(n1) for n0, n1 in extremes]
#     extremes_enes, extremes = clut.ut_sort(extremes_enes, extremes)

#     extremes = [tuple(p) for p in extremes]

#     #print('nodes ', nodes)
#     #print('links ', nlinks)
#     #print('ecc noides ', eccnodes)
#     #print('candidates ', extremes)
#     #print('candidates enes', extremes_enes)
#     return extremes

# def nodes_blobs_pair_candidates(fnodes,  graphid = 0):
    
#     depth    = 0
#     sel      = fnodes.graphid == 0
#     nodes    = fnodes.node[sel].values
#     #enes     = fnodes.energy[sel].values
#     links    = fnodes.links[sel].values
#     eccnodes = fnodes.node[sel & (fnodes.eccdistance <= depth)].values
    
#     nlinks = []
#     for link in links: nlinks += link
    
#     g = nx.Graph()
#     g.add_nodes_from(nodes)
#     g.add_edges_from(nlinks)
#     shortest_path = dict(nx.all_pairs_shortest_path_length(g))
#     ecc = nx.diameter(g)
    
#     #nx.draw(g, with_labels = True)
    
#     extremes = []
#     for i, n0 in enumerate(eccnodes):
#         for n1 in eccnodes[i+1 :]:
#             if (shortest_path[n0][n1] >= ecc - depth):
#                 extremes.append((n0, n1))
    
                
#     ext_extremes = []
#     for n0, n1 in extremes:
#         n0s = [(n0, ni) for ni in nodes if shortest_path[n0][ni] == 1]
#         n1s = [(n1, ni) for ni in nodes if shortest_path[n1][ni] == 1]
#         ext_extremes += [(nu, nv) for nu in n0s for nv in n1s if (nu, nv)] # not in ext_extremes]
                
#     def _ene(pair):
#         n0, n1 = pair
#         ene = float(fnodes[fnodes.node == n0].energy.values + fnodes[fnodes.node == n1].energy.values) 
#         return ene
        
#     ext_extremes_enes = [_ene(n0) + _ene(n1) for n0, n1 in ext_extremes] 
#     ext_extremes_enes, ext_extremes = clut.ut_sort(ext_extremes_enes, ext_extremes)

#     ext_extremes = [(tuple(p1), tuple(p2)) for p1, p2 in ext_extremes]

#     #print('nodes ', nodes)
#     #print('enes  ', enes)
#     #print('links ', nlinks)
#     #print('ecc noides ', eccnodes)
#     #print('candidates ', extremes)
#     #print('ext extremes ', ext_extremes)
#     #print('energies     ', ext_extremes_enes)
    
#     return ext_extremes


# def _find_mcblob_node(clouds, iblob):
#     """
    
#     return the node-id and node-energy for iblob = 1, 2

#     Parameters
#     ----------
#     clouds : DF, clouds
#     iblob : int, (1, 2) for the blob 1, 2

#     Returns
#     -------
#     node : int, node-id where is the MC blob 
#     enode : ene, energy of the node where is the MC blob

#     """

#     nodes = np.unique(clouds.enode[clouds.segblob == iblob])
#     if (len(nodes) <= 0): return -1, -1
#     enes  = [np.sum(clouds.energy[clouds.enode == node]) for node in nodes] 
#     enes, nodes = clut.ut_sort(enes, nodes)
#     node, enode = nodes[0], enes[0]
#     return node, enode 

# def get_mcblob_nodes(clouds):
    
#     # TODO
#     vals = list(np.unique(clouds.segblob))
#     vals = (1, -1) if -1 in vals else (1, 2)
#     #print(vals)
#     blobs = [_find_mcblob_node(clouds, i) for i in vals]
#     nodes, enes = [b[0] for b in blobs], [b[1] for b in blobs]
#     if (2 in vals):
#         enes, nodes = clut.ut_sort(enes, nodes)
        
#     return nodes, enes   
    

# def isaura_blobs_energy(dfs, etype = 0 ):
    
#     di = dfs['isaura']
#     dv = dfs['rcvoxels']

#     evts  = dv.dataset_id[dv.binclass == etype].values

#     eblob1  = [float(d.eblob1[d.trackID == 0]) for evt, d in di.groupby('dataset_id') 
#                  if bool(np.isin(evt, evts))]
#     eblob2 = [float(d.eblob2[d.trackID == 0]) for evt, d in di.groupby('dataset_id') 
#                  if bool(np.isin(evt, evts))]

#     return eblob1, eblob2


# def graph_connections(graph, glinks):
#     """
    
#     return the connection of the nodes in a graph

#     Parameters
#     ----------
#     graph : list(int), list of nodes IDs
#     glinks : list ( (int, int)) list of pairs fo connected nodes

#     Returns
#     -------
#     connections : list(int), liwht with the number of connections of each node

#     """
#     connections = [int(np.sum([node in link for link in glinks])) for node in graph]
#     return connections 


# def graph_loops(graph, glink):
#     """
    
#     return the number of loops in a graph

#     Parameters
#     ----------
#     graph : list (int), list of node IDs
#     glink : list ( (int, int)) list of pairs of connected nodes

#     Returns
#     -------
#     cycloes: list(list(int)), list of the list of nodes in the lopp

#     """
#     g = nx.Graph()
#     g.add_nodes_from(graph)
#     g.add_edges_from(glink)
#     cycles = nx.cycle_basis(g)
#     #print(cycles)
#     return cycles


# def graph_extremes(graph, glink, enes):
    
#     if (len(graph) <= 1): return (graph[0], -1), graph, 0
    
#     g = nx.Graph()
#     g.add_nodes_from(graph)
#     g.add_edges_from(glink)

#     ecc   = nx.eccentricity(g)
#     d     = nx.diameter(g)
#     extr  = [key for key in ecc.keys() if ecc[key] == d]
#     nextr = len(extr)

#     shortest_paths = dict(nx.all_pairs_shortest_path_length(g))
#     paths = [(extr[i], extr[j]) for i in range(nextr) for j in range(i + 1, nextr)
#               if shortest_paths[extr[i]][extr[j]] == d]
    
#     def _node_energy(i):
#         return [ene for ene, i1 in zip(enes, graph) if i1 == i][0]
    
#     pair = paths[0]
#     if (len(paths) > 1):
#         enes = [_node_energy(i0) + _node_energy(i1) for i0, i1 in paths]
#         enes, paths = clut.ut_sort(enes, paths, reverse = True)
#         pair = paths[0]

#     #print('eccentricity ', ecc)
#     #print('candidates   ', extr)
#     #print('extremes     ', extremes)
        
#     return pair, extr, d


# def graph_path(graph, glink):
    
#     g = nx.Graph()
#     g.add_nodes_from(graph)
#     g.add_edges_from(glink)

#     gpath           = dict(nx.all_pairs_shortest_path_length(g))
    
#     return gpath
    
# #def graph_path_nodes_at_distance(gpath, graph, origin, d):
# #    nodes_at_d = [node for node in graph if gpath[node][origin] == d]
# #    return nodes_at_d

# def graph_features(graph, glink, gpath):
    
#     g = nx.Graph()
#     g.add_nodes_from(graph)
#     g.add_edges_from(glink)

#     nnodes       = g.number_of_nodes() 
#     nedges       = g.number_of_edges()
#     diameter     = nx.diameter(g)

#     eccentricity = nx.eccentricity(g)
#     eccentric    = [node for node in graph if eccentricity[node] == diameter]
    
#     connections = [int(np.sum([node in link for link in glink])) for node in graph]
#     extremes    = [node for node, con in zip(graph, connections) if con == 1]

#     cycles      = nx.cycle_basis(g)

#     df = {}
#     df['nnodes']       = nnodes
#     df['nedges']       = nedges
#     df['diameter']     = diameter
#     df['extremes']     = extremes
#     df['eccentric']    = eccentric
#     df['cycles']       = cycles

#     return df

# def draw_graph(graph, glink, enes = None):
#     g = nx.Graph()
#     g.add_nodes_from(graph)
#     g.add_edges_from(glink)
#     enes = enes if enes is not None else np.ones(len(graph))
#     nx.draw(g, with_labels = True, node_size = 1e3 * enes, node_color = 'yellow');
#     return


# def graph_eccentric_nodes(graph, glink, origin, ecc = -1):
    
#     g = nx.Graph()
#     g.add_nodes_from(graph)
#     g.add_edges_from(glink)

#     shortest_path   = dict(nx.all_pairs_shortest_path_length(g))
#     dist_to_origin  = [shortest_path[i][origin] for i in graph]
#     ecc_to_origin   = np.max(dist_to_origin) if ecc == -1 else ecc
    
#     nodes_ecc_to_origin = [node for node in graph 
#                             if shortest_path[node][origin] == ecc_to_origin]

#     #print(graph)
#     #print(dist_to_origin)
#     #print(ecc_to_origin)        
#     #print(nodes_ecc_to_origin)
#     return nodes_ecc_to_origin


    
#    enes_ecc_to_origin, nodes_ecc_to_origin = \
#        clut.ut_sort(enes_ecc_to_origin, nodes_ecc_to_origin)
#        
#    return [origin, nodes_ecc_to_origin[0]]
    

# def graph_shortest_distance(graph, glink, origin, target):
    
#     g = nx.Graph()
#     g.add_nodes_from(graph)
#     g.add_edges_from(glink)

#     shortest_paths = dict(nx.all_pairs_shortest_path_length(g))
#     origins = [i for i in origin if i in graph]
#     shortest_dist = []
#     for j in target:
#         d = np.min([shortest_paths[i][j] for i in origins]) if j in graph else -1
#         shortest_dist.append(d)
    
#     return shortest_dist


# def graph_most_energetic(graphs, glinks, clouds):
    
#     ene_graphs = [np.sum([node_energy(id, clouds) for id in graph]) for graph in graphs]
    
#     ene_graphs, xgraphs = clut.ut_sort(ene_graphs, zip(graphs, glinks), reverse = True)

#     egraph        = ene_graphs[0]
#     graph, glink  = xgraphs[0] 

#     return egraph, graph, glink
    

# def graph_mclabel(graph, clouds):
    
#     def _seg(vals):
#         uval = np.unique(vals)
#         if 3 in uval: return 'blob'
#         if 2 in uval: return 'track'
#         if 1 in uval: return 'other'
#         return 'out'
    
#     def _extreme(vals):
#         uval = np.unique(vals)
#         if  1 in uval: return 'blob1'
#         if  2 in uval: return 'blob2'
#         if -1 in uval: return 'start'
#         return 'segment'
    
    
#     #segclass  = [_seg(clouds.segclass[clouds.enode == node])    for node in graph]
#     segblob   = [_extreme(clouds.segblob[clouds.enode == node])  for node in graph]
    
#     start, _     = get_mcblob_nodes(clouds)
#     for i, node in enumerate(graph):
#         if node in start: segblob[i]+='*'

#     df = {}
#     df['blob'] = [node for node, label in zip(graph, segblob) if 'blob' in label]
#     df['star'] = [node for node, label in zip(graph, segblob) if '*' in label]
    
#     return df, segblob
                       


    
#--- Utils

# def node_energy_from_clouds(id, clouds):
#         return np.sum(clouds.energy[clouds.enode == id])
    

# def node_size_from_clouds(id, clouds):
#         return np.sum(clouds.enode == id)


# def _ordered_nodes_from_clouds(nodes, clouds):
#     enes = [np.sum(clouds.energy[clouds.enode == node]) for node in nodes]
#     enes, ordered_nodes = clut.ut_sort(enes, nodes)
#     return ordered_nodes, enes

# nodes_ordered_by_energy = _ordered_nodes

#---- Analysis


# def ana_nodes_frame(clouds):
    
#     fnodes = nodes_frame(clouds)
#     fnodes = nodes_frame_extend_label(fnodes, clouds)
#     fnodes = nodes_frame_extend_isaura(fnodes, clouds)
#     _, _, graphs, glinks, idgraph = get_graphs(clouds, 'eisborder')
#     fnodes = nodes_frame_extend_graph(fnodes, graphs, glinks)
    
#     return fnodes
    
# #-- Run    


# def run(filename, ana, nevt = -1, axis = 0):
    
#     print('Filename : ', filename)
#     dfs        = get_dfs(filename)

#     evts       = get_events(dfs)
#     print('Events  :', len(evts))
    
#     df = None
#     nevt = len(evts) if nevt == -1 else nevt
#     mevt      = 0 
#     nfailures = 0
    
#     for evt in evts[: nevt]:
#         mevt += 1
        
#         if (evt % 100 == 0): print('event ', evt)
        
#         # clouds = get_clouds_event(dfs, evt)
        
#         # idf              = ana(clouds)
#         # idf['dataclass'] = np.unique(clouds.dataclass)[0]
#         # idf['event']     = evt
        
#         # df = idf if df is None else \
#         # pd.concat((df, idf), ignore_index = True, axis = axis)

        
#         try:
#             clouds = get_clouds_event(dfs, evt)
#         except:
#             nfailures += 1
#             continue
        
#         idf              = ana(clouds)
#         idf['dataclass'] = np.unique(clouds.dataclass)[0]
#         idf['event']     = evt
        
#         df = idf if df is None else \
#             pd.concat((df, idf), ignore_index = True, axis = axis)
         
#     print('processed number of events ', mevt)
#     print('clouds failures            ', nfailures)

#     df = df if axis != 1 else df.transpose()
#     return df
        
        


# def ana_graph_features(clouds):

#     nodes, nlinks, graphs , glinks, idgraph = get_graphs(clouds, 'eisborder')
#     egraph, graph, glink  = graph_most_energetic(graphs, glinks, clouds)
#     gpath                 = graph_path(graph, glink) 
#     df                    = graph_features(graph, glink, gpath)
#     #print(df)
#     genes                 = np.array([node_energy(node, clouds) for node in graph])
    
#     genes_ene_order, graph_ene_order = clut.ut_sort(genes, graph)
    
#     clouds['idgraph'] = idgraph

#     dfmc, _  = graph_mclabel(graph, clouds)
#     #print(dfmc)
    
#     blobs, eblobs = nodes_ordered_by_energy(dfmc['star'], clouds)
#     blob_sizes    = [np.sum(clouds.enode == blob) for blob in blobs]
    
#     ecc_nodes = df['eccentric']
#     graph_distance = [np.min([gpath[blob][origin] for origin in ecc_nodes])  
#                              for blob in blobs]
#     ene_distance   = [[i for i, node in enumerate(graph_ene_order) if node == blob][0]
#                       for blob in blobs]

#     df['ngraphs']     = len(graphs)
#     df['ene']         = np.sum(clouds.energy)
#     df['enegraph']    = egraph
#     for key in df.keys():
#         if type(df[key]) == list: df[key] = len(df[key])
    
#     df['eblob1']      =  eblobs[0] if len(blobs) >= 1 else -1
#     df['eblob2']      =  eblobs[1] if len(blobs) >= 2 else -1
#     df['sizeblob1']   =  blob_sizes[0] if len(blobs) >= 1 else -1
#     df['sizeblob2']   =  blob_sizes[1] if len(blobs) >= 2 else -1

#     df['disgraph1']   =  graph_distance[0] if len(blobs) >= 1 else -1
#     df['disgraph2']   =  graph_distance[1] if len(blobs) >= 2 else -1
#     df['disene1']     =  ene_distance[0] if len(blobs) >= 1 else -1
#     df['disene2']     =  ene_distance[1] if len(blobs) >= 2 else -1

#     for key in df.keys():
#         df[key] = np.array(df[key], type(df[key]))

#     df = pd.Series(df)
#     #print(df)

#     return df


# def ana_graph_blobs(clouds):

#     nodes, nlinks, graphs , glinks, idgraph = get_graphs(clouds, 'eisborder')
#     egraph, graph, glink  = graph_most_energetic(graphs, glinks, clouds)
#     gpath                 = graph_path(graph, glink) 
#     df                    = graph_features(graph, glink, gpath)
#     #diameter              = df['diameter']
    
    
#     #print(df)
#     genes                 = np.array([node_energy(node, clouds) for node in graph])
    
#     gextremes, _, _       = graph_extremes(graph, glink, genes)

    
#     genes_ene_order, graph_ene_order = clut.ut_sort(genes, graph)
    
#     clouds['idgraph'] = idgraph

#     dfmc, _ = graph_mclabel(graph, clouds)
#     #print(dfmc)
    
#     mc_blobs, _ = nodes_ordered_by_energy(dfmc['star'], clouds)    
#     ecc_nodes   = df['eccentric']
    
#     def _blob_data(blobs):
#         blobs, eblobs  = nodes_ordered_by_energy(blobs, clouds)
#         blob_sizes     = [np.sum(clouds.enode == blob) for blob in blobs]       
#         graph_distance = [np.min([gpath[blob][origin] for origin in ecc_nodes])  
#                           for blob in blobs]
#         ene_distance   = [[i for i, node in enumerate(graph_ene_order) if node == blob][0]
#                           for blob in blobs]
#         success        = [blob in mc_blobs for blob in blobs]
#         return blobs, eblobs, blob_sizes, graph_distance, ene_distance, success


#     def _mc():
#         return mc_blobs

#     def _paulina():
#         nodes = [int(clouds.enode[clouds.pinablob == i]) for i in (1, 2)]
#         nodes = [node for node in nodes if node in graph]
#         return nodes

#     def _maxene():
#         return graph_ene_order[:2]

#     def _graph():
#         return gextremes
#         # paths = [(node0, node1) for node0 in ecc_nodes 
#         #          for node1 in ecc_nodes if gpath[node0][node1] == diameter]
#         # paths_enes = [node_energy(node0, clouds) + node_energy(node1, clouds) 
#         #                for node0, node1 in paths]
        
#         # _, paths = clut.ut_sort(paths_enes, paths)
#         # path     = paths[0]
#         # path_ene = [node_energy(node, clouds) for node in path]
#         # path_ene, path = clut.ut_sort(path_ene, path)
#         # return path

#     ddf = {}
#     names = list(df.keys()) + ['method', 'ngraphs', 'ene', 'enegraph', 
#                                'node1', 'node2',
#                                'eblob1', 'eblob2', 'sizeblob1', 'sizeblob2',
#                                'disgraph1', 'disgraph2', 
#                                'disene1', 'disene2',
#                                'success1', 'success2']
#     for name in names: ddf[name] = []
    
#     methods = {}
#     methods['mc']       = _mc 
#     methods['paulina']  = _paulina
#     methods['maxene']   = _maxene
#     methods['graph']    = _graph

#     for method in methods.keys():
#         ddf['ngraphs'].append(len(graphs))
#         ddf['ene']    .append(np.sum(clouds.energy))
#         ddf['enegraph'].append(egraph)
#         for key in df.keys():
#             if type(df[key]) == list: 
#                 ddf[key].append(len(df[key]))     
#             else:
#                 ddf[key].append(df[key])
        
#         ddf['method'].append(method)
#         blobs = methods[method]()
#         #print(method, blobs)
#         blobs, eblobs, sblobs, gdis, edis, success = _blob_data(blobs)
#         #print('blobs  ', blobs)
#         #print('eblobs ', eblobs)
#         #print('sblobs ', sblobs)
#         #print('success ', success)
            
#         ddf['node1']   .append(blobs[0] if len(blobs) >= 1 else -1)
#         ddf['node2']   .append(blobs[1] if len(blobs) >= 2 else -1)
#         ddf['eblob1']   .append(eblobs[0] if len(blobs) >= 1 else -1)
#         ddf['eblob2']   .append(eblobs[1] if len(blobs) >= 2 else -1)
#         ddf['sizeblob1'].append(sblobs[0] if len(blobs) >= 1 else -1)
#         ddf['sizeblob2'].append(sblobs[1] if len(blobs) >= 2 else -1)
#         ddf['disgraph1'].append(gdis[0] if len(blobs) >= 1 else -1)
#         ddf['disgraph2'].append(gdis[1] if len(blobs) >= 2 else -1)
#         ddf['disene1']  .append(edis[0] if len(blobs) >= 1 else -1)
#         ddf['disene2']  .append(edis[1] if len(blobs) >= 2 else -1)
#         ddf['success1'] .append(success[0] if len(blobs) >= 1 else False)
#         ddf['success2'] .append(success[1] if len(blobs) >= 2 else False)

#     #print(ddf)
#     for key in ddf.keys():
#         ddf[key] = np.array(ddf[key], type(ddf[key]))
        
#     ddf = pd.DataFrame(ddf)
#     #print(df)

#     return ddf
    

# #--- Previous

# def ana_graph_categories(clouds):
    
    
#    # _, _, xclouds, _, idcloud              = get_graphs(clouds, 'eisborder')
#     #nodes, nlinks, xgraphs, glinks, idgraph = get_graphs(clouds, 'eispass')
#     nodes, nlinks, graphs , glinks, idgraph = get_graphs(clouds, 'eisborder')

#     #clouds['idcloud'] = idcloud    
#     clouds['idgraph'] = idgraph
    
    
#     egraph, graph, glink = graph_most_energetic(graphs, glinks, clouds)    
#     enes                 = np.array([node_energy(node, clouds) for node in graph])
#     connections          = graph_connections(graph, glink)
#     loops                = graph_loops      (graph, glink) 
#     pair, eccnodes, dis  = graph_extremes(graph, glink, enes)
           
#     enes_in_order, nodes_in_order = clut.ut_sort(enes, graph, reverse = True)              
                     
#     df = {}
#     #df['nclouds']   = np.array(len(graphs), int)
#     df['ngraphs']   = np.array(len(graphs), int)
#     df['energy']    = np.array(np.sum(clouds.energy), float)
#     df['nnodes']    = np.array(len(graph), int)
#     df['next']      = np.array(np.sum(np.array(connections, int) == 1), int)
#     df['nloops']    = np.array(len(loops), int)
#     df['necc']      = np.array(len(eccnodes), int)
#     df['ecc']       = np.array(dis, int)

#     #df['ecloud0']   = np.array(np.sum(clouds.energy[idcloud == graph), float)
#     df['egraph0']   = np.array(np.sum(clouds.energy[idgraph == graph[0]]), float)                                                    
    
#     blob_enes = [node_energy(node, clouds) for node in pair]
#     blob_enes, pair = clut.ut_sort(blob_enes, pair, reverse = True)

#     eblob1 = node_energy(pair[0], clouds)
#     eblob2 = node_energy(pair[1], clouds)
#     df['eblob1']    = np.array(eblob1, float)
#     df['eblob2']    = np.array(eblob2, float)
    
#     sblob1 = node_size(pair[0], clouds)
#     sblob2 = node_size(pair[1], clouds)
#     df['sblob1']   = np.array(sblob1, int)
#     df['sblob2']   = np.array(sblob2, int)

#     nodes_mc, _ = get_mcblob_nodes(clouds)
#     #print(nodes_mc)

#     df['success1'] = np.array(pair[0] in nodes_mc, bool)
#     df['success2'] = np.array(pair[1] in nodes_mc, bool)

#     dist = graph_shortest_distance(graph, glink, pair, nodes_mc)

#     df['disgraph1'] = np.array(dist[0], int)
#     df['disgraph2'] = np.array(dist[1], int)

#     def _disene(id):
#         for i, jd in enumerate(nodes_in_order):
#             if (id == jd): return i
#         return -1

#     #print(nodes_in_order)
#     #print(enes_in_order)
#     df['disene1'] = np.array(_disene(pair[0]), int)
#     df['disene2'] = np.array(_disene(pair[1]), int)

#     return pd.Series(df)



# #---- Ana mcblob node



# def ana_mcblob_nodes(clouds):
    
#     #nodes, nlinks, graphs , glinks, idgraph = get_graphs(clouds)
#     nodes, nlinks, graphs, glinks, idclouds = get_graphs(clouds, 'eisborder')


#     # TODO order graphs by energy    
#     graph = graphs[0]
#     glink = glinks[0]
#     connections = graph_connections(graph, glink)
#     loops       = graph_loops      (graph, glink) 

#     enes        = [np.sum(clouds.energy[clouds.enode == n]) for n in graph]
#     enes, ordered_nodes = clut.ut_sort(enes, graph)
#     #print(ordered_nodes)        
    
#     names = 'blob', 'node', 'enode', 'sizenode', 'ingraph', 'inloop', 'connections', 'eneorder'
#     df = {}
#     for name in names: df[name] = []
    
#     mc_nodes, mc_ene = get_mcblob_nodes(clouds)
    
#     for i in (1, 2):
#         node, enode = mc_nodes[i-1], mc_ene[i-1]
#         if (node <= -1): continue
#         snode = int(np.sum([clouds.enode == node]))
#         ingraph = node in graph
#         inloop, con, iorder = False, -1, -1
#         if (ingraph):
#             inloop  = np.sum([np.isin(node, loop) for loop in loops]) > 0
#             con     = int([con for con, n in zip(connections, graph) if n == node][0])
#             iorder  = int([i for i, n in enumerate(ordered_nodes) if n == node][0])
#         df['blob']  .append(i)
#         df['node']  .append(node)
#         df['enode'] .append(enode)
#         df['sizenode'].append(snode)
#         df['connections'].append(con)
#         df['eneorder'].append(iorder)
#         df['ingraph'].append(ingraph)
#         df['inloop'] .append(inloop)
    
#     df =  pd.DataFrame(df)
#     return df
            
# # -- Ana mcblob success        

# def ana_mcblob_success(clouds):
    
#     nodes, nlinks, graphs , glinks, idgraph = get_graphs(clouds, 'eisborder')    
#     egraph, graph, glink = graph_most_energetic(graphs, glinks, clouds)    
#     enes                 = np.array([node_energy(node, clouds) for node in graph])
#     pair, extremes, dis  = graph_extremes(graph, glink, enes)
    
#     #nodes_all            = np.unique([clouds.enode[clouds.eisnode == True]])
#     #nodes_in_eneorder, _ = _ordered_nodes(nodes_all, clouds)
    
#     onodes, enodes  = nodes_ordered_by_energy(graph, clouds)
    
#     nodes_mc, _  = get_mcblob_nodes(clouds)
    
#     def _mc():
#         return nodes_mc
    
#     def _paulina():
#         nodes = [int(clouds.enode[clouds.pinablob == i]) for i in (1, 2)]
#         return nodes
    
#     def _maxene():
#         return onodes[:2]
        
#     def _graph():
#         return pair

#     def _graph_maxene():
#         origin    = onodes[0]
#         #xnodes    = graph_eccentric_nodes(graph, glink, origin)
#         #xnodes, cc = nodes_ordered_by_energy(xnodes, clouds)
#         #print('mexene node   ', origin)
#         #print('ecc nodes     ', xnodes)
#         #print('ecc nodes ene ', cc)
#         #xpair  = [origin, xnodes[0]]
#         target  = pair[1]
#         xnodes  = graph_eccentric_nodes(graph, glink, target, 1)
#         xnodes, xenes = nodes_ordered_by_energy(xnodes, clouds)
#         etarget = node_energy(target, clouds)
#         xnode   = xnodes[0] if etarget < xenes[0] else target 
#         xpair   = [origin, xnode] if origin != xnode else [origin, target]
#         #print('pair   ', pair)
#         #print('xnodes ', xnodes, xenes)
#         #print('target ', target, etarget)
#         #print('xpair  ', xpair)
#         return xpair

#     def _enenode(id):
#         return np.sum(clouds.energy[clouds.enode == id])
    
#     def _sizenode(id):
#         return np.sum(clouds.enode == id)

    
#     def _enedist(id):    
#         for i, jd in enumerate(onodes):
#             if (id == jd): return i
#         return -1
  
#     def _success(id):
#         return np.isin(id, nodes_mc)
          
     
#     method = {}
#     method['mc']      = _mc
#     method['paulina'] = _paulina
#     method['maxene']  = _maxene 
#     method['graph']   = _graph
#     #method['graph_maxene'] = _graph_maxene
    
#     df = {}
#     names = ['method', 'node1', 'node2', 'ene1', 'ene2', 'size1', 'size2',
#              'success1', 'success2', 'disgraph1', 'disgraph2', 'disene1', 'disene2']
#     for name in names: df[name] = []
    
#     for key in method.keys():
#         nodes   = method[key]()
#         #print('nodes ', key, ' = ' , nodes)
#         enes    = [_enenode(node)  for node in nodes]
#         sizes   = [_sizenode(node) for node in nodes]
#         success = [_success(node)  for node in nodes]
#         dist    = graph_shortest_distance(graph, glink, nodes, nodes_mc)
#         edist   = [_enedist(node)   for node in nodes]
#         df['method'].append(key)
#         for i in (0, 1):
#             df['node'     + str(i + 1)].append(nodes[i])
#             df['ene'      + str(i + 1)].append(enes[i])
#             df['size'     + str(i + 1)].append(sizes[i])
#             df['success'  + str(i + 1)].append(success[i])
#             df['disgraph' + str(i + 1)].append(dist[i])
#             df['disene'   + str(i + 1)].append(edist[i])

#     return pd.DataFrame(df)    
# # --- Run



#----- Obsolete

        


# def _get_clouds(evoxels, voxel_size, x0, eblobs):
    
#     coors  = [evoxels[var].values for var in ('xbin', 'ybin', 'zbin')]
#     coors  = [x + size * (coor + 0.5) for x, size, coor in zip(x0, voxel_size, coors)]
#     #coors  = [size * coor for size, coor in zip(voxel_size, coors)]
#     ene    = evoxels['energy'].values
    
#     # creat cloud
#     bins, mask, cells, cloud = clouds.clouds(coors, voxel_size, ene)


#     # extend with segclass of the voxel    
#     cloud['dataclass'] = np.ones(len(cells[0]))
#     cloud['segclass'] = evoxels['segclass'].values
    
#     # extend with energy of the paolina blob
#     xposblobs, eneblobs = (eblobs['x'], eblobs['y'], eblobs['z']), eblobs['energy']
#     bcell = clouds.cells_value(bins, mask, xposblobs, eneblobs)
#     cloud['pinablob'] = bcell
    
#     return bins, mask, cells, cloud




# def ana_nodes(cloud):
            
#     sel           = cloud['segclass'] == seg_blob
#     blob_nodes    = cloud['enode'][sel].unique()

#     # create a DF with the nodes information
#     nodes         = cloud['enode'].unique()
#     nodes_seg     = np.array([int(cloud['segclass'][node]) for node in nodes])
#     nodes_isblob  = np.isin(nodes, blob_nodes)
#     nodes_size    = np.array([np.sum(cloud['enode'] == node) for node in nodes])
#     nodes_energy  = np.array([np.sum(cloud[cloud['enode'] == node].energy) for node in nodes])
#     nodes_enecell = np.array([float(cloud['energy'][node]) for node in nodes])
#     nodes_nlinks = [np.sum(cloud[cloud.enode == node]['eispass'] == True) for node in nodes]


#     def blob_order(vals, nodes):
#         vals, pos = clouds.ut_sort(vals, nodes)
#         ipos = [int(np.where(pos == node)[0]) for node in nodes]
#         return ipos 
    
#     nodes_osize    = blob_order(nodes_size   , nodes)
#     nodes_oenergy  = blob_order(nodes_energy , nodes)
#     nodes_oenecell = blob_order(nodes_enecell, nodes)

#     nnodes  = len(nodes)
#     dfnodes = pd.DataFrame()
#     dfnodes['blobs']     = np.ones(nnodes, int) * len(blob_nodes)
#     dfnodes['nodes']     = np.arange(nnodes)
#     dfnodes['segclass']  = nodes_seg
#     dfnodes['isblob']    = nodes_isblob
#     dfnodes['size']      = nodes_size
#     dfnodes['energy']    = nodes_energy 
#     dfnodes['enecell']   = nodes_enecell
#     dfnodes['osize']     = nodes_osize
#     dfnodes['oenergy']   = nodes_oenergy 
#     dfnodes['oenecell']  = nodes_oenecell
#     dfnodes['nlinks']    = nodes_nlinks
    

    
#     pinablobs_nodes  = cloud.enode[cloud.pinablob > 0].unique()
#     pb_nodes         = np.isin(nodes, pinablobs_nodes)
#     dfnodes['ispinablob']  = pb_nodes 
    
#     def _ana_pinablobs():
#         _icells = np.where(cloud['pinablob'] > 0)[0]
#         _ecells = cloud['pinablob'][_icells].values
#         _nnodes = cloud['enode'][_icells].values
#         _      , _icells = clouds.ut_sort(_ecells, _icells)
#         _ecells, _nnodes = clouds.ut_sort(_ecells, _nnodes)
#         _opinablobs = np.zeros(nnodes, int)
#         _epinablobs = np.zeros(nnodes, float)
#         _segpinablobs = np.zeros(nnodes, int)
#         for i, _n in enumerate(_nnodes):
#             _opinablobs  [nodes == _n] = i + 1
#             _epinablobs  [nodes == _n] = _ecells[i]
#             _segpinablobs[nodes == _n] = cloud['segclass'][_icells[i]]
#         return _opinablobs, _epinablobs, _segpinablobs
    
#     opina, epina, spina = _ana_pinablobs()
#     dfnodes['opinablob']   = opina
#     dfnodes['enepinablob'] = epina
#     dfnodes['segpinablob'] = spina
    
    
    
#     #_b = np.where(cloud['pinablob'] > 0)[0]
#     #_e = cloud['pinablob'][_b].values
#     #_n = cloud['enode'][_b].values
#     #_e, _pos     = clouds.ut_sort(_e, _n)
#     #pb_opinablob = np.zeros(nnodes, 0)
#     #pb_opinablob = 
    
#     dfnodes['ispinablob']  = pb_nodes 
    

#     return dfnodes


# def run_anablobs(filename, nevt = -1):
    
#     print('Filename : ', filename)
#     dfs        = get_dfs(filename)

#     evts       = get_events(dfs)
#     print('Events  :', len(evts))
#     voxel_size = get_voxel_size(dfs)
    
#     df = None
#     nevt = len(evts) if nevt == -1 else nevt
    
#     for evt in evts[: nevt]:
    
        
#         if (evt%100 == 0): print('event ', evt)
        
#         evoxels, ehits, etracks, eblobs = get_event(dfs, evt)
        
#         edataclass                  = int(evoxels.binclass.unique())
        
#         bins, mask, cells, cloud   = get_cloud(evoxels, voxel_size, eblobs)
        
#         dfnodes = ana_nodes(cloud)
#         dfnodes['dataclass'] = edataclass
#         dfnodes['event']     = evt
         
#         df = dfnodes if df is None else pd.concat((df, dfnodes))
         
#     return df
        
        
    
    