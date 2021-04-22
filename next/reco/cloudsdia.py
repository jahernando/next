from collections import namedtuple

import numpy             as np

import clouds            as clouds

import next.core.io      as nio

def ana_tracks(evt, emin = 0.020):
    """
    
    ana tracks for an event

    Parameters
    ----------
    evt  : pd.DataFrame, clouds DF
    emin : float, minimum energy to accept a track
            The default is 0.020.

    Returns
    -------
    df   : pd.DataFrame, DF with info of the tracks

    """
    
    tracks_ids = np.unique(evt.track.values)
    enes       = np.array([np.sum(evt.ene[evt.track == ik]) for ik in tracks_ids], float)
    
    tracks_ids, enes = clouds.sorted_by_energy(tracks_ids, enes)
    
    tracks_ids = [id for id, ene in zip(tracks_ids, enes) if ene >= emin]

    ntracks = len(tracks_ids)
    
    def _track(ik):
        sel   = (evt.track == ik).values
        ene   = np.sum(evt.ene[sel])
        size  = np.sum(sel)
        xave  = np.mean(evt.x0[sel])
        yave  = np.mean(evt.x1[sel])
        dz    = np.max(evt.x2[sel]) - np.min(evt.x2[sel])
        dzmin = np.min(evt.x2[sel])
        nodes = np.sum(evt.tnode == ik)
        sel   = (evt.kid == ik).values
        enode = evt.enode[sel]
        snode = np.sum(evt.node == ik)
        return ik, ene, size, xave, yave, dz, dzmin, nodes, enode, snode
        
    labels = ('itrk', 'idtrk', 'trk_ene', 'ntrk_size',
              'trk_xave', 'trk_yave', 'trk_dz', 'trk_dzmin',
              'ntrk_nodes', 'trk_enode', 'ntrk_snode')
    df     = nio.df_zeros(labels, ntracks)
    for i, ik in enumerate(tracks_ids):
        data = _track(ik)
        df['itrk'][i] = i
        for k, label in enumerate(labels[1:]):
            df[label][i] = data[k]

    df['evt_ene']   = np.sum(evt.ene)
    df['nevt_trks'] = ntracks
            
    return df   


def sorted_tracks(evt, emin = 0.02):
    
    tracks_ids = np.unique(evt.track.values)
    enes       = np.array([np.sum(evt.ene[evt.track == ik]) for ik in tracks_ids], float)
    tracks_ids, enes = clouds.sorted_by_energy(tracks_ids, enes)
    
    tracks_ids = [id for id, ene in zip(tracks_ids, enes) if ene >= emin]

    return tracks_ids



branch_sum = namedtuple('branch_sum', 
                        ('id', 'ene', 'nsize', 'x', 'y', 'z', 'ecell', 'nlength', 'ipos', 'idis'))


def _extreme(branch, evt, depth = 2):
    
    #print('branch ', branch)
    knodes = branch[ -depth :]
    enes   = [float(evt.enode[evt.kid == k]) for k in knodes]
    #print('extreme nodes ', knodes)
    #print('extreme enes  ', enes)

    ik   = knodes[np.argmax(enes)]
    #print('extreme node ', ik)
    
    ipos = np.argwhere(branch == ik)
    #print('position ', ipos)

    nlength = len(branch)
    #print('length ', nlength)
    idis    = nlength - ipos - 1
    #print('ext dist     ', idis)
    
    ipos = np.argwhere(branch == ik)
    #print('position ', ipos)

    sel   = (evt.kid == ik).values
    unode = evt.enode[sel].values
    snode = np.sum(evt.node == ik)
        
    x     = evt.x0 [sel]
    y     = evt.x1 [sel]
    z     = evt.x2 [sel]
    ecell = evt.ene[sel]
    
    nlength = len(branch)
    
    return ik, unode, snode, x, y, z, ecell, nlength, ipos, idis
    

def ana_extremes(evt, emin = 0.02, depth = 2):

    #print('depth ', depth)
    
    ok = True
    while ok:
        npass = np.sum(evt.tpass > 0)
        #print('trim before', np.sum(evt.tpass > 0))
        evt        = clouds.trim(evt)
        #print('trim after', np.sum(evt.tpass > 0))
        ok = npass > np.sum(evt.tpass > 0)

    tracks_ids = sorted_tracks(evt, emin)
    trk_id = tracks_ids[0]
    #print('track id', trk_id)
    
    sel   = (evt.track == trk_id).values
    tpass = evt.tpass.values * np.array(sel, float)
    kids  = evt.kid  .values
    node  = evt.node .values
    lnode = evt.lnode.values
    enode = evt.enode.values

    #TODO Insted of trim off the evt, can we remove the nodes from the passes??

    passes = clouds.get_passes(tpass, node, lnode)
    #print('passes ', passes)
    udist  = clouds.nodes_idistance(passes)
    #print('node distances ', udist)
    branches_ = clouds.get_function_branches(passes)
    
    # get the blob candidates
    knodes = np.array(list(udist.keys()), int)
    enes   = np.array([enode[kids == ki][0] for ki in knodes], float)
    knodes, enes = clouds.sorted_by_energy(knodes, enes)
    #print('nodes ', knodes)
    #print('nodes energy ', enes)
    
    knodes = [k for k, ene in zip(knodes, enes) if ene > emin]
    #print('nodes with energy : ', knodes)
    kblobs = [k for k in knodes if udist[k] <= depth]
    #print('nodes in extremes :', kblobs)
    
    labels = [label+'_ext' for label in branch_sum._fields]
    
    kblob  = kblobs[0]
    #print('main blob ', kblob)
    brans  = branches_(kblob)
    #print('branches ', brans)
    brans  = [bran  for bran in brans if len(bran) > depth]
    if (len(brans) == 0): 
        df          = nio.df_zeros(labels, 1)
        df['i_ext'] = 0
        df['n_ext'] = 0
        print('no extremes !!')
        return df
    
    #print('branches ', brans)
    nbrans = len(brans)
    df     = nio.df_zeros(labels, 1 + nbrans)
    
    df['i_ext'] = np.arange(1 + nbrans)
    df['n_ext'] = 1 + nbrans
    
    # main blob
    bran     = [kblob,]
    dat      = _extreme(bran, evt, 1)
    for k, label in enumerate(labels):
        df[label][0] = dat[k]
    df['idis_ext'][0] = udist[kblob] -1 # moify the distante to the extreme
        
    # other extremes
    for i, bran in enumerate(brans):
        dat = _extreme(bran, evt, depth)
        for k, label in enumerate(labels):
            df[label][ i + 1] = dat[k]
            
    return df


def ana_blobs(evt, emin = 0.020, eblobmin = 0.020):
    
    #get the best energetic track
    tracks_ids = sorted_tracks(evt, emin)
    trk_id = tracks_ids[0]
    
    # select passes of the main track
    sel   = (evt.track == trk_id).values
    epass = evt.tpass.values * np.array(sel, float)
    
    # get the values
    kids  = evt.kid  .values
    node  = evt.node .values
    lnode = evt.lnode.values
    enode = evt.enode.values
    
    # get the passes
    passes    = clouds.get_passes(epass, node, lnode)
    #print('passes', passes)
    knodes    = np.unique(np.concatenate(passes))
    #print('nodes', knodes)
    enes   = np.array([enode[kids == ki][0] for ki in knodes], float)
    knodes, enes = clouds.sorted_by_energy(knodes, enes)
    #print('nodes ', knodes)
    #print('enes  ', enes)
    #knodes = [k for k, ene in zip(knodes, enes) if ene > eblobmin]
    #print('good nodes ', knodes)
    #passes = [ipass for ipass in passes if np.sum(np.isin(ipass, knodes)) == 2]
    #print('good passes ', passes)
    
    udist     = clouds.nodes_idistance(passes)    
    branches_ = clouds.get_function_branches(passes)
    
    # get the blob candidates
    knodes = np.array(list(udist.keys()), int)
    enes   = np.array([enode[kids == ki][0] for ki in knodes], float)
    knodes, enes = clouds.sorted_by_energy(knodes, enes)
    #for i, k in enumerate(knodes):
    #    pos = float(evt.x0[kids == k]), float(evt.x1[kids == k]), float(evt.x2[kids == k])
    
    knodes = [k for k, ene in zip(knodes, enes) if ene > eblobmin]
    kblobs = [k for k in knodes if udist[k] <= 2]
    
    kblob0 = kblobs[0]
    brans0 = branches_(kblob0)
    
    # select branches with more than 2 nodes
    brans0 = [len(bran) > 2 for bran in brans0]
    
    #bran0 = brans0[0]
    #for bran in brans0[1:]:
    #    if (len(bran) > len(bran0)): bran0 = bran
    #print('bran0', kblob0, bran0)
    
    def _track(ik):
        sel   = (evt.track == ik).values
        ene   = np.sum(evt.ene[sel])
        size  = np.sum(sel)
        xave  = np.mean(evt.x0[sel])
        yave  = np.mean(evt.x1[sel])
        dz    = np.max(evt.x2[sel]) - np.min(evt.x2[sel])
        dzmin = np.min(evt.x2[sel])
        nodes = np.sum(evt.tnode == ik)
        sel   = (evt.kid == ik).values
        return ik, ene, size, xave, yave, dz, dzmin, nodes
    
    def _blob(kblob):
        sel   = kids == kblob
        unode = enode[sel]
        snode = np.sum(evt.node == kblob)
        idis  = udist[kblob]
        
        x     = evt.x0[sel]
        y     = evt.x1[sel]
        z     = evt.x2[sel]
        ecell = evt.ene[sel]
        
        brans  = branches_(kblob)
        nbrans = len(brans)
        #ibran  = brans[0]
        #for bran in brans[1:]:
        #     if (len(bran) > len(ibran)): ibran = bran
        #print('ibran ', kblob, ibran)

        lbran  = np.max([len(bran) for bran in brans])
        
        kpos0 = [np.argwhere(bran == kblob)[0] \
                for bran in brans0 if len(np.argwhere(bran == kblob)) == 1]
        kpos0 = np.min(kpos0) if len(kpos0) > 0 else -1
        #kpos0 = np.argwhere(bran0 == kblob )[0] if len(np.argwhere(bran0 == kblob )) == 1 else -1 
        #kpos  = np.argwhere(ibran == kblob0)[0] if len(np.argwhere(ibran == kblob0)) == 1 else -1 

    
        kpos1 = [np.argwhere(bran == kblob0)[0] \
                for bran in brans if len(np.argwhere(bran == kblob0)) == 1]
        kpos1 = np.min(kpos1) if len(kpos1) > 0 else -1
    
        return kblob, unode, snode, idis, x, y, z, ecell, nbrans, lbran, kpos0, kpos1
    
        
    labels = ('iblob', 'idblob', 'blob_e', 'blob_size', 'iblob_dist',
              'blob_x', 'blob_y', 'blob_z', 'blob_ecell',
              'nblob_brans', 'nblob_lbran',
              'iblob_i0', 'iblob_i1')
        
    nblobs = len(kblobs)
    df = nio.df_zeros(labels, nblobs)
    for i, kb in enumerate(kblobs):
        data = _blob(kb)
        df['iblob'][i] = i
        for k, label in enumerate(labels[1:]):
            df[label][i] = data[k]
        
    df['nblobs']    = nblobs
    df['evt_ene']   = np.sum(evt.ene)
    df['nevt_size'] = len(evt.ene)
    
    trk_ik, trk_ene, trk_size, trk_xave, trk_yave, trk_dz, trk_zmin, trk_nodes = _track(kblob0)
    df['trk_id']     = trk_id
    df['trk_ene']    = trk_ene
    #df['ntrk_size']  = trk_size
    #df['trk_xave']   = trk_xave
    #df['trk_yave']   = trk_yave
    df['trk_dz']     = trk_dz
    #df['trk_zmin']   = trk_zmin
    #df['ntrk_nodes'] = trk_nodes

    return df


# #import tables            as tb

# #to_df = pd.DataFrame.from_records
# from invisible_cities.reco import corrections as cof

# #import hipy.utils        as ut
# #import bes.bes           as bes
# #import bes.chits         as chits
# #import clouds            as clouds



# group_by_event = lambda df: df.groupby('event')

# def core_(source, opera, group = group_by_event, sink = None):

#     data  = source() if callable(source) else source
#     odata = []
#     for ievt, evt in group(data):
#         iodata = opera(evt)
#         odata.append( (ievt, iodata))
#     res = odata if sink is None else sink(odata)
#     return res
    
        

# #get_chits_filename = chits.get_chits_filename
# #get_krmap_filename = chits.get_krmap_filename
# #get_maps           = chits.get_maps
# #get_hits           = chits.get_hits


# def load_data(runs, sample_label = 'ds'):
#     """ load hits and maps data
#     """

#     fnames  = [get_chits_filename(run, sample_label + '_rough') for run in runs]
#     print(fnames)
#     dfhits   = [pd.read_hdf(fname, 'CHITs.lowTh') for fname in fnames]
#     dfhitHTs = [pd.read_hdf(fname, 'CHITs.highTh') for fname in fnames]


#     fnames = [get_krmap_filename(run) for run in runs]
#     dmaps  = [get_maps(fname) for fname in fnames]

#     return (dfhits, dfhitHTs, dmaps)


# def get_corrfac(maps):
#     """ given maps, return the correction factor function based on (x, y, z, times)
#     """

#     vdrift    = np.mean(maps.t_evol.dv)
#     print('drift velocity ', vdrift)
#     _corrfac  = cof.apply_all_correction(maps, apply_temp = True,
#                                          norm_strat = cof.norm_strategy.kr)
#     def corrfac(x, y, z, times):
#         dt = z/vdrift
#         return _corrfac(x, y, dt, times)

#     return corrfac



# def cloudsdia(runs, sample_label = 'ds', ntotal = 10000, type_hits = 'LT', q0 = 0.):
#     """ a City: read the hits and the maps, runs clouds and recompute energy, returns a DF
#     """

#     dfhitsLT, dfhitsHT, dmaps = load_data(runs, sample_label)

#     dfhits = dfhitsLT if type_hits == 'LT' else dfhitsHT

#     ddhs   = [cloudsdia_(dfh, dmap, ntotal, q0) for dfh, dmap in zip(dfhits, dmaps)]

#     ndfs = len(ddhs[0])
#     dfs = [bes.df_concat([ddh[i] for ddh in ddhs], runs) for i in range(ndfs)]

#     return dfs


# def get_clouds(evt, corrfac, q0 = 0.):

#     x, y, z, eraw, erec, q, times = chits.get_filter_hits(evt, q0)

#     # clouds
#     coors = (x, y, z)
#     steps = (10., 10., 2.)
#     dfclouds = clouds.clouds(coors, steps, eraw)
#     in_cells = clouds.get_values_in_cells(coors, steps, eraw)
#     dfclouds['erec'], _, _ = in_cells(coors, erec)
#     dfclouds['eraw'], _, _ = in_cells(coors, eraw)
#     dfclouds['q'], _, _    = in_cells(coors, q)

#     # clean nodes
#     #newcoors, newenes = clean_nodes(dfclouds)
#     #dfclouds = clouds.clouds(newcoors, steps, newenes)

#     # calibrate
#     dfclouds = cloud_calibrate(dfclouds, corrfac, times[0])

#     return dfclouds


# def cloudsdia_(dfhit, maps, ntotal = 10000, q0 = 0.):

#     def _locate(idata, data, i, ievent):
#         data['event'][i] = ievent
#         for label in idata.keys():
#             data[label][i] = idata[label]
#         return data

#     def _concat(idata, data, ievent):
#         idata['event'] = ievent
#         dat            = idata if data is None else pd.concat((data, idata), ignore_index = True)
#         return dat

#     corrfac = get_corrfac(maps)

#     nsize   = len(dfhit.groupby('event'))
#     n       = min(nsize, ntotal)

#     dfsum_hits   = chits.init_hits_summary(n)
#     dfsum_slices = None

#     dfsum_clouds = init_cloud_summary(n)

#     n = -1
#     for i, evt in dfhit.groupby('event'):

#         n += 1
#         if (n >= ntotal): continue

#         idat = chits.hits_summary(evt, q0, corrfac)
#         dfsum_hits = _locate(idat, dfsum_hits, n, i)
#         #print(idat)

#         idat = chits.slices_summary(evt, q0, corrfac)
#         dfsum_slices = _concat(idat, dfsum_slices, i)

#         if (n % 100 == 0):
#             print('event : ', i)

#         # clouds
#         if (n == 1 and q0 < 3.):
#             print('info: Clouds will run with q = 3')
#         dfclouds = get_clouds(evt, corrfac, max(q0, 3))
#         idat     = cloud_summary(dfclouds)
#         dfsum_clouds = _locate(idat, dfsum_clouds, n, i)

#     return [dfsum_hits, dfsum_slices, dfsum_clouds]

# #
# # def cloudsdia_(dfhit, dfhitHT, maps, ntotal = 100000):
# #     """ City Engine: loops in events, run clouds and makes the summary
# #     """
# #
# #     def hits_summary(x, y, z, eraw, erec, q):
# #         rmax = np.max(np.sqrt(x * x + y * y))
# #         idat = {'eraw'  : np.sum(eraw),
# #                 'erec'  : np.sum(erec),
# #                 'q'     : np.sum(q),
# #                 'nhits' : len(x),
# #                 'zmin'  : np.min(z),
# #                 'zmax'  : np.max(z),
# #                 'dz'    : np.max(z) - np.min(z),
# #                 'rmax'  : np.max(np.sqrt(x * x + y * y))
# #                 }
# #         return idat
# #
# #
# #     corrfac = get_corrfac(maps)
# #
# #     nsize = len(dfhit.groupby('event'))
# #     print('size', nsize)
# #
# #     labels  = ['event', 'eraw', 'erec', 'q', 'nhits', 'zmin', 'zmax', 'dz', 'rmax',
# #                'erawHT', 'erecHT', 'qHT', 'nhitsHT', 'zminHT', 'zmaxHT', 'dzHT', 'rmaxHT']
# #     labels += ['evt_ntracks', 'evt_nisos', 'evt_eisos', 'evt_ncells', 'evt_nnodes', 'evt_nrangs',
# #                'evt_ecells', 'evt_enodes', 'evt_erangs', 'evt_outcells', 'evt_outnodes', 'evt_outrangs',
# #                'evt_zmin', 'evt_zmax', 'evt_dz', 'evt_rmax', 'evt_enode1', 'evt_enode2',
# #                'trk_ncells', 'trk_nnodes', 'trk_nrangs', 'trk_ecells', 'trk_enodes', 'trk_erangs',
# #                'trk_outcells', 'trk_outnodes', 'trk_outrangs',
# #                'trk_zmin', 'trk_zmax', 'trk_dz', 'trk_rmax', 'trk_enode1', 'trk_enode2']
# #
# #     dat = {}
# #     for label in labels:
# #         dat[label] = np.zeros(min(nsize, ntotal))
# #
# #     dfiso    = None
# #     dfslice  = None
# #
# #     n = -1
# #     for i, evt in dfhit.groupby('event'):
# #
# #         n += 1
# #         if (n >= ntotal): continue
# #
# #         dat['event'][n] = i
# #
# #         # get HT hits info
# #         evtHT = dfhitHT.groupby('event').get_group(i)
# #         x, y, z, eraw, erec, q, times = get_hits(evtHT, ['X', 'Y', 'Z', 'E', 'Ec', 'Q', 'time'])
# #         idat = hits_summary(x, y, z, eraw, erec, q)
# #         for key in idat.keys():
# #             dat[key + 'HT'][n] = idat[key]
# #
# #         # get hits info
# #         x, y, z, eraw, erec, q, times = get_hits(evt, ['X', 'Y', 'Z', 'E', 'Ec', 'Q', 'time'])
# #         idat = hits_summary(x, y, z, eraw, erec, q)
# #         for key in idat.keys():
# #             dat[key][n] = idat[key]
# #
# #         if (n % 100 == 0):
# #             print('event : ', i, ', size : ', len(eraw))
# #
# #         # clouds
# #         coors = (x, y, z)
# #         steps = (10., 10., 2.)
# #         dfclouds = clouds.clouds(coors, steps, eraw)
# #         in_cells = clouds.get_values_in_cells(coors, steps, eraw)
# #         dfclouds['erec'], _, _ = in_cells(coors, erec)
# #         dfclouds['eraw'], _, _ = in_cells(coors, eraw)
# #         dfclouds['q'], _, _    = in_cells(coors, q)
# #         dfclouds = cloud_calibrate(dfclouds, corrfac, times[0])
# #
# #         ## info from clouds
# #         idat = cloud_summary(dfclouds)
# #         for key in idat.keys():
# #             dat[key][n] = idat[key]
# #         #key = 'evt_outcells'
# #         #print(key, idat[key], dat[key][n])
# #
# #         # summary of isolated clouds
# #         idfiso = cloudsdia_iso_summary(dfclouds)
# #         idfiso['event'] = i
# #         dfiso = idfiso if dfiso is None else pd.concat((dfiso, idfiso), ignore_index = True)
# #
# #         # summary of slices
# #         idfslice = cloudsdia_slice_summary(dfclouds)
# #         idfslice['event'] = i
# #         dfslice = idfslice if dfslice is None else pd.concat((dfslice, idfslice), ignore_index = True)
# #
# #
# #
# #     dfsum = pd.DataFrame(dat)
# #     return dfsum, dfiso, dfslice
# #

# def cloud_order_tracks(df):
#     """ returns the ids of the tracks ordered by energy
#     """

#     rangers  = df.ranger .values
#     erangers = df.eranger.values

#     # get the largest energetic track
#     sel      = rangers > -1
#     kids     = np.unique(rangers[sel])
#     enes     = np.array([np.sum(erangers[rangers == kid]) for kid in kids])
#     kids, enes = clouds.sorted_by_energy(kids, enes)
#     enes     = np.array(enes)
#     return kids, enes

# def init_cloud_summary(nsize = 1):
#     labels = ['evt_ntracks', 'evt_nisos', 'evt_eisos', 'evt_ncells', 'evt_nnodes', 'evt_nrangs',
#               'evt_ecells', 'evt_enodes', 'evt_erangs', 'evt_outcells', 'evt_outnodes', 'evt_outrangs',
#               'evt_zmin', 'evt_zmax', 'evt_dz', 'evt_rmax', 'evt_enode1', 'evt_enode2',
#               'trk_ncells', 'trk_nnodes', 'trk_nrangs', 'trk_ecells', 'trk_enodes', 'trk_erangs',
#               'trk_outcells', 'trk_outnodes', 'trk_outrangs',
#               'trk_zmin', 'trk_zmax', 'trk_dz', 'trk_rmax', 'trk_enode1', 'trk_enode2']
#     return bes.df_zeros(labels, nsize)


# def cloud_summary(df):
#     """ returns a summary of the cloud results (a dictionary)
#     """

#     # get data
#     x        = df.x0     .values
#     y        = df.x1     .values
#     z        = df.x2     .values
#     ecells   = df.ene    .values
#     enodes   = df.enode  .values
#     erangs   = df.eranger.values
#     trackid  = df.track  .values
#     rangers  = df.ranger .values
#     #erangers = df.eranger.values
#     nsize    = len(x)
#     labels   = list(df.columns)
#     cout     = df.cout   .values if 'cout' in labels else np.full(nsize, False)

#     # get the largest energetic track
#     kids, enes = cloud_order_tracks(df)
#     kid_track_best = kids[0]
#     ntracks        = len(kids)

#     # compute isolated tracks
#     nran_trk = np.array([np.sum(rangers == kid) for kid in kids])
#     ksel     = nran_trk == 1
#     nisos    = np.sum(ksel)  # number of isolated ranges
#     eisos    = np.sum(enes[ksel]) # energy of the isolated ranges

#     # general information about tracks and isolated tracks
#     dvals = {'evt_ntracks' : ntracks,
#              'evt_nisos'   : nisos,
#              'evt_eisos'   : eisos}


#     # selections
#     enodes0   = df.enode0.values   if 'enodes0'   in labels else enodes
#     erangs0   = df.eranger0.values if 'erangers0' in labels else erangs
#     sel_nodes = enodes0 > 0.
#     sel_rangs = erangs0 > 0.

#     def _vals(sel = None):

#         sel = np.full(nsize, True) if sel is None else sel

#         ncells   = np.sum(sel)
#         nnodes   = np.sum(sel & sel_nodes)
#         nrangs   = np.sum(sel & sel_rangs)

#         esumcells = np.sum(ecells[sel])
#         esumnodes = np.sum(enodes[sel])
#         esumrangs = np.sum(erangs[sel])

#         outcells  = np.sum(cout[sel])
#         outnodes  = np.sum(cout[sel & sel_nodes])
#         outrangs  = np.sum(cout[sel & sel_rangs])

#         zmin = np.min(z[sel])
#         zmax = np.max(z[sel])
#         dz   = zmax - zmin
#         rmax = np.max(np.sqrt(x[sel]*x[sel] + y[sel]*y[sel]))

#         xenodes = np.sort(enodes[sel & sel_nodes])
#         enode1  = xenodes[-1]
#         enode2  = xenodes[-2] if (len(xenodes) >= 2) else 0.

#         vals = {'ncells'   : ncells,
#                 'nnodes'   : nnodes,
#                 'nrangs'   : nrangs,
#                 'ecells'   : esumcells,
#                 'enodes'   : esumnodes,
#                 'erangs'   : esumrangs,
#                 'outcells' : outcells,
#                 'outnodes' : outnodes,
#                 'outrangs' : outrangs,
#                 'zmin'     : zmin,
#                 'zmax'     : zmax,
#                 'dz'       : dz,
#                 'rmax'     : rmax,
#                 'enode1'   : enode1,
#                 'enode2'   : enode2}

#         return vals

#     # info from the event
#     dval = _vals()
#     for label in dval.keys():
#         dvals['evt_' + label] = dval[label]

#     # info from the best track
#     kidbest = kids[0]
#     dval = _vals(trackid == kid_track_best)
#     for label in dval.keys():
#         dvals['trk_' + label] = dval[label]

#     #for label in dvals.keys():
#     #    print(label, dvals[label])

#     return dvals


# def cloud_calibrate(df, corrfac, itime):

#     x, y, z = df.x0.values, df.x1.values, df.x2.values

#     nsize  = len(x)
#     times  = itime * np.ones(nsize)
#     cfac   = corrfac(x, y, z, times)
#     cout   = np.isnan(cfac)
#     cfac[cout] = 0.

#     df['cfac']     = cfac
#     df['cout']     = cout

#     df['ene0']     = df['ene'].values[:]
#     df['ene']      = cfac * df['ene'].values

#     df['enode0']   = df['enode'].values[:]
#     df['enode']    = cfac * df['enode'].values

#     df['eranger0'] = df['eranger'].values[:]
#     df['eranger']  = cfac * df['eranger'].values

#     return df


# def cloudsdia_iso_summary(df):

#     x        = df.x0     .values
#     y        = df.x1     .values
#     z        = df.x2     .values
#     ecells   = df.ene    .values
#     enodes   = df.enode  .values
#     erangs   = df.eranger.values
#     trackid  = df.track  .values
#     rangers  = df.ranger .values

#     cout     = df.cout   .values
#     q        = df.q      .values
#     erec     = df.erec   .values
#     eraw     = df.eraw   .values


#     # order the tracks by energy
#     kids, enes = cloud_order_tracks(df)

#     # compute isolated tracks
#     nran_trk = np.array([np.sum(trackid == kid) for kid in kids])

#     ksel     = nran_trk == 1
#     nisos    = np.sum(ksel)  # number of isolated ranges
#     eisos    = np.sum(enes[ksel]) # energy of the isolated ranges

#     #best track
#     kid_track_best = kids[0]
#     dz = np.max(z[trackid == kid_track_best]) - np.min(z[trackid == kid_track_best])

#     ksel     = np.array(kids).astype(int)[ksel]
#     #print(ksel)

#     idat = {'x' : x[ksel],
#             'y' : y[ksel],
#             'z' : z[ksel],
#             'q' : q[ksel],
#             'erec': erec[ksel],
#             'eraw': eraw[ksel],
#             'out' : cout[ksel],
#             'xb'  : np.ones(nisos) * x[kid_track_best],
#             'yb'  : np.ones(nisos) * y[kid_track_best],
#             'zb'  : np.ones(nisos) * z[kid_track_best],
#             'dz'  : np.ones(nisos) * dz
#            }

#     return pd.DataFrame(idat)


# def cloudsdia_slice_summary(df):

#     z        = df.x2     .values
#     k2       = df.k2     .values
#     kids     = df.kid    .values
#     trackid  = df.track  .values

#     ecells   = df.ene    .values
#     enode    = df.enode  .values
#     tnode    = df.tnode  .values

#     q        = df.q      .values
#     erec     = df.erec   .values
#     eraw     = df.eraw   .values

#     ks      = np.unique(k2)
#     nslices = np.max(ks) + 1
#     kzs     = np.zeros(nslices)
#     zs      = np.zeros(nslices)
#     e0s     = np.zeros(nslices)
#     q0s     = np.zeros(nslices)
#     es      = np.zeros(nslices)
#     nisos   = np.zeros(nslices)
#     eisos   = np.zeros(nslices)
#     nnodes  = np.zeros(nslices)
#     enodes  = np.zeros(nslices)


#     # order the tracks by energy
#     ckids, enes = cloud_order_tracks(df)

#     # compute isolated tracks
#     nran_trk = np.array([np.sum(trackid == kid) for kid in ckids])
#     ksel     = nran_trk == 1
#     #nisos    = np.sum(ksel)  # number of isolated ranges
#     #eisos    = np.sum(enes[ksel]) # energy of the isolated ranges

#     kids_isos   = np.array(ckids).astype(int)[ksel]

#     for i, k in enumerate(ks):

#         sel      = k2 == k
#         kzs[k]   = k
#         zs[k]    = np.mean(z[sel])
#         e0s[k]   = np.sum(eraw[sel])
#         q0s[k]   = np.sum(q[sel])
#         es[k]    = np.sum(erec[sel])

#         # isos
#         ksel     = np.logical_and(sel, np.isin(kids, kids_isos))
#         nisos[k] = np.sum(ksel)
#         eisos[k] = np.sum(ecells[ksel])

#         # nodes
#         ksel      = np.logical_and(sel, ~np.isin(kids, kids_isos))
#         ksel      = np.logical_and(ksel, tnode > 0)
#         nnodes[k] = np.sum(ksel)
#         enodes[k] = np.sum(enode[ksel])


#     idat = {'k'      : kzs,
#             'z'      : zs,
#             'eraw'   : e0s,
#             'q'      : q0s,
#             'erec'   : es,
#             'nisos'  : nisos,
#             'eisos'  : eisos,
#             'nnodes' : nnodes,
#             'enodes' : enodes
#            }

#     return pd.DataFrame(idat)
