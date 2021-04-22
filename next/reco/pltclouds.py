import numpy             as np
import pandas            as pd

import matplotlib.pyplot as plt

import hipy.pltext       as pltext
#import hipy.hfit         as hfit

#import bes.bes           as bes
import bes.clouds        as clouds
#import bes.chits         as chits
#import bes.display       as nplay


#---- utils

def get_cells(df, ndim):
    return [df['x'+str(i)].values for i in range(ndim)]


def _ocells(cells, i = 0):
    return cells[i:] + cells[:i]


def _csel(vals, sel):
    return [val[sel] for val in vals]


_karg = pltext.karg


#
# Low level elements
#

def dcloud_cells(cells, enes = None, xaxis = 0, **kargs):
    """ Draw cells, if enes, with its energy
    """

    ndim, nsize = len(cells), len(cells[0])

    enes = np.ones(nsize) if enes is None else enes

    kargs = _karg('marker',   's', kargs)
    kargs = _karg('c' ,  enes, kargs)
    #kargs = _karg('s' ,  10 * enes, kargs)

    ax = plt.gca()
    xcells = _ocells(cells, xaxis)
    ax.scatter(*xcells, **kargs)
    return
    #if (chamber): draw_chamber(coors, ax)


def dcloud_nodes(cells, enodes, **kargs):
    """ Draw cells that enodes > 0
    """

    kargs = _karg('marker',   '*', kargs)

    sel = enodes > 0

    kargs = _karg('s', enodes[sel], kargs)

    dcloud_cells(_csel(cells, sel), enodes[sel], **kargs)
    return


def dcloud_grad(cells, epath, xaxis = 0, **kargs):
    """ Draw the gradient of the cells
    """

    ndim = len(cells)
    ncells = _csel(cells, epath)
    coors  = _ocells(cells , xaxis) if xaxis != 0 else cells
    vcoors = _ocells(ncells, xaxis) if xaxis != 0 else ncells

    xdirs =[vcoor - coor for vcoor, coor in zip(vcoors, coors)]
    opts = {'scale_units': 'xy', 'scale' : 2.} if ndim == 2 else {'length' : 0.4}

    plt.quiver(*coors, *xdirs, **opts, **kargs)


def dcloud_segments(cells, kids, epath, lpath, xaxis = 0, **kargs):
    """ Draw the segments associated to the pass with epass > 0
    """

    xcells   = _ocells(cells, xaxis) if xaxis != 0 else cells
    segments = [clouds.get_segment(xcells, clouds.get_pass_path(kid, epath, lpath)) for kid in kids]
    for segment in segments:
        #print(segment)
        kargs = _karg('c', 'black', kargs)
        plt.plot(*segment, **kargs)


#
#  High Level
#

def dcloud_steps(dfclouds, ndim, xaxis = 0, ncolumns = 1):

    cells  = get_cells(dfclouds, ndim)
    enes   = dfclouds.ene.values
    enodes = dfclouds.enode.values
    nodes  = dfclouds.node.values
    epath  = dfclouds.epath.values
    lpath  = dfclouds.lpath.values
    epass  = dfclouds.epass.values


    sdim = '3d' if ndim == 3 else '2d'
    subplot = pltext.canvas(6, ncolumns, 10, 12)

    subplot(1, sdim) # cloud
    dcloud_cells(cells, 1000. * enes, alpha = 0.3, xaxis = xaxis);
    #dcloud_grad(cells, epath, xaxis = xaxis)

    subplot(2, sdim) # gradient (cloud)
    dcloud_cells(cells, 1000. * enes, alpha = 0.05, xaxis = xaxis);
    dcloud_grad(cells, epath, xaxis = xaxis)

    subplot(3, sdim) # nodes (grandient, cloud)
    dcloud_cells(cells, nodes, alpha = 0.05, xaxis = xaxis);
    dcloud_grad (cells, epath, alpha = 0.1, xaxis = xaxis)
    dcloud_nodes(cells, 1000. * enodes, marker = 'o', alpha = 0.8, xaxis = xaxis)

    subplot(4, sdim) # links (nodes, cloud)
    dcloud_cells(cells, 1000. * nodes, alpha = 0.01, xaxis = xaxis);
    dcloud_grad (cells, lpath, xaxis = xaxis)
    dcloud_nodes(cells, 1000. * enodes, marker = 'o', alpha = 0.2, xaxis = xaxis)

    subplot(5, sdim) # pass (links, nodes, cloud)
    dcloud_cells(cells,     1000. * nodes , alpha = 0.01, xaxis = xaxis);
    dcloud_nodes(cells,     1000. * enodes, marker = 'o', alpha = 0.2, xaxis = xaxis)
    dcloud_grad (cells, lpath, alpha = 0.1, xaxis = xaxis)
    dcloud_nodes(cells, 5 * 1000. * epass , marker = '^', alpha = 0.9, xaxis = xaxis)

    subplot(6, sdim)
    dcloud_cells   (cells, alpha = 0.05, xaxis = xaxis);
    dcloud_nodes   (cells, 1000. * enodes, alpha = 0.8, marker = 'o', xaxis = xaxis)
    dcloud_nodes   (cells, 5 * 1000. * epass , marker = '^', alpha = 0.9, xaxis = xaxis)
    kids  = np.argwhere(epass > 0)
    dcloud_segments(cells, kids, epath, lpath, xaxis = xaxis)

    return



def dcloud_steps_tracks(dfclouds, ndim, ncolumns = 1, xaxis = 0, **kargs):

    cells  = get_cells(dfclouds, ndim)
    enes   = dfclouds.ene.values
    enodes = dfclouds.enode.values
    nodes  = dfclouds.node.values
    epath  = dfclouds.epath.values
    lpath  = dfclouds.lpath.values
    epass  = dfclouds.epass.values

    track  = dfclouds.track.values
    tnode  = dfclouds.tnode.values
    tpass  = dfclouds.tpass.values


    sdim   = '3d' if ndim == 3 else '2d'
    scale  = 1000.
    rscale = 5.

    subplot = pltext.canvas(2, ncolumns, 10, 12)

    subplot(1, sdim)
    dcloud_cells   (cells, alpha = 0.05, xaxis = xaxis);
    dcloud_nodes   (cells, scale * enodes, alpha = 0.8, marker = 'o', xaxis = xaxis)
    dcloud_nodes   (cells, rscale * scale * epass , marker = '^', alpha = 0.9, xaxis = xaxis)
    kids  = np.argwhere(epass > 0)
    dcloud_segments(cells, kids, epath, lpath, xaxis = xaxis)
    plt.title('paths between passes')

    subplot(2, sdim)
    plt.title('tracks')
    kidtrack = np.unique(track)
    for ii, kid in enumerate(kidtrack):
        sel  = track == kid
        #print('kid ', kid, 'nodes', ckid[tnode == kid], ' kpass ', ckids[tpass == kid])
        dcloud_cells(_csel(cells, sel), alpha = 0.05, xaxis = xaxis)
        sel  = tnode == kid
        dcloud_nodes(_csel(cells, sel), scale * enodes[sel],  alpha = 0.8,
                           marker = 'o', xaxis = xaxis)
        sel  = tpass == kid
        dcloud_nodes(_csel(cells, sel), rscale * scale * epass[sel], alpha = 0.9,
                               marker = '^',  xaxis = xaxis)

        dcloud_segments(cells, np.argwhere(sel), epath, lpath, xaxis = xaxis)


    return


def dcloud_tracks_3dviews(dfclouds, ncolumns = 2, xaxis = 0, **kargs):

    ndim   = 3
    cells  = get_cells(dfclouds, ndim)
    enes   = dfclouds.ene.values
    enodes = dfclouds.enode.values
    #nodes  = dfclouds.node.values
    epath  = dfclouds.epath.values
    lpath  = dfclouds.lpath.values
    epass  = dfclouds.epass.values

    track  = dfclouds.track.values
    tnode  = dfclouds.tnode.values
    tpass  = dfclouds.tpass.values

    sdim   = '3d' if ndim == 3 else '2d'
    scale  = 1000.
    rscale = 5.

    subplot = pltext.canvas(4, ncolumns, 10, 12)

    xlabels = 2 * ['x (mm)', 'y (mm)', 'z (mm)']

    def _view(i, ii):
        subplot(ii + 1, sdim)
        plt.title('view ' + str(i))
        kidtrack = np.unique(track)
        for ii, kid in enumerate(kidtrack):
            sel  = track == kid
            dcloud_cells(_csel(cells, sel), alpha = 0.05, xaxis = i)
            sel  = tnode == kid
            dcloud_nodes(_csel(cells, sel), scale * enodes[sel],  alpha = 0.8,
                               marker = 'o', xaxis = i)
            sel  = tpass == kid
            dcloud_nodes(_csel(cells, sel), rscale * scale * epass[sel], alpha = 0.9,
                                   marker = '^',  xaxis = i)
            dcloud_segments(cells, np.argwhere(sel), epath, lpath, xaxis = i)
            ax = plt.gca()
            ax.set_xlabel(xlabels[i]); ax.set_ylabel(xlabels[i+1]); ax.set_zlabel(xlabels[i+2])

    for ii, i in enumerate([0, 2]):
        _view(i, ii)

    return
