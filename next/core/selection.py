#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:30:15 2021

@author: hernando
"""

import numpy  as np
import pandas as pd
import tables as tb

import hipy.utils as ut

#----  Selections

def dfesme_ranges():

    ranges = {}

    ranges['numb_of_tracks.one']   = (0.5, 1.5)

    ranges['evt_out_of_map.False']  = (False, False)

    #ranges['nS2'] = (0.5, 1.5)

    ranges['energy']    = (0., 3.)

    ranges['energy.cs'] = (0.65, 0.71)
    ranges['energy.ds'] = (1.55, 1.75)
    ranges['energy.ph'] = (2.5, 3.)

    ranges['enecor.cs'] = (0.65, 0.71)
    ranges['enecor.ds'] = (1.55, 1.75)
    ranges['enecor.ph'] = (2.5, 3.)

    ranges['z_min']  = (50., 500.)
    ranges['z_max']  = (50., 500.)
    ranges['r_max']  = ( 0., 180.)

    ranges['dz_track.cs']  = ( 8., 32.)
    ranges['dz_track.ds']  = (24., 72.)
    ranges['dz_track.ph'] = (35., 130.)

    return ranges


def dfesme_selections(df):

    ranges     = dfesme_ranges()
    selections = Selections(df, ranges)

    selections.logical_and(('evt_out_of_map.False', 'numb_of_tracks.one', 'energy',
                            'z_min', 'z_max', 'r_max'), 'fidutial')

    selections.logical_and(('fidutial', 'energy.cs'), 'fidutial.cs')
    selections.logical_and(('fidutial', 'energy.ds'), 'fidutial.ds')
    selections.logical_and(('fidutial', 'energy.ph'), 'fidutial.ph')

    return selections, ranges


class Selections:
    """ dictorinay to hold selection (np.array(bool)) with some extendions:
    """

    class Sel(np.ndarray): pass

    def _sel(sel, info):

        csel = sel.view(Selections.Sel)
        csel.info = info
        return csel

    def __init__(self, df, ranges = None):

        self.df   = df
        self.size = len(df)
        self.sels = dict()

        if ranges is not None:
            for key in ranges.keys():
                self.set_range(key, ranges[key])

        return

    def __getitem__(self, key):

        return self.sels[key]

    def keys(self):

        return self.sels.keys()

    def __str__(self):
        s = ''
        for key in self.sels.keys():
            nevt = np.sum(self[key])
            ieff = nevt/self.size
            s += key + ' : ' + self[key].info + ', '
            s += str(nevt) + ', ' + '{0:6.5f}'.format(ieff) +'\n'
            # s += ' entries '+ str(nevt) + ', efficiency ' + '{0:6.5f}'.format(ieff) +'\n'
        return s

    def set_isin(self, dfref, name = 'isin',
                 labels = ['event', 'run'], offset = 10000000):

        sel = df_isin(self.df, dfref, labels = labels, offset = offset)

        self.sels[name] = Selections._sel(sel, name)

        return self[name]

    def set_range(self, name, range = None, varname = None, upper_limit_in = False):

        varname = varname if varname is not None else name.split('.')[0]

        if (range is not None) and (range[0] == range[1]):
            upper_limit_in = True

        sel = ut.in_range(self.df[varname], range, upper_limit_in)

        ss = str(varname) + ' [' + str(range[0]) + ', ' + str(range[1])
        cs = ']' if upper_limit_in is True else ')'
        ss += cs

        self.sels[name] = Selections._sel(sel, ss)

        return self[name]

    def logical_and(self, names, name = None):
        """ return the selection that is the logical and of the names selections
        names: list of names of the selections
        """

        assert len(names) >= 2
        name0, name1 = names[0], names[1]
        sel = self[name0] & self[name1]

        for iname in names[2:]:
            sel = sel & self[iname]

        if (name in self.sels.keys()):
            print('overwriting ', name, ' selection')

        ss = ''
        for iname in names:
            ss += self[iname].info
            if (iname != names[-1]): ss += ' & '
        #ss = [self[iname].info  for iname in names]
        csel = Selections._sel(sel, str(ss))

        if (name is not None):
            self.sels[name] = csel

        return csel
