import numpy  as np
import pandas as pd
import tables as tb

import hipy.utils as ut


#---- DataFrame

to_df = pd.DataFrame.from_records

DATADIR = "/data_extra2/jrenner/analysis/NEW"


def get_esmeralda_filename(run_number):
    run_number = str(run_number)
    filename   = datadir + f'/{run_number}' + f'/hdf5/cdst/trigger2/cdst_combined_{run_number}' + '.h5'
    return filename


def energy_correction(energy, dz, alpha = 2.76e-4):
    """ Apply Josh's energy correction by delta-z effect
    """
    return energy/(1 - alpha * dz)


def load_esmeralda_dfs(filename):
    """ load DF from esmeralda:
    input:
        filename: str, complete filename
    returns:
        dfe, DF, event data frame
        dfs, DF, summary data frame
        dft, DF, tracks data frame
    """
    f = tb.open_file(filename, 'r')

    dfe = to_df(f.root.DST.Events.read())
    dfs = to_df(f.root.Summary.Events.read())
    dft = to_df(f.root.Tracking.Tracks.read())


    dft['dz_track'] = dft['z_max'] - dft['z_min']
    dft['enecor']   = energy_correction(dft['energy'].values, dft['dz_track']. values)

    return dfe, dfs, dft


# def get_esmeralda_dft(filenames):
#     """ return the track DF from Esmeralda filenames
#     inputs:
#         filenames: tup(str), list of complete Esmeralda's filenames
#     returns:
#         dft: DF, track data frame
#     """
#
#     dft = None
#
#     for i, filename in enumerate(filenames):
#
#         print('data filename: ', filename)
#
#         idfe, idfs, idft = load_esmeralda_dfs(filename)
#
#         dft = idft if i == 0 else dft.append(idft, ignore_index = True)
#
#     return dft

def get_esmeralda_dfcomposite(filename):

    dfe, dfs, dft = load_esmeralda_dfs(filename)

    dfe_section = dfe[['event', 'time', 'nS2', 'S1e', 'S2e', 'S2q', 'Nsipm']]
    dfs_section = dfs[['event', 'evt_energy', 'evt_ntrks', 'evt_nhits', 'evt_out_of_map']]
    dft_section = dft[['event', 'trackID', 'energy', 'length', 'numb_of_voxels',
                       'numb_of_hits', 'numb_of_tracks',
                       'x_min', 'y_min', 'z_min', 'r_min', 'x_max', 'y_max', 'z_max', 'r_max',
                       'x_ave', 'y_ave', 'z_ave', 'r_ave',
                       'extreme1_x', 'extreme1_y', 'extreme1_z',
                       'extreme2_x', 'extreme2_y', 'extreme2_z',
                       'blob1_x', 'blob1_y', 'blob1_z', 'blob2_x',
                       'blob2_y', 'blob2_z', 'eblob1', 'eblob2',
                       'ovlp_blob_energy', 'dz_track', 'enecor']]

    dd = pd.merge(dft_section, dfe_section, on = 'event')
    dd = pd.merge(dd         , dfs_section, on = 'event')

    return dd



def df_zeros(labels, nsize = 1):
    dat = {}
    for label in labels:
        type = int if label[0] in ['i', 'j', 'k', 'n'] else float
        dat[label] = np.zeros(nsize).astype(type)
    dat['event'] = np.zeros(nsize).astype(int)
    return pd.DataFrame(dat)


def df_concat(dfs, runs = None, label = 'run'):

    runs = runs if runs is not None else range(len(dfs))
    for i, df in enumerate(dfs):
        df[label] = runs[i]

    return pd.concat(dfs, ignore_index = True)


def df_isin(df, df_ref, labels = ['event', 'run'], offset = 10000000):

    assert len(labels) <= 2

    run    = len(labels) == 2

    label  = labels[0]
    a    = np.array(df     [label].values)
    aref = np.array(df_ref [label].values)

    if (run):

        assert (max(np.max(a), np.max(aref)) < offset)

        label1  = labels[1]
        a       = a     + offset * df    [label1].values
        aref    = aref  + offset * df_ref[label1].values

    oks  = np.isin(a, aref)
    return oks


