#import numpy  as np
import pandas as pd
#import tables as tb


#---- DataFrame

group_by_event = lambda df: df.groupby('event')

def core_(source : callable,
          opera  : callable,
          group  : callable = group_by_event,
          sink   : callable = None,
          nevts  : int = 100):
    """
    
    run opera in the source data grouped by event and apply sink

    Parameters
    ----------
    source : callable or pd.DataFrame, function to provide the data or the data itself
    opera  : callable, operation, opera(evt-data) per event iteration
    group  : callable, optional, generator to iterate along events
             The default is group_by_event, that acts over pd.DataFrame
    sink   : callable, optional, last operation over the output data
             The default is None.
    nevts  : int, optional, printout the number of processed events
             The default is 100.

    Returns
    -------
    res    : the output of the event loop after sink.
             a pd.DataFrame if the otuput of opera(evt-data) is a pd.DataFrame 

    """

    data  = source() if callable(source) else source
    
    odata = []
    isdf = None
    icount = 0
    for ievt, evt in group(data):
        icount += 1
        if (icount % nevts == 0): print('event ', ievt)
        iodata = opera(evt)
        if (isdf is None):
            isdf = type(iodata) == pd.DataFrame
        if isdf:
            iodata['event'] = ievt
        else:
            iodata = (ievt, iodata)
        odata.append(iodata)
    if (isdf): 
        odata = pd.concat(odata, ignore_index = True)
    res = odata if sink is None else sink(odata)
    
    return res