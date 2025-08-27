import numpy as np
import matplotlib.dates as dates
import copy

def nearest_time(t64, tstamp):
    """find index of nearest time in t64 to tstamp"""
    t64 = np.array(t64)
    idx = (np.abs(t64 - tstamp)).argmin()
    return idx  

def num2dt64(tnum):
    """
    Converts matplotlib datenum to numpy datetime64.

    Input:
    -----
    tnum = datenum

    Output:
    ------
    t    = numpy datetime64
    """
    tdates = dates.num2date(tnum)
    if np.any(np.shape(tdates)):
        t = np.array([np.datetime64(tdates[ii].strftime("%Y-%m-%dT%H:%M:%S")) for ii in range(len(tdates))])
    else:
        t = np.datetime64(tdates.strftime("%Y-%m-%dT%H:%M:%S"))
    return t

class Bunch:
    """
    Creates a Bunch object that allows attribute-style access to dictionary keys.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def keys(self):
        return self.__dict__.keys()

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f"Bunch({self.__dict__})"
 