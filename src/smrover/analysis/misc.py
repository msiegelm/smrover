import numpy as np
import copy
import glob
import os
import matplotlib.dates as dates

class Bunch:
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


def rangeslice(x, x1, x2):
    """
    Return a slice object for selecting values between x1 and x2
    from a monotonically increasing array-like `x`.
    
    Parameters
    ----------
    x : array-like (float or numpy.datetime64)
        Monotonically increasing values.
    x1 : float or numpy.datetime64
        Start of the desired range.
    x2 : float or numpy.datetime64
        End of the desired range.
    
    Returns
    -------
    slice
        Slice object that can be used to index `x`.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional and monotonically increasing.")

    # verify monotonic increase
    if not np.all(x[1:] >= x[:-1]):
        raise ValueError("x must be monotonically increasing.")

    # find indices
    i1 = np.searchsorted(x, x1, side="left")
    i2 = np.searchsorted(x, x2, side="right")

    return slice(i1, i2)