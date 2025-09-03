import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import xarray as xr
from smrover.process.RDIadcp.process import writeBin2Pkl, load_pickle, adcp_dict2nc

#%%  Raw ADCP data
# Path of raw .bin file and base filename
fpath = '/Users/miks/Desktop/stateparks/smartmooring/data/seacliffs/rover/ADCP/Oct2024-Apr2025/'
fname = '_RDI_001.000'

## deployment location needed to save netCDF if unknown specify as []
lat = 36.92967
lon = -121.93279 
#%% File save parameters (.pkl and .nc)
svname = 'Seacliff_ADCP_raw'
svpath = '/Users/miks/Desktop/stateparks/smartmooring/data/seacliffs/rover/ADCP/Oct2024-Apr2025/raw/'

#%% Writes raw .bin file to .pkl file
writeBin2Pkl(fpath,fname,svpath,svname,num_av=1)

#%% Reads raw .pkl file
dat = load_pickle(svpath,svname+'.pkl')
#%% Saves subset of variables to netCDF
adcp_dict2nc(dat,processedby="M.Siegelman",lat=lat,lon=lon,saveas=True,svpath=svpath,svname=svname)

#%% Quick look at data
ds = xr.open_dataset(svpath + svname + '.nc')