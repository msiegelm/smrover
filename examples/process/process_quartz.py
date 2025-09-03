"""
Process RBR Quartz data from .rsk to nc


"""
from matplotlib import pyplot as plt
from smrover.process.RBR.process import *

plt.ion()
plt.rcParams['font.size'] = 16

#%% Raw .rsk File and Path 
fbase = '/Users/miks/Desktop/stateparks/smartmooring/data/seacliffs/rover/'
fpath  = fbase + "quartz/"
fname = "232364_20250403_1442" #base name without .rsk


#%% Save parameters 
svpath = fpath + 'processed/'
sname = "232364_20250403_1442_quartz"
spath = fbase
spath_proc = '/Users/miks/Desktop/stateparks/smartmooring/data/seacliffs/rover/quartz/processed/'
sname_proc = "232364_20250403_1442_quartz_proc"

#%%
## instrument deployment location
lat = 36.92967
lon = -121.93279 
## saves raw .rsk file to netCDF
rbr_rsk2nc_continuous(fpath,fname,sname,lat,lon,spath,instrumenttype="quartz") # raw files

## clips ends of time series to account for times instrument is out of the water
# if indmin = [] and indmax = [] then plot is made to assist with index selection
RBR_general_processing(spath,sname,spath_proc,sname_proc,patm_dbar=10.1325,indmin=66505,indmax=0)  # processed files
