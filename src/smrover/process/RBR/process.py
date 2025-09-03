import xarray as xr
import matplotlib.pyplot as plt
import operator as op
import numpy as np
from pyrsktools import RSK
import matplotlib.dates as dates

def dbar2m(P,lat):
    """
    Converts pressure [dbar] to depth [m]. Takes into account latitudinal variations in gravity
    From sw_dpth.m
    % Eqn 25, p26.  Unesco 1983.

    Input:
    ------
    P = pressure [dbar]
    lat = latitude [degrees]

    Output:
    ------
    Depth [m]
    """

    DEG2RAD = np.pi/180
    c1 = +9.72659
    c2 = -2.2512E-5
    c3 = +2.279E-10
    c4 = -1.82E-15
    gam_dash = 2.184e-6

    lat = np.abs(lat)
    X = np.sin(lat*DEG2RAD)

    X = X*X
    bot_line = 9.780318*(1.0+(5.2788E-3+2.36E-5*X)*X) + gam_dash*0.5*P
    top_line = (((c4*P+c3)*P+c2)*P+c1)*P
    DEPTHM   = top_line/bot_line

    return DEPTHM


def minsec2decdeg(deg,minutes,seconds):
    mins = seconds/60
    decdeg = (minutes + mins)/60
    decdegs = deg + decdeg
    return decdegs

def rbr_rsk2nc_continuous(fpath,fname,sname,lat,lon,sdir,instrumenttype=[]):

    """
    Read .rsk RBR file and output data as .netcdf

    Input:
    -----
    fpath = location of .xlsx file
    fname = file name [string w/o extension]
    sname = save name [string w/o extension]
    lat = latitude [decimal degrees]
    lon = longitude [decimal degrees]
    sdir = path to save file [string]

    Output:
    ------
    NetCDF file named sname in sdir

    """
    rsk = RSK(fpath + fname + ".rsk")
    rsk.open()
    rsk.readdata()
    if instrumenttype == "quartz":
        rsk.open()
        rsk.readdata()
        rsk.deriveBPR()
        t64 = rsk.data["timestamp"]
        temp = rsk.data["bpr_temperature"]
        pdbar = rsk.data["bpr_pressure"]

    else:
    # pd = rsk.data

        t64 = rsk.data["timestamp"]
        temp = rsk.data["temperature"]
        pdbar = rsk.data["pressure"]

    # bn = rsk.scheduleInfo.samplingCount
    # nb = rsk.deployment.sampleSize
    # burstn = [1]
    #
    # for jn in range(1,len(tsec)):
    #     dt = tsec[jn] - tsec[jn-1]
    #     if dt < dtcount*dt0:
    #         burstn.append(burstn[jn-1])
    #     else:
    #         burstn.append(burstn[jn-1]+1)
    # burstn = np.array(burstn)

    data_vars=dict(pdbar=(["time"], pdbar),temp=(["time"], temp))
    coords=dict(time=t64,lat = lat,lon =lon,)

    attrs=dict(description=("Raw Data from Palau Wave Gauge: %s"%fname), location = ("lat: %.03f, lon: %.03f"%(lat,lon)))

    ds = xr.Dataset(data_vars=data_vars,coords=coords,attrs=attrs)

    ds["pdbar"].attrs = {'units':'dbar', 'long_name':'total pressure'}
    ds["temp"].attrs = {'units':'deg C', 'long_name':'temperature'}

    ds.to_netcdf(sdir + sname + ".nc")
    rsk.close()
    print("File exported to: %s"%sdir)



def rbr_rsk2nc_concerto(fpath,fname,sname,lat,lon,sdir):

    """
    Read .rsk RBR file and output data as .netcdf

    Input:
    -----
    fpath = location of .xlsx file
    fname = file name [string w/o extension]
    sname = save name [string w/o extension]
    lat = latitude [decimal degrees]
    lon = longitude [decimal degrees]
    sdir = path to save file [string]

    Output:
    ------
    NetCDF file named sname in sdir

    """
    rsk = RSK(fpath + fname + ".rsk")
    rsk.open()
    rsk.readdata()
    rsk.derivesalinity()
    pd = rsk.data

    t64 = rsk.data["timestamp"]
    cond = rsk.data["conductivity"]
    temp = rsk.data["temperature"]
    pdbar = rsk.data["pressure"]
    salt = rsk.data["salinity"]

    # tnum = dates.date2num(t64)
    # tsec = (tnum-tnum[0])*86400
    # dt0 = np.min(np.diff(tsec))
    # bn = rsk.scheduleInfo.samplingCount
    # nb = rsk.deployment.sampleSize
    # burstn = [1]
    #
    # for jn in range(1,len(tsec)):
    #     dt = tsec[jn] - tsec[jn-1]
    #     if dt < dtcount*dt0:
    #         burstn.append(burstn[jn-1])
    #     else:
    #         burstn.append(burstn[jn-1]+1)
    # burstn = np.array(burstn)

    data_vars=dict(pdbar=(["time"], pdbar),temp=(["time"], temp),cond=(["time"], cond),salt=(["time"], salt))
    coords=dict(time=t64,lat = lat,lon =lon,)

    attrs=dict(description=("Raw Data from Palau Wave Gauge: %s"%fname), location = ("lat: %.03f, lon: %.03f"%(lat,lon)))

    ds = xr.Dataset(data_vars=data_vars,coords=coords,attrs=attrs)

    ds["pdbar"].attrs = {'units':'dbar', 'long_name':'total pressure'}
    ds["temp"].attrs = {'units':'deg C', 'long_name':'temperature'}
    ds["cond"].attrs = {'units':'mS/cm', 'long_name':'conductivity'}
    ds["salt"].attrs = {'units':'PSU', 'long_name':'salinity'}

    ds.to_netcdf(sdir + sname + ".nc")
    rsk.close()
    print("File exported to: %s"%sdir)

def rbr_rsk2nc(fpath,fname,sname,lat,lon,sdir,dtcount=100):

    """
    Read .xlsx RBR file and output data as .netcdf

    Input:
    -----
    fpath = location of .xlsx file
    fname = file name [string w/o extension]
    sname = save name [string w/o extension]
    lat = latitude [decimal degrees]
    lon = longitude [decimal degrees]
    sdir = path to save file [string]
    dtcount = number of samples dt to assume that we are on a new burst [int]
                (don't make dtcount*dt longer than the time interval between bursts)
    Output:
    ------
    NetCDF file named sname in sdir

    """
    rsk = RSK(fpath + fname + ".rsk")
    rsk.open()
    rsk.readprocesseddata()
    pd = rsk.processedData

    t64 = [pd[ii][0] for ii in range(len(pd))]
    temp = [pd[ii][1] for ii in range(len(pd))]
    pdbar = [pd[ii][2] for ii in range(len(pd))]
    tnum = dates.date2num(t64)
    tsec = (tnum-tnum[0])*86400
    dt0 = np.min(np.diff(tsec))
    # bn = rsk.scheduleInfo.samplingCount
    # nb = rsk.deployment.sampleSize
    burstn = [1]

    for jn in range(1,len(tsec)):
        dt = tsec[jn] - tsec[jn-1]
        if dt < dtcount*dt0:
            burstn.append(burstn[jn-1])
        else:
            burstn.append(burstn[jn-1]+1)
    burstn = np.array(burstn)

    data_vars=dict(burstn=(["time"], burstn),pdbar=(["time"], pdbar),temp=(["time"], temp))
    coords=dict(time=t64,lat = lat,lon =lon,)

    attrs=dict(description=("Raw Data from Palau Wave Gauge: %s"%fname), location = ("lat: %.03f, lon: %.03f"%(lat,lon)))

    ds = xr.Dataset(data_vars=data_vars,coords=coords,attrs=attrs)

    ds["pdbar"].attrs = {'units':'dbar', 'long_name':'total pressure'}
    ds["temp"].attrs = {'units':'deg C', 'long_name':'total pressure'}

    ds.to_netcdf(sdir + sname + ".nc")
    rsk.close()
    print("File exported to: %s"%sdir)

def RBR_general_processing(fpath,fname,sdir,sname,patm_dbar=10.1325,indmin=[],indmax=[]):
    ds=xr.open_dataset(fpath+fname+".nc")
    varnames = list(ds.variables)
    pdbar = ds.pdbar.values
    t64 = ds.time.values
    temp = ds.temp.values
    lat = ds.lat.values
    lon = ds.lon.values
    if op.contains(varnames,"salt"):
        salt = ds.salt.values
        cond = ds.cond.values

    dep = dbar2m(pdbar-patm_dbar,ds.lat.values)

    if op.contains(varnames,"salt"):

        data_vars=dict(pdbar=(["time"], pdbar),dep=(["time"], dep),temp=(["time"], temp),cond=(["time"], cond),salt=(["time"], salt))
        coords=dict(time=t64,lat = lat,lon =lon,)
        attrs=dict(description=("Clipped Data from Palau Wave Gauge: %s"%fname), location = ("lat: %.03f, lon: %.03f"%(ds.lat.values,ds.lon.values)))

        ds = xr.Dataset(data_vars=data_vars,coords=coords,attrs=attrs)

        ds["pdbar"].attrs = {'units':'dbar', 'long_name':'total pressure'}
        ds["temp"].attrs = {'units':'deg C', 'long_name':'temperature'}
        ds["cond"].attrs = {'units':'mS/cm', 'long_name':'conductivity'}
        ds["salt"].attrs = {'units':'PSU', 'long_name':'salinity'}
        ds["dep"].attrs = {'units':'m', 'long_name':'water level'}

    else:
        data_vars=dict(pdbar=(["time"], pdbar),dep=(["time"], dep),temp=(["time"], temp))

        coords=dict(time=t64,lat = lat,lon =lon,)
        attrs=dict(description=("Clipped Data from Palau Wave Gauge: %s"%fname), location = ("lat: %.03f, lon: %.03f"%(ds.lat.values,ds.lon.values)))
        ds = xr.Dataset(data_vars=data_vars,coords=coords,attrs=attrs)
        ds["pdbar"].attrs = {'units':'dbar', 'long_name':'total pressure'}
        ds["temp"].attrs = {'units':'deg C', 'long_name':'temperature'}
        ds["dep"].attrs = {'units':'m', 'long_name':'water level'}


    if not np.any(indmax) and not np.any(indmin):
        fig,ax = plt.subplots()
        ax.plot(dep,".-")

        indmin = int(input("What should the first (inclusive) index be?: "))
        indmax = int(input("What should the last (exclusive) index be? if none say 0: "))

    if indmax == 0:
        indmax = len(ds.time.values)
    ds=ds.isel(time=slice(indmin,indmax))
    ds.to_netcdf(sdir + sname + ".nc")
    print ("New file: %s "%(sdir + sname + ".nc"))


def HOBO_general_processing(fpath,fname,sdir,sname,patm_dbar=10.1325,indmin=[],indmax=[]):
    ds=xr.open_dataset(fpath+fname+".nc")
    varnames = list(ds.variables)
    pdbar = ds.pdbar.values
    t64 = ds.time.values
    lat = ds.lat.values
    lon = ds.lon.values
    if op.contains(varnames,"salt"):
        salt = ds.salt.values
        cond = ds.cond.values

    dep = dbar2m(pdbar-patm_dbar,ds.lat.values)

    if op.contains(varnames,"salt"):

        data_vars=dict(pdbar=(["time"], pdbar),dep=(["time"], dep),temp=(["time"], temp),cond=(["time"], cond),salt=(["time"], salt))
        coords=dict(time=t64,lat = lat,lon =lon,)
        attrs=dict(description=("Clipped Data from %s"%fname), location = ("lat: %.03f, lon: %.03f"%(ds.lat.values,ds.lon.values)))

        ds = xr.Dataset(data_vars=data_vars,coords=coords,attrs=attrs)

        ds["pdbar"].attrs = {'units':'dbar', 'long_name':'total pressure'}
        ds["temp"].attrs = {'units':'deg C', 'long_name':'temperature'}
        ds["cond"].attrs = {'units':'mS/cm', 'long_name':'conductivity'}
        ds["salt"].attrs = {'units':'PSU', 'long_name':'salinity'}
        ds["dep"].attrs = {'units':'m', 'long_name':'water level'}

    elif op.contains(varnames,"temp"):
        data_vars=dict(pdbar=(["time"], pdbar),dep=(["time"], dep),temp=(["time"], temp))
        coords=dict(time=t64,lat = lat,lon =lon,)
        attrs=dict(description=("Clipped Data from %s"%fname), location = ("lat: %.03f, lon: %.03f"%(ds.lat.values,ds.lon.values)))
        ds = xr.Dataset(data_vars=data_vars,coords=coords,attrs=attrs)
        ds["pdbar"].attrs = {'units':'dbar', 'long_name':'total pressure'}
        ds["temp"].attrs = {'units':'deg C', 'long_name':'temperature'}
        ds["dep"].attrs = {'units':'m', 'long_name':'water level'}

    else:
        data_vars=dict(pdbar=(["time"], pdbar),dep=(["time"], dep))
        coords=dict(time=t64,lat = lat,lon =lon,)
        attrs=dict(description=("Clipped Data from %s"%fname), location = ("lat: %.03f, lon: %.03f"%(ds.lat.values,ds.lon.values)))
        ds = xr.Dataset(data_vars=data_vars,coords=coords,attrs=attrs)
        ds["pdbar"].attrs = {'units':'dbar', 'long_name':'total pressure'}
        ds["dep"].attrs = {'units':'m', 'long_name':'water level'}




    if not np.any(indmax) and not np.any(indmin):
        fig,ax = plt.subplots()
        ax.plot(dep,".-")

        indmin = int(input("What should the first (inclusive) index be?: "))
        indmax = int(input("What should the last (exclusive) index be? if none say 0: "))

    if indmax == 0:
        indmax = len(ds.time.values)
    ds=ds.isel(time=slice(indmin,indmax))
    ds.to_netcdf(sdir + sname + ".nc")
    print ("New file: %s "%(sdir + sname + ".nc"))
