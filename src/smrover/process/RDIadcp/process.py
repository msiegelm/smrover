# Acoustic Doppler Current Profiler (ADCP) module
import os
import numpy as np
import pickle
import matplotlib.dates as dates
import xarray as xr
import sys 

def mattime2pytime(tmat):
    """
    Converts matlab datenum time stamp to python datenum and np.datetime64
    Input:
    -----
    tmat = time in matlab datenum

    Output:
    ------
    tpy  = time in python datenum
    t64  = time in python np.datetime64
    """
    tpy = tmat-719529
    t64 = num2dt64(tpy)
    return tpy,t64

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

def movingaverage(interval, window_size):
    """
    From wgpack helperfun.py
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def nan_interpolate(x, n):
    '''
    From wgpack helperfun.py

    This function linearly interpolates over NaNs, if the number of contiguous NaNs is less than n.
    :param x: data array
    :param n: maximum NaN gap over which interpolation is permitted.
    :return: de-NaN'ed array
    '''
    import pandas as pd
    return pd.Series(x).interpolate(method='linear', limit=n).values

def motion_correct_ADCP_gps(adcpr, dt_gps, mag_dec=None, qc_flg=False, dtc=None, three_beam_flg=None):
    '''
    This function corrects ADCP velocities for Wave Glider motion using GPS-derived velocities.
    Reads-in output from rdrdadcp.py (python)
    :param adcpr: output structure from rdradcp.py
    :param dt_gps: Motion correction time-averaging interval for GPS-derived velocities (s)
    :param mag_dec: magnetic delcination [deg]. Overrides the EB setting in the config file (necessary when EB is
                    configured incorrectly)
    :param qc_flg: Along-beam velocities will be rejected for instrument tilts greater than a tilt threshold
                    (e.g., 20 deg)
    :param dtc: time offset correction as numpy.timedelta64
    :param three_beam_flg: if not None use a three-beam solution. The value (1,2,3,4) corresponds to the
                           beam number that is excluded from the velocity solution
    :return: dictionary containing motion-corrected ADCP velocities and auxiliary variables
    References:
    https://github.com/rustychris/stompy/blob/master/stompy/io/rdradcp.py
    Other resources:
    https://seachest.ucsd.edu/cordc/analysis/rdradcpy
    https://pypi.org/project/ADCPy/
    '''
    import datetime
    import numpy as np
    import pandas as pd
    from geopy.distance import distance
    # local imports
    # Collect variables:
    if type(adcpr)==dict:
        # time (matlab datenum format)
        mtime = adcpr['mtime']
        # nav variables
        pitch = adcpr['pitch']
        roll = adcpr['roll']
        heading = adcpr['heading']
        nav_elongitude = adcpr['nav_elongitude']
        nav_elatitude = adcpr['nav_elatitude']
        # Bottom tracking
        bt_range = adcpr['bt_range'].T
        # Doppler velocities (beam)
        b1_vel = adcpr['east_vel'].T
        b2_vel = adcpr['north_vel'].T
        b3_vel = adcpr['vert_vel'].T
        b4_vel = adcpr['error_vel'].T
        # QC variables
        perc_good = adcpr['perc_good'].T
        corr = adcpr['corr'].T
        intens = adcpr['intens'].T
        # Temperature
        temperature = adcpr['temperature']
        # config variables
        ranges = adcpr['config']['ranges']
        beam_angle = adcpr['config']['beam_angle']
        EA = adcpr['config']['xducer_misalign']
        EB = adcpr['config']['magnetic_var']
        xmit_pulse = adcpr['config']['xmit_pulse']
        xmit_lag = adcpr['config']['xmit_lag']
    else:
        # time (matlab datenum format)
        mtime = adcpr.nav_mtime
        # nav variables
        pitch = adcpr.pitch
        roll = adcpr.roll
        heading = adcpr.heading
        nav_elongitude = adcpr.nav_elongitude
        nav_elatitude = adcpr.nav_elatitude
        # Bottom tracking
        bt_range = adcpr.bt_range
        # Doppler velocities (beam)
        b1_vel = adcpr.east_vel
        b2_vel = adcpr.north_vel
        b3_vel = adcpr.vert_vel
        b4_vel = adcpr.error_vel
        # QC variables
        perc_good = adcpr.perc_good
        corr = adcpr.corr
        intens = adcpr.intens
        # Temperature
        temperature = adcpr.temperature
        # config variables
        ranges = adcpr.config.ranges
        beam_angle = adcpr.config.beam_angle
        EA = adcpr.config.xducer_misalign
        EB = adcpr.config.magnetic_var
        xmit_pulse = adcpr.config.xmit_pulse
        xmit_lag = adcpr.config.xmit_lag

    # convert matlab datenum to datetime
    nav_time = []
    for mt in mtime:
        nav_time.append(datetime.datetime.fromordinal(int(mt))
                        + datetime.timedelta(days=mt % 1)
                        - datetime.timedelta(days=366))

    # convert to pandas datetime
    nav_time = pd.to_datetime(nav_time)
    # correct time offset
    if dtc is None:
        pass
    else:
        # correct time offset
        nav_time = nav_time + dtc

    # bottom-tracking
    bt_range_mean = np.mean(bt_range, axis=1)
    # raw echo intensities
    b1_intens = intens[:, :, 0]
    b2_intens = intens[:, :, 1]
    b3_intens = intens[:, :, 2]
    b4_intens = intens[:, :, 3]

    # Magnetic declination correction (necessary when EB is configured incorrectly)
    if mag_dec is None:
        pass
    else:
        # override EB with user specified magnetic declination
        heading = (heading - EB + mag_dec) % 360

    # ------------------------------------------------------------
    # Process GPS data
    # ------------------------------------------------------------
    # ping to ping dt
    dt_p2p = np.array(np.median(np.diff(nav_time)), dtype='timedelta64[s]').item().total_seconds()
    # number of points (full-step)
    wz_gps = int(dt_gps / dt_p2p)
    # number of points (half-step)
    nn = int(wz_gps / 2)
    # calculate GPS based velocities
    sog_gps, sog_gpse, sog_gpsn, cog_gps, pitch_mean, roll_mean = [], [], [], [], [], []
    for i, t in enumerate(nav_time[nn:-nn]):
        # apply central differencing scheme
        ii = i + nn
        # dt in seconds
        # dt = np.array(nav_time[ii+nn]-nav_time[ii-nn], dtype='timedelta64[s]').item().total_seconds()
        # this method is probably faster
        dt = (nav_time[ii + nn] - nav_time[ii - nn]).value / 1E9
        # Calculate cog and sog from WG mwb coordinates
        p1 = (nav_elatitude[ii - nn], nav_elongitude[ii - nn])
        p2 = (nav_elatitude[ii + nn], nav_elongitude[ii + nn])
        sog = distance(p1, p2).m / dt
        cog = get_bearing(p1, p2)
        # store values
        cog_gps.append(cog)
        sog_gps.append(sog)
        sog_gpse.append(sog * np.sin(np.deg2rad(cog)))
        sog_gpsn.append(sog * np.cos(np.deg2rad(cog)))
        pitch_mean.append(np.mean(pitch[ii - nn:ii + nn]))
        roll_mean.append(np.mean(roll[ii - nn:ii + nn]))

    # concatenate nans
    app_nan = np.zeros(nn) + np.nan
    cog_gps = np.concatenate((app_nan, np.array(cog_gps), app_nan))
    sog_gps = np.concatenate((app_nan, np.array(sog_gps), app_nan))
    sog_gpse = np.concatenate((app_nan, np.array(sog_gpse), app_nan))
    sog_gpsn = np.concatenate((app_nan, np.array(sog_gpsn), app_nan))
    # # ------------------------------------------------------------
    # # low-pass filter raw ADCP heading
    # huf = movingaverage(np.sin(np.deg2rad(heading)), window_size=wz_gps)
    # hvf = movingaverage(np.cos(np.deg2rad(heading)), window_size=wz_gps)
    # headingf = np.rad2deg(np.arctan2(huf, hvf))
    # compute float heading (corrected for heading offset (EA)
    heading_float = (heading - EA) % 360

    # ------------------------------------------------------------
    # Q/C ADCP velocities
    # ------------------------------------------------------------
    # Note that percent-good (perc_good < 100) already masks velocity data with nans
    if qc_flg:
        pass

    # ------------------------------------------------------------
    # Process ADCP velocities
    # ------------------------------------------------------------
    # Beam to Instrument
    # Constants
    c = 1
    theta = np.deg2rad(beam_angle)  # beam angle
    a = 1 / (2 * np.sin(theta))
    b = 1 / (4 * np.cos(theta))
    d = a / np.sqrt(2)

    if three_beam_flg is not None:
        # then use 3-beam solution
        if three_beam_flg == 1:
            b1_vel = -b2_vel + b3_vel + b4_vel
        elif three_beam_flg == 2:
            b2_vel = -b1_vel + b3_vel + b4_vel
        elif three_beam_flg == 3:
            b3_vel = b1_vel + b2_vel - b4_vel
        elif three_beam_flg == 4:
            b4_vel = b1_vel + b2_vel - b3_vel

    # compute instrument velocities
    x_vel = (c * a * (b1_vel - b2_vel)).T
    y_vel = (c * a * (b4_vel - b3_vel)).T
    z_vel = (b * (b1_vel + b2_vel + b3_vel + b4_vel)).T
    err_vel = (d * (b1_vel + b2_vel - b3_vel - b4_vel)).T

    # ------------------------------------------------------------
    # Instrument to Ship
    h = -EA
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel
    w = (-cp * sr) * x_vel + (sp) * y_vel + (cp * cr) * z_vel
    # TODO: calculate and output velocities in vehicle reference frame
    # ------------------------------------------------------------
    # Instrument to Earth
    h = heading
    Tilt1 = np.deg2rad(pitch)
    Tilt2 = np.deg2rad(roll)
    P = np.arctan(np.tan(Tilt1) * np.cos(Tilt2))
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    # cp = np.cos(P)
    # sp = np.sin(P)
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel
    # Correct ADCP velocities (gps-derived velocities)
    Evel = u + sog_gpse
    Nvel = v + sog_gpsn
    # ------------------------------------------------------------
    # Create output dictionary for motion-corrected ADCP data
    adcpmdict = {
        'time': nav_time,
        'longitude': nav_elongitude,
        'latitude': nav_elatitude,
        'ranges': ranges,
        'Evel': Evel,
        'Nvel': Nvel,
        'err_vel': err_vel,
        'cog_gps': cog_gps,
        'sog_gps': sog_gps,
        'sog_gpse': sog_gpse,
        'sog_gpsn': sog_gpsn,
        'heading_float': heading_float,
        'temperature': temperature,
        }
    return adcpmdict

def get_bearing(p1,p2):
    '''
    This function returns the bearing from initial point to destination point.
    References:
    http://www.movable-type.co.uk/scripts/latlong.html
    :param p1: (lat1,lon1) point - Latitude/longitude of initial point
    :param p2: (lat2,lon2) point - Latitude/longitude of destination point
    :return: Bearing in degrees from north (0°..360°)
    '''
    # extract coordinates
    lat1,lon1 = p1[0],p1[1]
    lat2,lon2 = p2[0],p2[1]
    # convert to radians
    lat1,lat2 = lat1*np.pi/180, lat2*np.pi/180
    dlon = (lon2-lon1)*np.pi/180
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    y = np.sin(dlon) * np.cos(lat2)
    return (np.arctan2(y, x)*180/np.pi)%360

def rdradcp_output_to_dictionary(adcpr):
    '''
    This function convert rdradcp output to a dictionary. This allows straightforward storage of the data in various
    formats
    :param adcpr: rdradcp output
    :return: rdradcp output organized as a dictionary structure
    '''
    from matplotlib.dates import date2num, num2date

    # convert to matlab time 
    mtime = []
    for mtime_r in adcpr.mtime:
        dt = num2date(mtime_r).replace(tzinfo=None)
        mtime.append(datetime2matlabdn(dt))
    mtime = np.array(mtime)
    tnum,time = mattime2pytime(mtime)
    # mtime = np.array(adcpr.mtime) #mns
    # Convert config to dictionary
    config_dict = {
        'beam_angle': adcpr.config.beam_angle,
        'beam_freq': adcpr.config.beam_freq,
        'beam_pattern': adcpr.config.beam_pattern,
        'bin1_dist': adcpr.config.bin1_dist,
        'bin_mapping': adcpr.config.bin_mapping,
        'blank': adcpr.config.blank,
        'cell_size': adcpr.config.cell_size,
        'config': adcpr.config.config,
        'coord': adcpr.config.coord,
        'coord_sys': adcpr.config.coord_sys,
        'corr_threshold': adcpr.config.corr_threshold,
        'evel_threshold': adcpr.config.evel_threshold,
        'fls_target_threshold': adcpr.config.fls_target_threshold,
        # 'h_adcp_beam_angle': adcpr.config.h_adcp_beam_angle,
        'magnetic_var': adcpr.config.magnetic_var,
        'min_pgood': adcpr.config.min_pgood,
        'n_beams': adcpr.config.n_beams,
        'n_cells': adcpr.config.n_cells,
        'n_codereps': adcpr.config.n_codereps,
        'name': adcpr.config.name,
        # 'navigator_basefreqindex': adcpr.config.navigator_basefreqindex,
        'numbeams': adcpr.config.numbeams,
        'orientation': adcpr.config.orientation,
        'pings_per_ensemble': adcpr.config.pings_per_ensemble,
        'prof_mode': adcpr.config.prof_mode,
        'prog_ver': adcpr.config.prog_ver,
        'ranges': adcpr.config.ranges,
        'sensors_avail': adcpr.config.sensors_avail,
        'sensors_src': adcpr.config.sensors_src,
        # 'serialnum': adcpr.config.serialnum,
        'simflag': adcpr.config.simflag,
        'sourceprog': adcpr.config.sourceprog,
        'time_between_ping_groups': adcpr.config.time_between_ping_groups,
        'use_3beam': adcpr.config.use_3beam,
        'use_pitchroll': adcpr.config.use_pitchroll,
        'water_ref_cells': adcpr.config.water_ref_cells,
        'xducer_misalign': adcpr.config.xducer_misalign,
        'xmit_lag': adcpr.config.xmit_lag,
        'xmit_pulse': adcpr.config.xmit_pulse,
    }

    # Convert adcpr to dictionary
    adcpr_dict = {
        "name": adcpr.name,
        "config": config_dict,
        "mtime": mtime,
        "number": adcpr.number,
        "pitch": adcpr.pitch,
        "roll": adcpr.roll,
        "heading": adcpr.heading,
        "pitch_std": adcpr.pitch_std,
        "roll_std": adcpr.roll_std,
        "heading_std": adcpr.heading_std,
        "depth": adcpr.depth,
        "temperature": adcpr.temperature,
        "salinity": adcpr.salinity,
        "pressure": adcpr.pressure,
        "pressure_std": adcpr.pressure_std,
        "east_vel": adcpr.east_vel.T,
        "north_vel": adcpr.north_vel.T,
        "vert_vel": adcpr.vert_vel.T,
        "error_vel": adcpr.error_vel.T,
        "corr": adcpr.corr.T,
        "status": adcpr.status.T,
        "intens": adcpr.intens.T,
        "bt_range": adcpr.bt_range.T,
        "bt_vel": adcpr.bt_vel.T,
        "bt_corr": adcpr.bt_corr.T,
        "bt_ampl": adcpr.bt_ampl.T,
        "bt_perc_good": adcpr.bt_perc_good.T,
        "perc_good": adcpr.perc_good.T,
        "time": time
    }
    return adcpr_dict

def datetime2matlabdn(dt):
    # This function converts Python datetime to Matlab datenum
    # References: https://stackoverflow.com/questions/8776414/python-datetime-to-matlab-datenum
    import datetime
    mdn = dt + datetime.timedelta(days = 366)
    frac_seconds = (dt-datetime.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds

def writeBin2Pkl(fpath,fname,svpath,svname,num_av=1):
    from smrover.process.RDIadcp.rdradcp import rdradcp #add rdradcp to path

    dat = rdradcp(fpath+fname,num_av=num_av) #read in ADCP data
    datd = rdradcp_output_to_dictionary(dat) #convert to dictionary for saving
    with open(svpath+svname + '.pkl', 'wb') as outfile:
        pickle.dump(datd, outfile)

    print('File saved to: '+svpath+svname+'.pkl')


def load_pickle(fpath,fname):
    """
    Read in pickle file and return data.
    Parameters:
    -----------
    fpath : str
        File path to pickle file.   
    fname : str
        Name of pickle file.
    
    Returns:
    --------
    data : object
        Data contained in pickle file.
    """
    with open(fpath+fname, "rb") as f:   # "rb" = read binary
        data = pickle.load(f)
    return data


def adcp_dict2nc(dat,processedby=[],lat=[],lon=[],saveas=False,svpath=[],svname=[]):
    """ 
    Function to convert ADCP dictionary to xarray dataset. 
    Parameters:
    -----------
    dat : dict
        Dictionary containing ADCP data.
    lat : float, optional
        Latitude of deployment location. Default is [].
    lon : float, optional
        Longitude of deployment location. Default is [].
    saveas : bool, optional
        If True, save dataset as netcdf file. Default is False.
    svpath : str, optional
        File path to save netcdf file. Default is [].
    svname : str, optional
        Name of netcdf file. Default is []. 
    
    Returns:
    --------
    ds : xarray.Dataset
        Xarray dataset containing ADCP data.
    Notes:
    ------
    - If saveas is True, svpath and svname must be provided.
    
    """

    # data variables
    data_vars = {
        'Evel'          : (['ranges', 'time'], dat['east_vel'], {'units': 'm/s'}),
        'Nvel'          : (['ranges', 'time'], dat['north_vel'], {'units': 'm/s'}),
        'Wvel'          : (['ranges', 'time'], dat['vert_vel'], {'units': 'm/s'}),
        'err_vel'       : (['ranges', 'time'], dat['error_vel'], {'units': 'm/s'}),
        'corr'       : (['beams','ranges', 'time'], dat['corr'], {'units': 'N/A'},{'long_name': 'Correlation: Range 0 - 255'}),
        'intens'       : (['beams','ranges', 'time'], dat['intens'], {'units': 'Counts'},{'long_name': 'Echo Intensity'}),
        'perc_good'       : (['beams','ranges', 'time'], dat['perc_good'], {'units': 'Percent'}),
        'pressure'       : (['time'], dat['pressure'], {'units': 'daPa'}),
        'depth'       : (['time'], dat['depth'], {'units': 'm'}),
        'roll'      : (['time'], dat['roll'], {'units': 'degrees'}),
        'pitch'      : (['time'], dat['pitch'], {'units': 'degrees'}),
        'heading' : (['time'], dat['heading'], {'units': 'degrees'}),
        'temperature'    : (['time'], dat['temperature'], {'units': 'degrees_C'}),
    }
    if np.any(lat):
        coords = {
            'time'          : ('time', dat['time']),
            'ranges'        : ('ranges', dat['config']['ranges'], {'units': 'm'},{'long_name': 'Distance from instrument head'}),
            'beams'        : ('beams', np.arange(1, dat['config']['numbeams'] + 1), {'units': 'count'},{'long_name': 'Beam number'}),
            'latitude'      : ((), lat, {'units': 'degrees_north'}),
            'longitude'     : ((), lon, {'units': 'degrees_east'}),
        }
    else:
    # coordinates
        coords = {
            'time'          : ('time', dat['time']),
            'ranges'        : ('ranges', dat['config']['ranges'], {'units': 'm'},{'long_name': 'Distance from instrument head'}),
            'beams'        : ('beams', np.arange(1, dat['config']['numbeams'] + 1), {'units': 'count'},{'long_name': 'Beam number'}),
        }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    attrs=dict(description=("Raw Selected Data from Seacliff ADCP."), location = ("lat: %.03f, lon: %.03f"%(lat,lon)), processedby = ("%s"%processedby))
    ds.attrs = attrs
    if saveas:

        ds.to_netcdf(svpath + svname + ".nc")
        print("File exported to: %s"%(svpath + svname + ".nc"))
    else:
        return ds
