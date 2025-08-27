import numpy as np
from scipy import signal
import matplotlib.dates as dates
from .tools import num2dt64, Bunch

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


##################################
def dispr(h,freq):
    """
    Dispersion relationship solver. Based on dispr.m by C.S. Wu.

    Input:
    ------
    h = depth [m]
    f = frequency [Hz]

    Output:
    ------
    kh = wave number * depth
    k = wave number [rad m^-1]
    """
    if isinstance(freq,(list, tuple, np.ndarray)) and isinstance(h,(list, tuple, np.ndarray)):
        lf = len(freq)
        lh =len(h)

        kh = np.nan * np.ones((lf,lh))
        omega = 2*np.pi*freq
        a = h[np.newaxis,:]*(omega[:,np.newaxis]**2)/9.81 #omega^2h/g = khtanhkh
        b = 1 + 1.26*np.exp(-1.84 * a[a >= 1]) #deep water
        t = np.exp(-2 * a[a >= 1] * b)
        akk1 = 1 + (2 * t * (1 + t))
        kh[a >= 1] = a[a >= 1] * akk1 #deep water kh
        c = 1 + (a[a < 1]/6) * (1 + a[a < 1]/5)
        kh[a < 1] = np.sqrt(a[a < 1]) * c
        k = kh/h

    elif isinstance(freq,(list, tuple, np.ndarray)) and not isinstance(h,(list, tuple, np.ndarray)):
        kh = np.nan * np.ones_like(freq)
        omega = 2*np.pi*freq
        a = h*(omega**2)/9.81 #omega^2h/g = khtanhkh
        b = 1 + 1.26*np.exp(-1.84 * a[a >= 1]) #deep water
        t = np.exp(-2 * a[a >= 1] * b)
        akk1 = 1 + (2 * t * (1 + t))
        kh[a >= 1] = a[a >= 1] * akk1 #deep water kh
        c = 1 + (a[a < 1]/6) * (1 + a[a < 1]/5)
        kh[a < 1] = np.sqrt(a[a < 1]) * c
        k = kh/h


    elif not isinstance(freq,(list, tuple, np.ndarray)) and not isinstance(h,(list, tuple, np.ndarray)):
        kh = np.nan 
        omega = 2*np.pi*freq
        a = h*(omega**2)/9.81 #omega^2h/g = khtanhkh

        if a >= 1:
            #deepwater
            b = 1 + 1.26*np.exp(-1.84 * a) #deep water
            t = np.exp(-2 * a * b)
            akk1 = 1 + (2 * t * (1 + t))
            kh = a * akk1 #deep water kh
        else:
            c = 1 + (a/6) * (1 + a/5)
            kh = np.sqrt(a) * c
        k = kh/h

    # elif not isinstance(freq,(list, tuple, np.ndarray)) and not isinstance(h,(list, tuple, np.ndarray)):
    #     kh = np.nan * np.ones_like(freq)
    #     omega = 2*np.pi*freq
    #     a = h*(omega**2)/9.81 #omega^2h/g = khtanhkh
    #     b = 1 + 1.26*np.exp(-1.84 * a[a >= 1]) #deep water
    #     t = np.exp(-2 * a[a >= 1] * b)
    #     akk1 = 1 + (2 * t * (1 + t))
    #     kh[a >= 1] = a[a >= 1] * akk1 #deep water kh
    #     c = 1 + (a[a < 1]/6) * (1 + a[a < 1]/5)
    #     kh[a < 1] = np.sqrt(a[a < 1]) * c
    #     k = kh/h

    


    return kh,k

def wave_energy_density(dep,dt,nperseg=None,noverlap=0,nsmooth=7,nfft=None):
    """
    Estimates 1-D Wave Energy Density Spectrum from pressure time series.

    Boxcar smoothing is used to increase confidence in estimate.
    Uses Welch's method to compute estimate of PSD.  By default, segment averaging turned off (nperseg = None).

    Parameters:
    ----------
    dep : array
        Time series of depth (m)
    dt : float
        Time step (s)
    nperseg : int or None
        Length of each segment for Welch's method (default: None, which uses entire length of dep)
    noverlap : int
        Number of points to overlap between segments (default: 0)
    nsmooth : int or None
        Length of boxcar smoothing window (default: 7). Must be odd integer or None (no smoothing).
    nfft : int or None
        Length of FFT (default: None, which uses nperseg)
    
    Returns:
    -------
    out : Dict
        Dict object with the following fields:
        freq : array
            Frequencies (Hz)
        Spp : array
            Pressure spectrum (m^2/Hz)
        Edens : array
            Wave energy density spectrum (m^2/Hz)
    """
    if nsmooth is not None and nsmooth % 2 != 1:
        raise ValueError("nsmooth must be None or an odd integer")
    fs = 1/dt
    if nperseg == None:
        nperseg = len(dep)
    t_det = signal.detrend(dep)
    ff, Spp = signal.welch(t_det, fs=fs, nperseg=nperseg, noverlap=noverlap,return_onesided=True,nfft=nfft)
    ff = ff[1:]
    Spp = Spp[1:]
    if nsmooth is not None:
        window = np.ones((nsmooth),dtype=float)/float(nsmooth)
        Spp = np.convolve(Spp,window,mode='valid')
        ff = np.convolve(ff,window,mode='valid')

    h = np.nanmean(dep)
    kh, kk = dispr(h, ff)
    Kpp = np.cosh(kh)
    Kpp2 = Kpp**2
    ###### does not amplify anything over cthresh to to prent from disproportionately amplifying noise floor at high freqs
    cthresh = 20
    Kpp2[Kpp2 > cthresh] = cthresh
    Edens = Spp*Kpp2
    out = {}
    out['freq'] = ff
    out['Spp'] = Spp
    out['Edens'] = Edens
    return out

def calc_Hs(dep,dt,Tmin,Tmax,calcTp=False,cthresh=20):
    """
    Calculates significant wave height from bottom pressure time series.
    
    """
    if dep.ndim == 1:
        ns = len(dep)
        h = np.nanmean(dep)
    else:
        (ns,nb) = np.shape(dep)
        h = np.nanmean(dep,axis=0)
    np2 = int(np.floor(ns/2))
    freq = np.arange(1,int(np2+1))/(ns*dt)

    kh, kk = dispr(h, freq)
    Kpp = np.cosh(kh)
    m = (freq>=1/Tmin)
    m2 = (freq <= 1/Tmax)
    if dep.ndim == 1:
        Kpp[m]=0
        Kpp[m2]=0
    else:
        Kpp[m,:]=0
        Kpp[m2,:]=0     
    Kpp2 = Kpp**2
    Kpp2[Kpp2 > cthresh] = cthresh #does not amplify past cthresh (prevent noise amplification at HF)

    if dep.ndim == 1:
        detdep = signal.detrend(dep)
        fp = np.fft.fft(detdep)/ns
        ran = np.arange(1, int(np2+1))
        fp = fp[ran]
        fp = 2*np.abs(fp)**2 #power spectrum
        fp = fp * Kpp2 #surface displacement from pressure
        Hs = 4*np.sqrt(np.sum(fp)) #calculate significant wave height

    else:
        detdep = signal.detrend(dep,axis=0)
        fp = np.fft.fft(detdep,axis=0)/ns
        ran = np.arange(1, int(np2+1))
        fp = fp[ran,:]
        fp = 2*np.abs(fp)**2 #power spectrum
        fp = fp * Kpp2 #surface displacement from pressure
        Hs = 4*np.sqrt(np.sum(fp,axis=0)) #calculate significant wave height
    if calcTp:
        if dep.ndim == 1:
            im = np.argmax(fp)
            Tp = 1/freq[im]
        else:
            im = np.argmax(fp,axis=0)
            Tp = 1/freq[im]
        return Hs,Tp
    else:
        return Hs

def sig_waveheight(time,dep,dt,TrngSS,TrngIG,TrngSea,TrngSwell,cthresh=20):
    """
    Significant wave height

    Parameters:
    ------
    time = time (datetime64, same shape as time)
    dep = depth [m]
    dt = Sample Rate [sec]
    TrngSS = [SS-band Minimum Period, Max Period]
    TrngIG = [IG-band Minimum Period, Max Period]
    TrngSea = [Sea-band Minimum Period, Max Period]
    TrngSwell = [Swell-band Minimum Period, Max Period]

    cthresh = threshold to prevent overamplification of HF

    Returns:
    ------
    Hs = Sea and Swell Significant wave height [m]
    Hig = Ig Significant wave height [m]
    Tp = Peak Period of SS-band [sec]
    """
    out = Bunch()
    ### Sea and Swell
    TminSS = TrngSS[0]
    TmaxSS = TrngSS[-1]
    Hs,Tp = calc_Hs(dep,dt,TminSS,TmaxSS,calcTp=True,cthresh=cthresh)

    #infragravity significant wave height
    TminIG = TrngIG[0]
    TmaxIG = TrngIG[-1]
    Hig = calc_Hs(dep,dt,TminIG,TmaxIG,calcTp=False,cthresh=cthresh)

    #infragravity sea
    TminSea = TrngSea[0]
    TmaxSea = TrngSea[-1]
    Hsea = calc_Hs(dep,dt,TminSea,TmaxSea,calcTp=False,cthresh=cthresh)

    #infragravity swell
    TminSwell = TrngSwell[0]
    TmaxSwell = TrngSwell[-1]
    Hswl = calc_Hs(dep,dt,TminSwell,TmaxSwell,calcTp=False,cthresh=cthresh)

    out.Hss = Hs
    out.Hig = Hig
    out.Hsea = Hsea
    out.Hswl = Hswl
    out.Tp = Tp
    out.tnum = np.nanmean(dates.date2num(time),axis=0)
    out.t64 = num2dt64(out.tnum)
    return out
