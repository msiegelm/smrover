import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates

def plot_spectrogram(sg,clim,ylim=[],tlim=[],title=[],saveas=False,figpath=[],sname=[]):
    dtt = np.diff(dates.date2num(sg["t64"]))
    kk = np.where((dtt > 30))[0]
    (nf,nt) = np.shape(sg["Edens"])
    if np.any(kk):
        oo = np.ones((nf,2))*np.nan
        tt = dates.date2num(sg["t64"])
        ttdc = dates.date2num(sg["t64"])
        dts = tt[1]-tt[0]
        psdf = sg["Edens"].copy()
        psdfdc = sg["Edens"].copy()
        if len(kk) == 1:
            ind = kk[0]
            tmn1 = ttdc[ind] + dts
            tmn2 = ttdc[ind+1] - dts
            tt = np.hstack((ttdc[:ind+1],tmn1,tmn2,ttdc[ind+1:]))
            psdf = np.hstack((psdfdc[:,:ind+1],oo,psdfdc[:,ind+1:]))

        else:
            for jk in range(len(kk)):
                ind = kk[jk]
                tmn1 = ttdc[ind] + dts
                tmn2 = ttdc[ind+1] - dts
                if jk == len(kk)-1:
                    psdf1 = psdf
                    tt1 = tt
                    tt = np.hstack((tt1,tmn1,tmn2,ttdc[ind+1:]))
                    psdf = np.hstack((psdf1,oo,psdfdc[:,ind+1:]))

                else:
                    if jk == 0:
                        psdf1 = psdf[:,:ind+1]
                        tt1 = tt[:ind+1]
                    else:
                        psdf1 = psdf
                        tt1 = tt
                    tt = np.hstack((tt1,tmn1,tmn2,ttdc[ind+1:kk[jk+1]]))
                    psdf = np.hstack((psdf1,oo,psdfdc[:,ind+1:kk[jk+1]]))
    else:
        tt = dates.date2num(sg["t64"])
        psdf = sg["Edens"].copy()

    # xmg,ymg = np.meshgrid(tt,sg.freq)
    fig,ax=plt.subplots(figsize=(13,4))
    pc=ax.pcolormesh(dates.num2date(tt),np.log10(sg["freq"]),np.log10(psdf),vmin=clim[0],vmax=clim[-1],cmap="jet")

    if np.any(ylim):
        ax.set_ylim([ylim[0],ylim[-1]])
    ax.set_ylabel("Frequency [$log_{10}$(Hz)]",fontsize=16)
    ax.tick_params(labelsize=16)
    ax.tick_params("x",rotation=90)
    cb = fig.colorbar(pc)
    cb.ax.tick_params(labelsize=16)
    cb.set_label("PSD [log$_{10}$($m^2Hz^{-1}$)]",fontsize=16)
    ax.set_title(title,fontsize=16)
    ax.xaxis.set_major_locator(dates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(dates.DayLocator(interval=1))
    # ax.plot(dates.num2date(tt),np.ones(len(tt))*np.log10(1/5))
    # ax.plot(dates.num2date(tt),np.ones(len(tt))*np.log10(1/10))
    # ax.plot(dates.num2date(tt),np.ones(len(tt))*np.log10(1/20))

    if np.any(tlim):
        ax.set_xlim([tlim[0],tlim[1]])
    if saveas:
        fig.savefig(figpath + sname + '.png', dpi=300, bbox_inches='tight')

    return fig, ax
