import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u

from dust_extinction.averages import G03_SMCBar, G24_SMCAvg

if __name__ == "__main__":

    fontsize = 14

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(10.0, 12.0))


    taus = [0.5, 1.5, 4.5]
    geoms = ["dusty", "shell", "cloudy"]
    cols = ["r", "g", "b"]
    lgeoms = ["h", "c"]
    lines = ["-", "--"]
    dusttype = "fA0.00"
    for i, ctau in enumerate(taus):
        ax = axes[i, 0]
        for cgeom, ccol in zip(geoms, cols):
            for clgeom, cline in zip(lgeoms, lines):
                fname = f"sge2026_{cgeom}_{clgeom}_{dusttype}_tau{ctau:.4f}_global_lum.table.fits"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=u.UnitsWarning)
                    dtab = Table.read(f"{cgeom}/{fname}")
                wave = 1.0 / dtab["wavelength"].value
                tauatt = -1.0 * np.log(dtab["Flux"] / dtab["Flux_Input"])
                tauatt_direct = -1.0 * np.log(dtab["Flux_rt_d"] / dtab["Flux_Input"])
                tauatt_scat = -1.0 * np.log(dtab["Flux_rt_s"] / dtab["Flux_Input"])
                ax.plot(wave, tauatt, color=ccol, linestyle=cline, 
                        label=rf"{cgeom} ({clgeom},{dusttype}) ($\tau_V$ = {ctau:.2f})")

                # normalized at V band
                tauV = np.interp([0.55], 1.0 / wave, tauatt)
                axes[i, 1].plot(wave, tauatt / tauV[0], color=ccol, linestyle=cline, 
                              label=rf"{cgeom} ({clgeom},{dusttype}) ($\tau_V$ = {ctau:.2f})")                

        ax.plot(wave, dtab["tau_norm"] * ctau, "k-", label="Extinction curve")
        axes[i, 1].plot(wave, dtab["tau_norm"], "k-", label="Extinction curve")

        ax.set_xlabel(r"$\lambda$ [$\mu$m]")
        axes[i, 1].set_xlabel(r"$\lambda$ [$\mu$m]")
        ax.set_ylabel(r"$\tau_\mathrm{att}$")
        axes[i, 1].set_ylabel(r"$\tau_\mathrm{att}/\tau_\mathrm{V}$")

        ax.set_ylim(0.0, 7.5)
        ax.set_xlim(0.0, 10.0)
        axes[i, 1].set_ylim(0.0, 7.5)
        axes[i, 1].set_xlim(0.0, 10.0)

        #ax.set_xscale("log")
        #ax.set_yscale("log")

        axes[i, 1].legend(fontsize=0.5*fontsize)

    smc1 = G03_SMCBar()
    axes[2, 1].plot(smc1.obsdata_x, smc1.obsdata_axav, "ro")

    smc2 = G24_SMCAvg()
    axes[2, 1].plot(smc2.obsdata_x, smc2.obsdata_axav, "gs")

    plt.tight_layout()

    plt.show()
