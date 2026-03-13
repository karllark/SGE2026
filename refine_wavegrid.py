import argparse
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u

from dust_extinction.parameter_averages import G23
from dust_extinction.grain_models import WD01

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 14

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    dtab = Table.read("wg00_mw_empir_props.dat",
                      names=["wave", "ext", "albedo", "g"],
                      format="ascii.basic")

    Qext = dtab["ext"]
    Qsca = Qext * dtab["albedo"]
    Qabs = Qext - Qsca
    ax = axes[0, 0]
    ax.plot(dtab["wave"], Qext, "ko", label="WG00 Qext")
    ax.plot(dtab["wave"], Qsca, "bs", label="WG00 Qsca")
    ax.plot(dtab["wave"], Qabs, "gs", label="WG00 Qabs")

    mwext = G23()
    nwave = np.logspace(np.log10(0.0912), np.log10(30.0), num=100) * u.micron
    # add extra points on the 10 micron silicate feature
    x1 = np.max(nwave[nwave < 7 * u.micron])
    x2 = np.min(nwave[nwave > 29.5 * u.micron])
    nwave_uvbump = np.logspace(np.log10(x1.value), np.log10(x2.value), num=50) * u.micron
    gvals1 = nwave < x1
    gvals2 = nwave > x2
    nwave = np.concatenate([nwave[gvals1], nwave_uvbump, nwave[gvals2]])
    # add extra points on the 2175 A bump
    x1 = np.max(nwave[nwave < 0.17 * u.micron])
    x2 = np.min(nwave[nwave > 0.26 * u.micron])
    nwave_uvbump = np.logspace(np.log10(x1.value), np.log10(x2.value), num=20) * u.micron
    gvals1 = nwave < x1
    gvals2 = nwave > x2
    nwave = np.concatenate([nwave[gvals1], nwave_uvbump, nwave[gvals2]])

    ax.plot(nwave, mwext(nwave), "k-", label="G23 Qext", alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda^{-1}$ [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"$Q/Q_\mathrm{ext}(V)$")
    ax.legend(fontsize=0.8*fontsize)

    ax = axes[0, 1]
    ax.plot(dtab["wave"], dtab["albedo"], "ko", label="albedo")
    ax.plot(dtab["wave"], dtab["g"], "gs", label="g")

    ax.set_xscale("log")
    ax.set_xlim(0.0912, 30.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$\lambda^{-1}$ [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"albedo or g")
    ax.legend(fontsize=0.8*fontsize)

    # now the new one
    ax = axes[1, 0]

    Qext = mwext(nwave)
    cs = CubicSpline(dtab["wave"], dtab["albedo"])
    albedo = cs(nwave)
    cs = CubicSpline(dtab["wave"], dtab["g"])
    g = cs(nwave)

    # WD01 grain model
    wd01mod = WD01()
    cs = CubicSpline(1.0 / wd01mod.data_x, wd01mod.data_albedo)
    modalbedo = cs(nwave)
    cs = CubicSpline(1.0 / wd01mod.data_x, wd01mod.data_g)
    modg = cs(nwave)

    # use model beyond 3 micron
    gvals = nwave > 3.0 * u.micron
    albedo[gvals] = modalbedo[gvals]
    # mix albedo between 2-3 micron
    gvals = (nwave > 2.0 * u.micron) & (nwave < 3.0 * u.micron)
    weights = (nwave[gvals] - 2.0 * u.micron) / 1.0 * u.micron
    weights = weights.value
    albedo[gvals] = (1.0 - weights) * albedo[gvals] + weights * modalbedo[gvals]

    # model beyond 3 micron
    gvals = nwave > 3.0 * u.micron
    g[gvals] = modg[gvals]
    # mix g between 0.6-3 micron
    gvals = (nwave > 0.6 * u.micron) & (nwave < 3.0 * u.micron)
    weights = (nwave[gvals] - 0.6 * u.micron) / 2.4 * u.micron
    weights = weights.value
    g[gvals] = (1.0 - weights) * g[gvals] + weights * modg[gvals]

    Qsca = Qext * albedo
    Qabs = Qext - Qsca

    ax.plot(nwave, Qext, "ko", label="new Qext")  
    ax.plot(nwave, Qsca, "bs", label="new Qsca")  
    ax.plot(nwave, Qabs, "gs", label="new Qabs")    

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda^{-1}$ [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"$Q/Q_\mathrm{ext}(V)$")
    ax.legend(fontsize=0.8*fontsize)

    ax_inset = ax.inset_axes([0.1, 0.15, 0.35, 0.35])
    ax_inset.plot(nwave, Qext, "ko", label="new Qext")  
    ax_inset.plot(nwave, Qsca, "bs", label="new Qsca")  
    ax_inset.plot(nwave, Qabs, "gs", label="new Qabs")  
    ax_inset.set_xscale("linear")
    ax_inset.set_yscale("linear")
    ax_inset.set_xlim(0.13, 0.3)
    ax_inset.set_ylim(0.5, 3.5)

    ax = axes[1, 1]

    ax.plot(1.0 / wd01mod.data_x, wd01mod.data_albedo, "b-",
            label="WD01 model albedo", alpha=0.4)
    ax.plot(1.0 / wd01mod.data_x, wd01mod.data_g, "b:",
            label="WD01 model g", alpha=0.4)
    ax.plot(nwave, albedo, "ko", label="new albedo")
    ax.plot(nwave, g, "gs", label="new g")

    ax.set_xscale("log")
    ax.set_xlim(0.0912, 30.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$\lambda^{-1}$ [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"albedo or g")
    ax.legend(fontsize=0.8*fontsize)

    plt.tight_layout()

    otab = Table()
    otab["wave"] = nwave
    otab["ext"] = Qext
    otab["albedo"] = albedo
    otab["g"] = g
    otab.write("sge2026_mw_empir_props.dat", format="ascii.commented_header", overwrite=True)

    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
