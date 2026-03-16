import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wavenum", help="use wave numbers on x axis", action="store_true"
    )
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

    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(10.0, 12.0), sharex=True, sharey=True)

    taus = [0.5, 1.5, 4.5]
    geoms = ["dusty", "shell", "cloudy"]
    cols = ["r", "g", "b"]
    lgeoms = ["h", "c"]
    lines = ["-", "--"]
    for i, ctau in enumerate(taus):
        for cgeom, ccol in zip(geoms, cols):
            for clgeom, cline in zip(lgeoms, lines):

                for k, cfA in enumerate([1.00, 0.00]):
                    dusttype = f"fA{cfA:.2f}"
                    fname = f"sge2026_{cgeom}_{clgeom}_{dusttype}_tau{ctau:.4f}_global_lum.table.fits"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=u.UnitsWarning)
                        dtab = Table.read(f"{cgeom}/{fname}")
                    wave = dtab["wavelength"].value
                    scat_to_direct = dtab["Flux_rt_s"] / dtab["Flux_rt_d"]

                    if args.wavenum:
                        wave = 1.0 / wave

                    axes[i, k].plot(
                        wave,
                        scat_to_direct,
                        color=ccol,
                        linestyle=cline,
                        label=rf"{cgeom} ({clgeom}, {dusttype})",
                    )


        axes[i, 0].set_title(rf"$\tau_V = {ctau:.2f}$")
        axes[i, 1].set_title(rf"$\tau_V = {ctau:.2f}$")

        if args.wavenum:
            xlabel = r"$1/\lambda$ [$\mu$m$^{-1}$]"
        else:
            xlabel = r"$\lambda$ [$\mu$m]"
        if i == 2:
            axes[i, 0].set_xlabel(xlabel)
            axes[i, 1].set_xlabel(xlabel)
        axes[i, 0].set_ylabel(r"$F_\mathrm{SCAT}/F_\mathrm{DIRECT}$")

        if args.wavenum:
            axes[i, 0].set_xlim(0.0, 10.0)
            axes[i, 1].set_xlim(0.0, 10.0)
            axes[i, 0].set_ylim(0.0, 2.0)
            axes[i, 1].set_ylim(0.0, 2.0)
        else:
            for j in range(2):
                axes[i, j].set_xscale("log")
                axes[i, j].set_yscale("log")
                axes[i, j].set_ylim(4e-6, 1e1)
                axes[i, j].set_xlim(0.912, 30.0)

        if i == 0:
            axes[i, 0].legend(ncols=2, fontsize=0.45 * fontsize)

    plt.tight_layout()

    save_str = "figs/sge2026_atten_examples_fA{fA:.2f}"
    if args.wavenum:
        save_str = f"{save_str}_wavenum"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
