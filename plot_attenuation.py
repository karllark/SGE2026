import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fA", help="mixure coefficient between MW and SMC", type=float, default=1.0
    )
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

    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(10.0, 12.0), sharex=True)

    taus = [0.5, 1.5, 4.5]
    geoms = ["dusty", "shell", "cloudy"]
    cols = ["r", "g", "b"]
    lgeoms = ["h", "c"]
    lines = ["-", "--"]
    dusttype = f"fA{args.fA:.2f}"
    for i, ctau in enumerate(taus):
        ax = axes[i, 0]
        for cgeom, ccol in zip(geoms, cols):
            for clgeom, cline in zip(lgeoms, lines):
                fname = f"sge2026_{cgeom}_{clgeom}_{dusttype}_tau{ctau:.4f}_global_lum.table.fits"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=u.UnitsWarning)
                    dtab = Table.read(f"{cgeom}/{fname}")
                wave = dtab["wavelength"].value
                norm_wave = 0.55
                tauatt = -1.0 * np.log(dtab["Flux"] / dtab["Flux_Input"])
                tauV = np.interp([norm_wave], wave, tauatt)

                if args.wavenum:
                    wave = 1.0 / wave

                ax.plot(
                    wave,
                    tauatt,
                    color=ccol,
                    linestyle=cline,
                    label=rf"{cgeom} ({clgeom}, {dusttype})",
                )

                # normalized at V band
                axes[i, 1].plot(
                    wave,
                    tauatt / tauV[0],
                    color=ccol,
                    linestyle=cline,
                    label=rf"{cgeom} ({clgeom}, {dusttype})",
                )

        ax.set_title(rf"$\tau_V = {ctau:.2f}$")
        axes[i, 1].set_title(rf"$\tau_V = {ctau:.2f}$")

        ax.plot(wave, dtab["tau_norm"] * ctau, "k-", label="Extinction curve")
        axes[i, 1].plot(wave, dtab["tau_norm"], "k-", label="Extinction curve")

        if args.wavenum:
            xlabel = r"$1/\lambda$ [$\mu$m$^{-1}$]"
        else:
            xlabel = r"$\lambda$ [$\mu$m]"
        if i == 2:
            ax.set_xlabel(xlabel)
            axes[i, 1].set_xlabel(xlabel)
        ax.set_ylabel(r"$\tau_\mathrm{att}$")
        axes[i, 1].set_ylabel(r"$\tau_\mathrm{att}/\tau_\mathrm{att,V}$")

        if args.wavenum:
            ax.set_xlim(0.0, 10.0)
            axes[i, 1].set_xlim(0.0, 10.0)
            ax.set_ylim(0.0, 7.5)
            axes[i, 1].set_ylim(0.0, 7.5)
        else:
            ax.set_xscale("log")
            axes[i, 1].set_xscale("log")
            ax.set_yscale("log")
            axes[i, 1].set_yscale("log")

        if i == 0:
            ax.legend(ncols=2, fontsize=0.45 * fontsize)

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
