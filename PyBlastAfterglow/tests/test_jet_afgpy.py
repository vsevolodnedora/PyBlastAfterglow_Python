"""
    compare uniform jet evolution vs afterglowpy
    https://github.com/geoffryan/afterglowpy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PyBlastAfterglow.uutils import cgs
from PyBlastAfterglow.wrapper import BlastWave

thetaObs = 0.  # [0., 0.16]
freqobs = 1e9  # [1e9, 1e18]
thetaC = 0.1  # Half-opening angle in radians
thetaW = 0.1  # Truncation angle, unused for top-hat
E0 = 1e52  # Isotropic-equivalent energy in erg
n0 = 1e-3  # circumburst density in cm^{-3}
p = 2.2  # electron energy distribution index
eps_e = 1e-1  # epsilon_e
eps_B = 1e-2  # epsilon_B
dL = 3.09e26  # Luminosity distance in cm
z = 0.028
xi_N = 1.  # Fraction of electrons accelerated
b = 0  # power law index, unused for top-hat

package_dir = Path(__file__).parent.parent.parent
# where to read sampled files
data_dir = f"{package_dir}/data"
# where to save figures
figures_dir = f"{data_dir}/crosscheck_figures/jet_afgpy/"
#
comp_data_dir = f"{data_dir}/afterglowpy_lightcurves/"

def test_compare_jet_lightcurves(withSpread = False, a = 1.,
                                 eq_dthetadr="dthetadr_None",
                                 thetaMax=np.pi/2.,
                                 save="test_full_jet_with_afterglowpy_nospread.png",
                                 load_data=True
                                ):

    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax = axes

    t = np.geomspace(1.0e4, 1.0e8, 300)

    #for (i_thetaobs, i_freq, i_color) in [(0.16, 1.e18, "red")]:
    for (i_thetaobs, i_freq, i_color) in [(0., 1e9, "blue"), (0.16, 1e9, "orange"), (0., 1e18, "green"), (0.16, 1.e18, "red")]:


        # --- # uniform structured model # --- #
        # import time
        # tstart = time.time()

        model = BlastWave.from_analytic_pars(
            nlayers=100,
            EEc0=E0,
            Gamc0=150.,
            thetaCo=thetaC,
            theta0j=thetaW,
            r_pars=(8., 22., 1000),
            dens_pars=(n0, None, None, None, None),
            driver=("Nava", {"aa": a, "useSpread": withSpread, "epsilon_e_rad": 0, "adiabLoss": True,
                                       "eq_dthetadr": eq_dthetadr, "thetaMax": thetaMax,
                                       "eq_gammaAdi": "gamma_adi_peer", "eq_rhoprime": "rho2_transrel",
                                       "tprompt": 1e3, "epsilon_e_rad_RS": 0., "adiabLoss_RS": True,
                                       "ode_rtol": 1e-3, "ode_nsteps": 3000}),
            electrons=("Electron_BPL", {"p": p, "eps_e": eps_e, "eps_b": eps_B}),
            synchrotron=("Synchrotron_WSPN99", {"ssa":False}),
            eats=("EATS_StructuredLayersSource_Jit", {})
        )

        lightcurve = model.eats.lightcurve(i_thetaobs, t, i_freq, z, dL)

        # tend = time.time()

        # print("total time: {}".format(tend - tstart)); exit(0)

        ax.plot(t, lightcurve * 1e23 * 1e3, color=i_color, ls='-',
                label=r"$\theta_{obs}=$" + "{:.2f}".format(i_thetaobs) + r" $\nu$={:.1e}".format(i_freq))


        # --- # afterglowpy # --- #

        Y = np.array([i_thetaobs, E0, thetaC, thetaW, b, 0, 0, 0, n0, p, eps_e, eps_B, xi_N, dL])
        Z = {'z': z}

        #Fnu = grb.fluxDensity(t, i_freq, jetType, 0, *Y, **Z, spread=withSpread)


        # ax.plot(t, Fnu, color=i_color, ls='-', label=r"$\theta_{obs}=$"+"{:.2f}".format(i_thetaobs)+r" $\nu$={:.1e}".format(i_freq))

        # --- # JOELIB # --- #

        # import os, sys
        # sys.path.append("../../grb2/joelib")
        # from joelib3 import jetHeadGauss, light_curve_peer_SJ
        # jet = jetHeadGauss(EEc0=E0, Gamc0=150, nn=n0, epE=eps_e, epB=eps_B, pp=p, steps=1000, Rmin=1e8, Rmax=1e22,
        #              evolution='peer', nlayers=100, initJoAngle=thetaW*2, coAngle=thetaC*2, aa=1, structure='uniform',
        #              kk=0, shell_type='thin', Rb=1., withSpread=withSpread)
        # tt, lc, _, _ = light_curve_peer_SJ(jet, p, i_thetaobs, i_freq, dL, "discrete", t, None)
        #
        # ax.plot(tt, lc[0, :] * 1e23 * 1e3, color=i_color, ls=':')

        # --- # uniform simple model # --- #

        # model = UniformModel(
        #     E0=E0,
        #     Gamma0=1000.,
        #     theta0=thetaC,
        #     r_pars=(8., 22., 1000),
        #     dens_pars=(n0, None, None, None, None),
        #     driver_kwargs={"aa": np.nan, "useSpread": False, "epsilon_e_rad": 0, "adiabLoss": True,
        #                    "tprompt": 1e3, "epsilon_e_rad_RS": 0., "adiabLoss_RS": True,
        #                    "ode_rtol": 1e-3, "ode_nsteps": 3000,
        #                    "ncells": 1},
        #     electron_kwargs={"p": p, "eps_e": eps_e, "eps_b": eps_B}
        # )
        #
        # lightcurve = model.eats.lightcurve(i_thetaobs, t, i_freq, z, dL)
        #
        # ax.plot(t, lightcurve*1e23*1e3, color=i_color, ls='--')

        if load_data:
            if withSpread:
                fname = "afterglowpy_theta{:d}_lognu{:d}_spread.txt".format(int(i_thetaobs*180/np.pi),
                                                                            int(np.log10(i_freq)))
            else:
                fname = "afterglowpy_theta{:d}_lognu{:d}.txt".format(int(i_thetaobs * 180 / np.pi),
                                                                            int(np.log10(i_freq)))

            _t, _ref_F_afgpy, _ref_F = np.loadtxt(comp_data_dir+fname, unpack=True)

            # res = np.vstack((t, Fnu, lightcurve * 1e23 * 1e3)).T
            #np.savetxt("../data/"+fname, res, delimiter=' ', header='# t F(afterglowpy) F(our) | flux in mJy, time in days')
            ax.plot(_t, _ref_F_afgpy, color=i_color, ls='--')
            ax.plot(_t, _ref_F, color=i_color, ls=':', lw=2.)

    ax.plot([-1., -1.], [-2., -2.], color='gray', ls='-', label="Our code")
    ax.plot([-1., -1.], [-2., -2.], color='gray', ls=':', label=r"Our code [Ref]")
    ax.plot([-1., -1.], [-2., -2.], color='gray', ls='--', label=r"afterglowpy")

    ax.set_title(r"Comparison with afterglowpy (e.g. Fig.2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Flux (mJy)")
    ax.set_xlim(1e4, 1e8)
    ax.set_ylim(1e-9, 1e2)
    ax.legend()

    if not (save is None):
        plt.savefig(figures_dir + save)

    plt.show()

test_compare_jet_lightcurves(withSpread = False, a = 0, eq_dthetadr="dthetadr_None", thetaMax=np.pi/2.,
                                           save="test_jet_with_afterglowpy_nospread.png", load_data=True)



test_compare_jet_lightcurves(withSpread = True, a = 1, eq_dthetadr="dthetadr_AA", thetaMax=np.pi/2.,
                                           save="test_jet_with_afterglowpy_spread.png", load_data=True)