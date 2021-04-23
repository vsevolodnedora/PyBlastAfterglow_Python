"""

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm
from pathlib import Path

# from PyBlastAfterglow.dynamics import EqOpts, Driver_Nava_FS, Driver_Peer_FS
# from PyBlastAfterglow.electrons import Electron_BPL
# from PyBlastAfterglow.synchrotron import Synchrotron_Joh06, Synchrotron_WSPN99
# from PyBlastAfterglow.eats import EATS_StructuredLayersSource, generate_skyimage
from PyBlastAfterglow.wrapper import BlastWave

package_dir = Path(__file__).parent.parent.parent
# where to read sampled files
data_dir = f"{package_dir}/data"
# where to save figures
figures_dir = f"{data_dir}/crosscheck_figures/struct_jet/"

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
Gamc0 = 150
t = np.geomspace(1.0e4, 1.0e8, 300)
nlayers = 100


def compare_jet_lightcurves(withSpread = False,
                            a = 1.,
                            eq_dthetadr="None",
                            thetaMax=np.pi/2.,
                            driver="Peer",
                            electrons="Electron_BPL",
                            synchrotron="Synchrotron_WSPN99", ssa=False,
                            ax=None,
                            ls='--', label=False, lw=1.):


    #for (i_thetaobs, i_freq, i_color) in [(0.16, 1.e18, "red")]:
    for (i_thetaobs, i_freq, i_color) in [
        (0., 1e9, "blue"),
        (0.16, 1e9, "orange"),
        (0., 1e18, "green"),
        (0.16, 1.e18, "red")
    ]:


        # --- # uniform structured model # --- #

        model = BlastWave.from_analytic_pars(
            nlayers=nlayers,
            EEc0=E0,
            Gamc0=Gamc0,
            thetaCo=thetaC,
            theta0j=thetaW,
            r_pars=(8., 22., 1000),
            dens_pars=(n0, None, None, None, None),
            driver=(driver, {"useSpread":withSpread, "aa":a, "ncells":1, "ode":'dop853',
                             "ode_pars":{"rtol":1e-5, "nsteps":1000, "first_step": 'default'},
                             "eq_delta":"default", "eq_dthetadr":eq_dthetadr, "eq_dmdr":"default", "eq_gammaAdi":"Nava",
                             "eq_rho2":"rel", "thetaMax":thetaMax, "adiabLoss":True, "epsilon_e_rad":0.}),
            # driver=(driver, {"aa": a, "useSpread": withSpread, "epsilon_e_rad": 0, "adiabLoss": True,
            #                "eq_dthetadr": eq_dthetadr, "thetaMax": thetaMax,
            #                "eq_gammaAdi": "gamma_adi_peer", "eq_rhoprime": "rho2_transrel",
            #                "tprompt": 1e3, "epsilon_e_rad_RS": 0., "adiabLoss_RS": True,
            #                "ode_rtol": 1e-3, "ode_nsteps": 3000}),#Driver_Nava_FS,
            electrons=(electrons, {"p": p, "eps_e": eps_e, "eps_b": eps_B}),
            synchrotron=(synchrotron, {"ssa": ssa}),
            eats=("EATS_StructuredLayersSource", {})
        )

        lightcurve = model.eats.lightcurve(i_thetaobs, t, i_freq, z, dL)
        lightcurve = np.sum(lightcurve, axis=1)
        if label:
            ax.plot(t, lightcurve * 1e23, color=i_color, ls=ls, lw=lw,
                    label=r"$\theta_{obs}=$" + "{:.2f}".format(i_thetaobs) + r" $\nu$={:.1e}".format(i_freq))
        else:
            ax.plot(t, lightcurve * 1e23, color=i_color, ls=ls, lw=lw)

class TestStructJet():

    def plot_test_effect_spreading(self):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        ax = axes

        compare_jet_lightcurves(withSpread=False,
                                a=-1.,
                                eq_dthetadr="None",
                                thetaMax=np.pi / 2.,
                                driver="Peer",
                                electrons="Electron_BPL",
                                synchrotron="Synchrotron_WSPN99",
                                ax=ax,
                                ls='-', label=True)

        compare_jet_lightcurves(withSpread=True,
                                a=1.,
                                eq_dthetadr="AA",
                                thetaMax=np.pi / 2.,
                                driver="Peer",
                                electrons="Electron_BPL",
                                synchrotron="Synchrotron_WSPN99",
                                ax=ax,
                                ls='--')

        # ax.plot([0., -0], [-0., -0.], color='blue', ls='-', label="Collimated")
        # ax.plot([0., -0.], [-0., -0.], color='red', ls='--', label=r"Spreading $a=1$")
        # ax.plot([0., -0], [-0., -0.], color='pink', ls='-', label="Collimated")
        # ax.plot([0., -0.], [-0., -0.], color='yellow', ls='--', label=r"Spreading $a=1$")
        # ax.plot([0., -0], [-0., -0.], color='magenta', ls='-', label="Collimated")
        # ax.plot([0., -0.], [-0., -0.], color='cyan', ls='--', label=r"Spreading $a=1$")
        # ax.plot([0., -0], [-0., -0.], color='green', ls='-', label="Collimated")
        # ax.plot([0., -0.], [-0., -0.], color='orange', ls='--', label=r"Spreading $a=1$")

        line1, = ax.plot([0., -0], [-0., -0.], color='gray', ls='-')
        line2, = ax.plot([0., -0.], [-0., -0.], color='gray', ls='--')

        leg2 = ax.legend([line1, line2], ["Collimated", r"Spreading $a=1$"], loc='upper left')
        ax.add_artist(leg2)

        ax.set_title(r"Spreading effect on lightcurve")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.minorticks_on()
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Flux (mJy)")
        ax.set_xlim(1e4, 1e8)
        ax.set_ylim(1e-9, 1e2)
        ax.legend()

        #if not (save is None):
        plt.savefig(figures_dir + "test_lightcurves_spread_effect.png")

        plt.show()

    def plot_test_effect_syn_meth(self):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        ax = axes

        compare_jet_lightcurves(withSpread=False,
                                a=-1.,
                                thetaMax=np.pi / 2.,
                                driver="Peer",
                                electrons="Electron_BPL",
                                synchrotron="Synchrotron_WSPN99",
                                ax=ax,
                                ls='-', label=True)

        compare_jet_lightcurves(withSpread=False,
                                a=-1.,
                                thetaMax=np.pi / 2.,
                                driver="Peer",
                                electrons="Electron_BPL",
                                synchrotron="Synchrotron_Joh06",
                                ax=ax,
                                ls='--')

        compare_jet_lightcurves(withSpread=False,
                                a=-1.,
                                thetaMax=np.pi / 2.,
                                driver="Peer",
                                electrons="Electron_BPL",
                                synchrotron="Synchrotron_Joh06", ssa=True,
                                ax=ax,
                                ls=':', lw=1.2)

        # ax.plot([0., -0], [-0., -0.], color='blue', ls='-', label="Collimated")
        # ax.plot([0., -0.], [-0., -0.], color='red', ls='--', label=r"Spreading $a=1$")
        # ax.plot([0., -0], [-0., -0.], color='pink', ls='-', label="Collimated")
        # ax.plot([0., -0.], [-0., -0.], color='yellow', ls='--', label=r"Spreading $a=1$")
        # ax.plot([0., -0], [-0., -0.], color='magenta', ls='-', label="Collimated")
        # ax.plot([0., -0.], [-0., -0.], color='cyan', ls='--', label=r"Spreading $a=1$")
        # ax.plot([0., -0], [-0., -0.], color='green', ls='-', label="Collimated")
        # ax.plot([0., -0.], [-0., -0.], color='orange', ls='--', label=r"Spreading $a=1$")

        line1, = ax.plot([0., -0], [-0., -0.], color='gray', ls='-')
        line2, = ax.plot([0., -0.], [-0., -0.], color='gray', ls='--')
        line3, = ax.plot([0., -0.], [-0., -0.], color='gray', ls=':')

        leg2 = ax.legend([line1, line2, line3], ["WSPN", r"Joh06", r"Joh06 [SSA]"], loc='upper left')
        ax.add_artist(leg2)

        ax.set_title(r"Spreading effect on lightcurve")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.minorticks_on()
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Flux (mJy)")
        ax.set_xlim(1e4, 1e8)
        ax.set_ylim(1e-9, 1e2)
        ax.legend()

        #if not (save is None):
        plt.savefig(figures_dir + "test_lightcurves_nospread_synch_meothds.png")

        plt.show()

    def spherical_uniform_source(self):
        thetaC = np.pi / 2.  # Half-opening angle in radians
        thetaW = np.pi / 2.  # Truncation angle, unused for top-hat

        model = BlastWave.from_analytic_pars(
            structure="uniform",
            nlayers=nlayers,
            EEc0=E0,
            Gamc0=Gamc0,
            thetaCo=thetaC,
            theta0j=thetaW,
            r_pars=(8., 22., 1000),
            dens_pars=(n0, None, None, None, None),
            driver=("Peer", {"aa": -1, "useSpread": False, "epsilon_e_rad": 0, "adiabLoss": True,
                                     "eq_dthetadr": "dthetadr_None", "thetaMax": np.pi / 2.,
                                     "eq_gammaAdi": "gamma_adi_peer", "eq_rhoprime": "rho2_transrel",
                                     "tprompt": 1e3, "epsilon_e_rad_RS": 0., "adiabLoss_RS": True,
                                     "ode_rtol": 1e-3, "ode_nsteps": 3000}),  # Driver_Nava_FS,
            electrons=("Electron_BPL", {"p": p, "eps_e": eps_e, "eps_b": eps_B}),
            synchrotron=("Synchrotron_WSPN99", {"ssa": False}),
            eats=("EATS_StructuredLayersSource", {})
        )

        lightcurve = model.eats.lightcurve(0., t, freqobs, z, dL)


        lightcurve_uni = model.eats.uniformspherelightcurve(t, freqobs, z, dL)

        fig, axes = plt.subplots(nrows=1, ncols=1)
        ax = axes

        # line1, = ax.plot([0., -0], [-0., -0.], color='gray', ls='-')
        # line2, = ax.plot([0., -0.], [-0., -0.], color='gray', ls='--')

        # leg2 = ax.legend([line1, line2], ["Collimated", r"Spreading $a=1$"], loc='upper left')
        # ax.add_artist(leg2)
        ax.plot(t, lightcurve, color="blue", label="structured")
        ax.plot(t, lightcurve_uni, color="red", label="Uni. Sphere")

        ax.set_title(r"Spreading effect on lightcurve")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.minorticks_on()
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Flux (mJy)")
        ax.set_xlim(1e4, 1e8)
        # ax.set_ylim(1e-9, 1e2)
        ax.legend()


        plt.savefig(figures_dir + "test_lightcurves_uniform_sphere.png")

        plt.show()


if __name__ == '__main__':
    o_test = TestStructJet()
    o_test.plot_test_effect_spreading()
    o_test.plot_test_effect_syn_meth()
    o_test.spherical_uniform_source()