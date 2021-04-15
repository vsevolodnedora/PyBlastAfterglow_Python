"""
    testing the velocity structuried (uniform angular) blast against Hotokezaka model
    e.g., 1511.05580
    comparing with plots from 1809.11161
    The ejecta data is publicly available at: 10.5281/zenodo.3588344
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import time

from PyBlastAfterglow.uutils import cgs
from PyBlastAfterglow.dynamics import get_Gamma
from PyBlastAfterglow.wrapper import BlastWave

package_dir = Path(__file__).parent.parent.parent
# where to read sampled files
data_dir = f"{package_dir}/data"
# where to save figures
figures_dir = f"{data_dir}/crosscheck_figures/ejecta_hkz/"
#
comp_data_dir = f"{data_dir}/hotokezaka_lightcurves/"

# load and process ejecta histogram
def load_vinf_hist(fpath):
    hist1 = np.loadtxt(fpath)
    vinf, mass = hist1[:, 0], hist1[:, 1]
    return (vinf, mass)

def kinetic_energy(velocity, mass):
    """ velocity [c] mass [Msun] -> Ek [ergs]"""
    return mass * cgs.solar_m * (velocity * cgs.c) ** 2

def _get_hist_cleared_data(fpath, beta_min=0.):

    hist_vinf, hist_vinf_M = load_vinf_hist(fpath)

    hist_vinf = hist_vinf[hist_vinf_M > 0.]
    hist_vinf_M = hist_vinf_M[hist_vinf_M > 0.]

    if beta_min > 0.:
        hist_vinf_M = hist_vinf_M[hist_vinf > beta_min]
        hist_vinf = hist_vinf[hist_vinf > beta_min]

    assert len(hist_vinf) > 0

    hist_ek = kinetic_energy(hist_vinf, hist_vinf_M)

    return (hist_vinf, hist_vinf_M, hist_ek)

def get_shell_data_from_hist(fpath):
    i_vinf, i_mass, i_ek = _get_hist_cleared_data(fpath)
    mtot = np.sum(i_mass)
    # i_mass = np.cumsum(i_mass[::-1])[::-1]
    # i_mass = i_mass / np.sum(i_mass) * mtot *2
    # i_ek = np.cumsum(i_ek[::-1])[::-1]

    i_mass *= cgs.solar_m

    shells = [{} for i in range(len(i_vinf))]
    for i in range(len(i_vinf)):
        shells[i]["Ek"] = i_ek[::-1][i]
        shells[i]["mass"] = i_mass[i]
        shells[i]["beta"] = i_vinf[i]
        shells[i]["theta"] = np.pi / 2.

    return shells

# for a "fast" parallel run
class EngineModel(object):

    def __init__(self,# shell_data, **kwargs):
                 shell_data,
                 nlayers=None,
                 r_pars=(8., 22., 500),
                 dens_pars=(1e-1, None, None, None, None),
                 driver=("Driver_Peer_FS", {"aa": -1, "useSpread": False, "epsilon_e_rad": 0, "adiabLoss": True,
                                     "eq_dthetadr": "dthetadr_None", "thetaMax": np.pi / 2.,
                                     "eq_gammaAdi": "gamma_adi_peer", "eq_rhoprime": "rho2_transrel",
                                     "tprompt": 1e3, "epsilon_e_rad_RS": 0., "adiabLoss_RS": True,
                                     "ode_rtol": 1e-3, "ode_nsteps": 3000, "ncells": 0}),
                 electrons=("Electron_BPL", {"p":2.5, "eps_e":1e-1, "eps_b":1e-1}),
                 synchrotron=("Synchrotron_Joh06", {"ssa": True}),
                 eats=("EATS_StructuredLayersSource", {}),
                 obs=("lightcurve", {"alpha_obs":0., "timegrid":np.logspace(-2., 7., 500),
                                     "freq":3e9, "z":0.0099, "d_l":46.6e6*cgs.pc})
                 ):
        self.nlayers = nlayers
        self.r_pars = r_pars
        self.dens_pars = dens_pars
        self.driver = driver
        self.electrons = electrons
        self.synchrotron = synchrotron
        self.eats = eats
        self.obstyp = obs[0]
        self.obspar = obs[1]

        # ejecta data


        # pars = copy.deepcopy(kwargs)
        #
        # # model parameters
        # self.nlayers=pars["nlayers"]
        # self.r_pars = pars["r_pars"]
        # self.dens_pars = pars["dens_pars"]
        # self.driver=pars["driver"]
        # self.electrons = pars["electrons"]
        # self.synchrotron = pars["synchrotron"]
        # self.eats = pars["eats"]
        # self.obstyp = pars["obs"][0]
        # self.obspar = pars["obs"][1]

        # ejecta data
        self.shell_data = shell_data

    def __call__(self, idx):

        # selecting data for a given shell
        data = self.shell_data[idx]

        # assert data["mass"][0]  >1.

        if hasattr(data["mass"], '__len__'):

            o_model = BlastWave.from_profile(
                nlayers=len(data["ctheta"]),
                dist_cthetas=data["ctheta"],
                dist_MM0s=data["mass"],
                dist_EEs=data["Ek"],
                dist_Gam0s=get_Gamma(data["beta"]),
                dist_Beta0s=data["beta"],
                r_pars=self.r_pars,
                dens_pars=self.dens_pars,
                driver=self.driver,
                electrons=self.electrons,
                synchrotron=self.synchrotron,
                eats=self.eats
            )

        else:

            assert data["mass"] > 1. # check if it is in kg and not in Msun

            if self.nlayers is None:
                raise ValueError("For non-uniform non-profile model, nlayers cannot be none")

            # # # uniform model
            o_model = BlastWave.from_analytic_pars_ejecta(
                nlayers=self.nlayers,
                Ek=data["Ek"],
                mass=data["mass"],
                beta=data["beta"],
                theta0=data["theta"],
                structure="uniform",
                r_pars=self.r_pars,
                dens_pars=self.dens_pars,
                driver=self.driver,
                electrons=self.electrons,
                synchrotron=self.synchrotron,
                eats=self.eats
            )

        if self.obstyp == "lightcurve":

            lightcurve_fj = o_model.eats.lightcurve(**self.obspar, jet='principle')
            lightcurve_cj = o_model.eats.lightcurve(**self.obspar, jet='counter')
            lightcurve = (np.sum(lightcurve_fj, axis=1) + np.sum(lightcurve_cj, axis=1))

            return (idx, lightcurve)
        elif self.obstyp == "skymap":

            fluxes, xx, yy, rr = o_model.eats.skymap(**self.obspar)

            return (idx, fluxes, xx, yy, rr)
        else:
            raise NameError("Obstype: {} not recognized".format(self.obstyp))

def run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs, n_cpu=None):

    if n_cpu is None: ncpus = os.cpu_count()
    else: ncpus = int(n_cpu)

    try:
        pool = Pool(ncpus)  # on 8 processors
        engine = EngineModel(
            shell_data,
            nlayers=nlayers,
            r_pars=r_pars,
            dens_pars=dens_pars,
            driver=driver,
            electrons=electrons,
            synchrotron=synchrotron,
            eats=eats,
            obs=obs
        )
        data_outputs = pool.map(engine, range(len(shell_data)))
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    # sort outputs by the index of a shell
    data_outputs = sorted(data_outputs, key=lambda x: x[0])
    data_outputs = [output[1] for output in data_outputs]

    model_lightcurves = np.vstack((data_outputs))

    return (np.sum(model_lightcurves, axis=0), model_lightcurves)

def compare1():

    times = np.logspace(-2., 7., 500)
    nlayers = 30
    r_pars = (8., 22., 500)
    dens_pars = (1e-1, None, None, None, None)
    driver = ("Peer", {"aa": -1, "useSpread": False, "epsilon_e_rad": 0, "adiabLoss": True,
                               "eq_dthetadr": "dthetadr_None", "thetaMax": np.pi / 2.,
                               "eq_gammaAdi": "gamma_adi_peer", "eq_rhoprime": "rho2_transrel",
                               "tprompt": 1e3, "epsilon_e_rad_RS": 0., "adiabLoss_RS": True,
                               "ode_rtol": 1e-3, "ode_nsteps": 3000, "ncells": 0})
    electrons = ("Electron_BPL", {"p": 2.5, "eps_e": 1e-1, "eps_b": 1e-1})
    synchrotron = ("Synchrotron_Joh06", {"ssa": True})
    eats = ("EATS_StructuredLayersSource", {})
    obs = ("lightcurve", {"alpha_obs": 0., "timegrid": times * cgs.day,
                          "freq": 3e9, "z": 0.001, "d_l": 100e6 * cgs.pc})

    do_dd2 = True
    do_bhblp = True
    do_shfho_all = True

    # # # --- plot

    fig = plt.figure()
    ax = plt.axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    # # # --- DD2 model

    if do_dd2:

        color="orange"

        shell_data = get_shell_data_from_hist(comp_data_dir+"DD2_ejecta_det_1.dat")
        lc1, _ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=1, alpha=2.0, ls=':')#, label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=1, alpha=2.0, ls='--')#, label="From Hist 2D")

    # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
    # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
    # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir + "DD2_n01_100Mpc.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=1, alpha=2.0, ls='-')

    # # # --- BHBlp model

    if do_bhblp:

        color = "blue"

        shell_data = get_shell_data_from_hist(fpath=comp_data_dir+"BHBlp_ejecta_det_1.dat")
        lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=1, alpha=2.0, ls=':')#, label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=1, alpha=2.0, ls='--')#, label="From Hist 2D")

        # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
        # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir+"BHBlp_n01_100Mpc.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=1, alpha=2.0, ls='-')

    # # # --- SFHo model

    if do_shfho_all:

        color = "blue"
        model = "SFHo_M135135_LK"
        electrons = ("Electron_BPL", {"p": 2.16, "eps_e":  0.1, "eps_b": 0.01})
        obs = ("lightcurve", {"alpha_obs": 0., "timegrid": times * cgs.day,
                              "freq": 3e9, "z": 0.0099, "d_l": 41.6e6 * cgs.pc})
        dens_pars = (5e-3, None, None, None, None)

        shell_data = get_shell_data_from_hist(fpath=comp_data_dir+"SFHo_ejecta_det_1.dat")
        lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls=':')#, label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls='--')#, label="From Hist 2D")

        # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
        # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir+"SFHo_n0005.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=2, alpha=2.0, ls='-')

        # ------------------ #

        color = "orange"
        dens_pars = (1e-3, None, None, None, None)

        lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls=':')  # , label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls='--')  # , label="From Hist 2D")

        # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
        # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir+"SFHo_n0001.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=2, alpha=2.0, ls='-')

        # ------------------ #

        color = "red"
        dens_pars = (1e-4, None, None, None, None)

        lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls=':')  # , label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls='--')  # , label="From Hist 2D")

        # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
        # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir+"SFHo_n00001.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=2, alpha=2.0, ls='-')

    # # # --- plot rest --- # # #

    ax.plot([-1., -1], [-1., -1], color="gray", lw=1, ls='--', label="This code")
    ax.plot([-1., -1], [-1., -1], color="gray", lw=1, ls='-', label="Hotokezaka code")

    l1, = ax.plot([-1., -1], [-1., -1], color="orange", lw=1, ls='-')#, label=r"DD2 M135135 LK")
    l2, = ax.plot([-1., -1], [-1., -1], color="blue", lw=1, ls='-')#, label=r"BHBlp M135135 LK")

    l3, = ax.plot([-1., -1], [-1., -1], color="blue", lw=2, ls='-')#, label=r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)")
    l4, = ax.plot([-1., -1], [-1., -1], color="orange", lw=2, ls='-')#, label=r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)")
    l5, = ax.plot([-1., -1], [-1., -1], color="red", lw=2, ls='-')#, label=r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)")

    leg2 = ax.legend([l1, l2, l3, l4, l5], [r"DD2 M135135 LK",
                                            r"BHBlp M135135 LK",
                                            r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)",
                                            r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)",
                                            r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)"
                                            ], loc='upper left')
    ax.add_artist(leg2)
    ax.legend(loc='upper right')

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1e1, 3e4) # ax.set_xlim(1e1, 4e3)
    ax.set_ylim(1e0, 2e3)
    # ax.set_ylabel(r"$F_\nu\ {\rm at}\ 100\,{\rm Mpc}\,[{\rm \mu J y}] $")
    ax.set_ylabel(r"$F_\nu\, [{\rm \mu J y}] $")
    ax.set_xlabel(r"$t$ [days]")
    ax.set_title("Comparison with Fig. 30 and Fig. 31 from Radice+2018")

    #if not (save is None):
    plt.savefig(figures_dir + "ejecta_afterglow_vs_hotokezaka.png")

    plt.show()

def performance():
    times = np.logspace(-2., 7., 500)
    nlayers = 30
    r_pars = (8., 22., 500)
    dens_pars = (1e-1, None, None, None, None)
    driver = ("Peer", {"aa": -1, "useSpread": False, "epsilon_e_rad": 0, "adiabLoss": True,
                       "eq_dthetadr": "dthetadr_None", "thetaMax": np.pi / 2.,
                       "eq_gammaAdi": "gamma_adi_peer", "eq_rhoprime": "rho2_transrel",
                       "tprompt": 1e3, "epsilon_e_rad_RS": 0., "adiabLoss_RS": True,
                       "ode_rtol": 1e-3, "ode_nsteps": 3000, "ncells": 0})
    electrons = ("Electron_BPL", {"p": 2.5, "eps_e": 1e-1, "eps_b": 1e-1})
    synchrotron = ("Synchrotron_Joh06", {"ssa": True})
    eats = ("EATS_StructuredLayersSource", {})
    obs = ("lightcurve", {"alpha_obs": 0., "timegrid": times * cgs.day,
                          "freq": 3e9, "z": 0.001, "d_l": 100e6 * cgs.pc})

    do_dd2 = True
    do_bhblp = False
    do_shfho_all = False

    tstart = time.time()

    # # # --- plot

    fig = plt.figure()
    ax = plt.axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    # # # --- DD2 model

    if do_dd2:
        color = "orange"

        lcs = []
        shell_data = get_shell_data_from_hist(comp_data_dir + "DD2_ejecta_det_1.dat")
        o_model = EngineModel(shell_data, nlayers, r_pars, dens_pars, driver,  electrons, synchrotron, eats, obs)
        for i_shell in range(len(shell_data)):
            print("shell:{}/{}".format(i_shell,len(shell_data)))
            _, lc = o_model(i_shell)
            lcs.append(lc)
        lc1 = np.sum(np.vstack((lcs)), axis=0)

        # lc1, _ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=1, alpha=2.0, ls=':')  # , label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=1, alpha=2.0, ls='--')#, label="From Hist 2D")

        # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
        # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir + "DD2_n01_100Mpc.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=1, alpha=2.0, ls='-')

    # # # --- BHBlp model

    if do_bhblp:
        color = "blue"

        shell_data = get_shell_data_from_hist(fpath=comp_data_dir + "BHBlp_ejecta_det_1.dat")
        lc1, _ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=1, alpha=2.0, ls=':')  # , label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=1, alpha=2.0, ls='--')#, label="From Hist 2D")

        # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
        # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir + "BHBlp_n01_100Mpc.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=1, alpha=2.0, ls='-')

    # # # --- SFHo model

    if do_shfho_all:
        color = "blue"
        model = "SFHo_M135135_LK"
        electrons = ("Electron_BPL", {"p": 2.16, "eps_e": 0.1, "eps_b": 0.01})
        obs = ("lightcurve", {"alpha_obs": 0., "timegrid": times * cgs.day,
                              "freq": 3e9, "z": 0.0099, "d_l": 41.6e6 * cgs.pc})
        dens_pars = (5e-3, None, None, None, None)

        shell_data = get_shell_data_from_hist(fpath=comp_data_dir + "SFHo_ejecta_det_1.dat")
        lc1, _ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls=':')  # , label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls='--')#, label="From Hist 2D")

        # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
        # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir + "SFHo_n0005.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=2, alpha=2.0, ls='-')

        # ------------------ #

        color = "orange"
        dens_pars = (1e-3, None, None, None, None)

        lc1, _ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls=':')  # , label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls='--')  # , label="From Hist 2D")

        # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
        # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir + "SFHo_n0001.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=2, alpha=2.0, ls='-')

        # ------------------ #

        color = "red"
        dens_pars = (1e-4, None, None, None, None)

        lc1, _ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls=':')  # , label="From Hist 1D")
        #
        # shell_data = o_ej.get_shell_data_from_hist_2d(n_theta=nlayers)
        # lc1,_ = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color=color, linewidth=2, alpha=2.0, ls='--')  # , label="From Hist 2D")

        # shell_data = o_ej.get_shell_data_from_corr_2d(n_theta=nlayers)
        # lc1 = run(shell_data, nlayers, r_pars, dens_pars, driver, electrons, synchrotron, eats, obs)
        # ax.plot(times, lc1 * 1e23 * 1e6, color="blue", linewidth=1, alpha=2.0, ls='-', label="From Corr 2D")

        kenta_t, kenta_f = np.loadtxt(comp_data_dir + "SFHo_n00001.dat", unpack=True, usecols=(0, 2))
        ax.plot(kenta_t, kenta_f, color=color, linewidth=2, alpha=2.0, ls='-')

    # # # --- plot rest --- # # #

    tend = time.time()
    print("Time: {}".format(tend - tstart))

    ax.plot([-1., -1], [-1., -1], color="gray", lw=1, ls='--', label="This code")
    ax.plot([-1., -1], [-1., -1], color="gray", lw=1, ls='-', label="Hotokezaka code")

    l1, = ax.plot([-1., -1], [-1., -1], color="orange", lw=1, ls='-')  # , label=r"DD2 M135135 LK")
    l2, = ax.plot([-1., -1], [-1., -1], color="blue", lw=1, ls='-')  # , label=r"BHBlp M135135 LK")

    l3, = ax.plot([-1., -1], [-1., -1], color="blue", lw=2,
                  ls='-')  # , label=r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)")
    l4, = ax.plot([-1., -1], [-1., -1], color="orange", lw=2,
                  ls='-')  # , label=r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)")
    l5, = ax.plot([-1., -1], [-1., -1], color="red", lw=2,
                  ls='-')  # , label=r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)")

    leg2 = ax.legend([l1, l2, l3, l4, l5], [r"DD2 M135135 LK",
                                            r"BHBlp M135135 LK",
                                            r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)",
                                            r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)",
                                            r"SFHo M135135 LK ($n = 5\times10^{-3}$ cm$^{-3}$)"
                                            ], loc='upper left')
    ax.add_artist(leg2)
    ax.legend(loc='upper right')

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1e1, 3e4)  # ax.set_xlim(1e1, 4e3)
    ax.set_ylim(1e0, 2e3)
    # ax.set_ylabel(r"$F_\nu\ {\rm at}\ 100\,{\rm Mpc}\,[{\rm \mu J y}] $")
    ax.set_ylabel(r"$F_\nu\, [{\rm \mu J y}] $")
    ax.set_xlabel(r"$t$ [days]")
    ax.set_title("Comparison with Fig. 30 and Fig. 31 from Radice+2018")

    # if not (save is None):
    plt.savefig(figures_dir + "ejecta_afterglow_vs_hotokezaka.png")

    plt.show()

if __name__ == '__main__':
    # compare1()
    performance()