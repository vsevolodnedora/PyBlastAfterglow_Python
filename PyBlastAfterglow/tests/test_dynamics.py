"""
    pass

"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


package_dir = Path(__file__).parent.parent.parent
# where to read sampled files
data_dir = f"{package_dir}/data"
# where to save figures
figures_dir = f"{data_dir}/crosscheck_figures/dynamics/"

from PyBlastAfterglow.dynamics import (Driver_Nava_FS,Driver_Nava_FSRS,Driver_Peer_FS)
from PyBlastAfterglow.dynamics import (get_beta,get_Rdec2,get_Rdec,get_sedovteylor,get_Gamma,evolove_driver,get_bm79)
from PyBlastAfterglow.uutils import cgs, find_nearest_index

class TestDynWithRadLoss:

    # Uniform density, 2 __models, Self-Similar solutions
    def test_plot_dyn_evol_nonspread_radloss(self):
        # source parameters # shock parameters
        Gamma0 = 1000
        theta0 = np.pi / 180. * 5  # np.pi/2.
        E0 = 1e55 * (1 - np.cos(theta0)) / 2.
        # Grids
        Rstart, Rend, nR = 10 ** 8., 10 ** 22., 1001
        r_grid = np.logspace(np.log10(Rstart), np.log10(Rend), nR)
        # CBM
        s = 0
        nCM = 1e-1
        # Radiation
        p = 2.2
        eB = 10 ** (-3)

        epsilone = 0.2  # -0.33#np.log10(2.5e-1) #Fraction of energy in the electrons

        aa = -1  # Sets some equtions

        # --- run dynamics
        # o_nava_nosp1 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)
        # o_nava_norad = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa, radiativeLosses=False, fixed_epsilon=False)
        # o_nava_rad1 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa, radiativeLosses=True, remix_radloss=True, fixed_epsilon=True, epsolon_rad=1.)
        # o_nava_rad2 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa, radiativeLosses=True, remix_radloss=True, fixed_epsilon=False)

        o_nava_norad = evolove_driver(
            driver=Driver_Nava_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=-1., ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )
        o_nava_rad1 = evolove_driver(
            driver=Driver_Nava_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=-1., ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=1.,
        )
        # o_nava_rad2 = evolove_driver(
        #     driver=NavaFS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(), r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0.
        # )

        # aa = 1  # Sets some equtions
        # # --- run dynamics
        # o_peer = EvolveShellPeer(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, p=p, eB=eB, epsilone=epsilone, useSpread=True, aa=aa)

        # o_nava = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)

        # --- analytic arguments
        # R = np.logspace(np.log10(Rstart), np.log10(Rend), nR)
        # M0 = E0 * cgs.c ** -2 / Gamma0
        # # rdec = Equations.get_Rdec(M0, nCM * (cgs.me + cgs.mp), Gamma0, s)
        # rdec = Equations.get_Rdec2(E0, nCM, Gamma0)
        # mask = R > rdec
        # # tdec = o_dyn.tobs[find_nearest_index(o_dyn.R, rdec)]
        # Gamma_bm = get_bm79(E0, nCM * (cgs.me + cgs.mp), s, R)
        # delta_R_bm = R[find_nearest_index(o_nava.out_fs_arr["Gamma"], 0.1 * Gamma0)] / \
        #              R[find_nearest_index(Gamma_bm, 0.1 * Gamma0)]
        #
        # beta0 = Equations.beta(Gamma0)
        # beta_st = get_sedovteylor(rdec, beta0, o_nava.out_fs_arr["R"])
        # delta_R_st = R[find_nearest_index(o_nava.out_fs_arr["beta"], 0.1 * beta0)] / \
        #              R[find_nearest_index(beta_st, 0.1 * beta0)]

        # --- plot

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6.2, 4.6), sharex="all")
        ip = 0

        # ax.plot(R[mask] * delta_R_bm, Gamma_bm[mask], color="black", ls='-', label=r"$\Gamma_{\rm BM79}$")

        # ax.plot(R[mask] * delta_R_st, Gamma0 * beta_st[mask], color="black", ls='--', label=r"$\Gamma_0 \beta_{\rm ST}$")

        # ax.plot(o_nava_nosp1.out_fs_arr["R"], o_nava_nosp1.out_fs_arr["Gamma"], color="black", ls='-')  # , label=r"$\Gamma$ [P]")
        # ax.plot(o_nava_nosp2.out_fs_arr["R"], o_nava_nosp2.out_fs_arr["Gamma"], color="gray", ls='-')  # , label=r"$\Gamma$ [P]")
        ax.plot(o_nava_norad.vals["R"], o_nava_norad.vals["Gamma"], color="black", ls='-')  # , label=r"$\Gamma$ [N]")
        ax.plot(o_nava_rad1.vals["R"], o_nava_rad1.vals["Gamma"], color="red", ls='-')  # , label=r"$\Gamma$ [N]")
        # ax.plot(o_nava_rad2.get("R"), o_nava_rad2.get("Gamma"), color="green", ls='-')
        # ax.plot(o_peer.out_fs_arr["R"], o_peer.out_fs_arr["Gamma"], color="gray", ls='-')#, label=r"$\Gamma$ [P]")

        # ax.plot(o_nava.out_fs_arr["R"], o_nava.out_fs_arr["Gamma"], color="green", ls='-')#, label=r"$\Gamma$ [N]")
        # ax.plot(o_peer_.out_fs_arr["R"], o_peer_.out_fs_arr["Gamma"], color="red", ls='-')#, label=r"$\Gamma$ [N]")

        # ax.plot(o_nava_nosp1.out_fs_arr["R"], Gamma0 * o_nava_nosp1.out_fs_arr["beta"], color="black", ls='--')#, label=r"$\Gamma_0 \beta$ [P]")
        # ax.plot(o_nava_nosp2.out_fs_arr["R"], Gamma0 * o_nava_nosp2.out_fs_arr["beta"], color="gray", ls='--')  # , label=r"$\Gamma_0 \beta$ [P]")
        ax.plot(o_nava_norad.vals["R"], Gamma0 * o_nava_norad.vals["beta"], color="black",
                ls='--')  # , label=r"$\Gamma_0 \beta$ [N]")
        ax.plot(o_nava_rad1.vals["R"], Gamma0 * o_nava_rad1.vals["beta"], color="red",
                ls='--')  # , label=r"$\Gamma_0 \beta$ [N]")
        # ax.plot(o_nava_rad2.get("R"), Gamma0 * o_nava_rad2.get("beta"), color="green", ls='--')
        # ax.plot(o_peer.out_fs_arr["R"], Gamma0 * o_peer.out_fs_arr["beta"], color="gray", ls='--')#, label=r"$\Gamma$ [P]")

        ax.plot([-10. - 20], [-20. - 30], color='gray', ls='-', label=r"$\Gamma$")
        ax.plot([-10. - 20], [-20. - 30], color='gray', ls='--', label=r"$\Gamma_0\beta$")
        ax.plot([-10. - 20], [-20. - 30], color='black', ls='-', label=r"Adiabatic")
        ax.plot([-10. - 20], [-20. - 30], color='red', ls='-', label=r"Radiative")
        ax.plot([-10. - 20], [-20. - 30], color='green', ls='-', label=r"Semi-radiative")
        # ax.plot([-10. - 20], [-20. - 30], color='red', ls='-', label=r"$d\theta/dr = f(\theta)$")

        # ax.axvline(x=rdec, color="gray", linestyle=":", label=r"$R_{dec}$")
        ax.set_xlabel(r"$R$ [cm]", fontsize=12)
        ax.set_ylabel(r"$\Gamma$, $\Gamma_0\beta$", fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(**{"axis": 'both', "which": 'both', "labelleft": True, 'labelright': False, "tick1On": True,
                          "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True,
                          "left": True,
                          "right": True})
        legend = {"fancybox": False, "loc": 'lower left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax.legend(**legend)
        ax.set_ylim(1e0, 2e3)
        ax.set_xlim(1e16, 1e21)

        ax2 = ax.twinx()
        # ax2.plot(o_nava_sp1.out_fs_arr["R"], o_nava_sp1.out_fs_arr["theta"]*180/np.pi, color="green", ls=':')#, label=r"$m$ [N]")
        # ax2.plot(o_nava_rad2.get("R"), o_nava_rad2.get("Erad2")/(o_nava_rad2.get("Esh2")*epsilone), color="green", ls=':', label=r'$\varepsilon_{rad}$')#, label=r"$m$ [P]")
        # ax2.plot(o_nava_norad.out_fs_arr["R"], np.log10(o_nava_norad.out_fs_arr["gamma_c"]), color="gray", ls=':')#, label='$\gamma_c$')
        # ax2.plot(o_nava_norad.out_fs_arr["R"], np.log10(o_nava_norad.out_fs_arr["gamma_min"]), color="gray", ls=':')#, label='$\gamma_{min}$')
        # ax2.plot(o_nava_rad2.out_fs_arr["R"], np.log10(o_nava_rad2.out_fs_arr["gamma_c"]), color="red", ls=':', label='$\gamma_c$')
        # ax2.plot(o_nava_rad2.out_fs_arr["R"], np.log10(o_nava_rad2.out_fs_arr["gamma_min"]), color="green", ls=':', label='$\gamma_{min}$')
        # ax2.plot(o_peer.out_fs_arr["R"], o_peer.out_fs_arr["theta"]*180/np.pi, color="gray", ls=':')#, label=r"$\Gamma$ [P]")
        ax2.set_xlabel(r"$R$", fontsize=14)
        ax2.set_ylabel(r"$\varepsilon_{rad}$", fontsize=14)  # r"$\rho_{\rm CBM}$"
        ax2.set_xscale("log")
        ax2.set_yscale("linear")
        tick_params = {"axis": 'both', "which": 'both', "labelleft": False, 'labelright': True, "tick1On": False,
                       "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True, "left": False,
                       "right": True}
        ax2.tick_params(**tick_params)
        ax2.minorticks_on()

        legend = {"fancybox": False, "loc": 'upper right',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax2.legend(**legend)

        # ax.text(0.05, 0.7, r"$\Gamma_0=1000$" + "\n" +
        #                   "$n=0.1$" + "\n" + "$E_0=10^{55}$", color="black", fontsize=12,
        #               transform=ax.transAxes)

        dic = {"left": 0.16, "bottom": 0.12, "top": 0.98, "right": 0.91, "hspace": 0}
        plt.subplots_adjust(**dic)
        plt.savefig(figures_dir + "unif_fs_comp_nospread_radloss.png")
        # plt.tight_layout()
        plt.show()


class TestDynNonUniformCBM:

    # inoform density, 1 model, RS, 3 Gamma0
    def task_plot_evold_unif_rs(self):
        # source parameters # shock parameters
        Gamma0 = 1000.
        theta0 = np.pi / 180. * 5  # np.pi/2.
        E0 = 1.e55 * (1 - np.cos(theta0)) / 2.
        # Grids
        Rstart, Rend, nR = 10. ** 8, 10. ** 22, 1001
        r_grid = np.logspace(np.log10(Rstart), np.log10(Rend), nR)
        # CBM
        s = 0
        nCM = 1e-1
        # Radiation
        p = 2.2
        eB = 10 ** (-3)
        pRS = 2.05
        eB3 = 10 ** (-4)
        epsilone3 = 10 ** (-3.)

        epsilone = 0.2  # -0.33#np.log10(2.5e-1) #Fraction of energy in the electrons

        aa = -1  # Sets some equtions

        o_nava_fs = evolove_driver(
            driver=Driver_Nava_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=-1., ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )

        o_nava_rs = evolove_driver(
            driver=Driver_Nava_FSRS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=-1., ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-8, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0., tprompt=1e3, adiabLoss_RS=True, reverseShock=True,
            epsilon_e_rad_RS=0.
        )

        # o_nava_fs = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=-1, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi / 2.,
        #     eq_gammaAdi = EqOpts.gamma_adi_peer, eq_rhoprime = EqOpts.rho2_transrel
        # )
        #
        # o_nava_rs = evolove_driver(
        #     driver=Driver_Nava_FSRS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=-1, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., reverseShock=True, adiabLoss_RS=True, tprompt=1e3,
        #     epsilon_e_rad_RS=0., eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )

        # o_nava_rs = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                           r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(nCM, None, None, None, None),
        #                           ncells=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                           adiabaticLosses=True, aa=aa,
        #                           eB3=eB3, epsilone3=epsilone3, pRS=pRS, reverseShock=True, tprompt=1e3)
        #
        # o_nava_fs = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(nCM, None, None, None, None),
        #                          ncells=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa,
        #                          eB3=eB3, epsilone3=epsilone3, pRS=pRS, reverseShock=False, tprompt=1e3)
        # pass

        # --- run dynamics
        # o_peer = EvolveShellPeer(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(nCM, None, None, None, None),
        #                          ncells=None, p=p, eB=eB, epsilone=epsilone, useSpread=False, aa=aa)

        # o_nava1 = EvolveShellNava(E0=E0, Gamma0=Gamma0[0], thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(nCM, None, None, None, None),
        #                          ncells=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa,
        #                          eB3=eB3, epsilone3=epsilone3, pRS=pRS, reverseShock=True, tprompt=1e3)
        # o_nava2 = EvolveShellNava(E0=E0, Gamma0=Gamma0[1], thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(nCM, None, None, None, None),
        #                          ncells=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa,
        #                          eB3=eB3, epsilone3=epsilone3, pRS=pRS, reverseShock=True, tprompt=1e3)
        # o_nava3 = EvolveShellNava(E0=E0, Gamma0=Gamma0[2], thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(nCM, None, None, None, None),
        #                          ncells=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa,
        #                          eB3=eB3, epsilone3=epsilone3, pRS=pRS, reverseShock=True, tprompt=1e3)

        # --- analytic arguments
        # R = np.logspace(np.log10(Rstart), np.log10(Rend), nR)
        # M0 = E0 * cgs.c ** -2 / Gamma0
        # rdec = Equations.get_Rdec(M0, nCM * (cgs.me + cgs.mp), Gamma0, s)
        # mask = R > rdec
        # # tdec = o_dyn.tobs[find_nearest_index(o_dyn.R, rdec)]
        # Gamma_bm = get_bm79(E0, nCM * (cgs.me + cgs.mp), s, R)
        # delta_R_bm = R[find_nearest_index(o_nava.out_fs_arr["Gamma"], 0.1 * Gamma0)] / \
        #              R[find_nearest_index(Gamma_bm, 0.1 * Gamma0)]
        #
        # beta0 = Equations.beta(Gamma0)
        # beta_st = get_sedovteylor(rdec, beta0, o_nava.out_fs_arr["R"])
        # delta_R_st = R[find_nearest_index(o_nava.out_fs_arr["beta"], 0.1 * beta0)] / \
        #              R[find_nearest_index(beta_st, 0.1 * beta0)]

        # --- plot

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6.2, 4.6), sharex="all")
        ip = 0

        ax.plot(o_nava_fs.vals["R"], o_nava_fs.vals["Gamma"], color="green", ls='-', label=r"$\Gamma$ [FS]")
        ax.plot(o_nava_rs.vals["R"], o_nava_rs.vals["Gamma"], color="red", ls='-', label=r"$\Gamma$ [FS & RS]")
        # ax.plot(o_nava2.out_fs_arr["R"], o_nava2.out_fs_arr["Gamma"], color="blue", ls='-')#, label=r"$\Gamma$ [N]")
        # ax.plot(o_nava3.out_fs_arr["R"], o_nava3.out_fs_arr["Gamma"], color="red", ls='-')#, label=r"$\Gamma$ [N]")

        ax.plot(o_nava_fs.vals["R"], Gamma0 * o_nava_fs.vals["beta"], color="green", ls='--',
                label=r"$\Gamma_0 \beta$ [FS]")
        ax.plot(o_nava_rs.vals["R"], Gamma0 * o_nava_rs.vals["beta"], color="red", ls='--',
                label=r"$\Gamma_0 \beta$ [FS & RS]")
        # ax.plot(o_nava2.out_fs_arr["R"], Gamma0[1] * o_nava2.out_fs_arr["beta"], color="blue", ls='--')#, label=r"$\Gamma_0 \beta$ [N]")
        # ax.plot(o_nava3.out_fs_arr["R"], Gamma0[2] * o_nava3.out_fs_arr["beta"], color="red",ls='--')#, label=r"$\Gamma_0 \beta$ [N]")

        # ax.axvline(x=rdec, color="gray", linestyle=":", label=r"$R_{dec}$")
        ax.set_xlabel(r"$R$ [cm]", fontsize=12)
        ax.set_ylabel(r"$\Gamma$, $\Gamma_0\beta$", fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(**{"axis": 'both', "which": 'both', "labelleft": True, 'labelright': False, "tick1On": True,
                          "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True,
                          "left": True,
                          "right": True})
        legend = {"fancybox": False, "loc": 'lower left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax.legend(**legend)
        ax.set_ylim(1e0, 2e3)
        ax.set_xlim(1e14, 2e21)  # (1e8,1e20)#(1e15, 2e21)
        # ax.grid(True)

        ax.text(0.05, 0.65, r"$\Gamma_0=1000$" + "\n" +
                "$n=0.1$" + "\n" + "$E_0=10^{55}$", color="black", fontsize=12,
                transform=ax.transAxes)

        ax2 = ax.twinx()
        ax2.plot(o_nava_fs.vals["R"], o_nava_fs.vals["M2"], color="green", ls=':', label=r"$m_2$ [FS]")
        ax2.plot(o_nava_rs.vals["R"], o_nava_rs.vals["M3"], color="red", ls=':', label=r"$m_3$ [RS]")
        # ax2.plot(o_nava2.out_fs_arr["R"], o_nava2.out_rs_arr["M3"], color="blue", ls=':')#, label=r"$\Gamma_0 \beta$ [N]")
        #  ax2.plot(o_nava3.out_fs_arr["R"], o_nava3.out_rs_arr["M3"], color="red", ls=':')#, label=r"$\Gamma_0 \beta$ [N]")

        # ax2.plot(o_peer.out_fs_arr["R"], o_peer.out_fs_arr["m"], color="gray", ls='--', label=r"$m$ [P]")
        ax2.set_xlabel(r"$R$", fontsize=14)
        ax2.set_ylabel(r"$m$ $[M_{\odot}]$", fontsize=14)  # r"$\rho_{\rm CBM}$"
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        tick_params = {"axis": 'both', "which": 'both', "labelleft": False, 'labelright': True, "tick1On": False,
                       "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True, "left": False,
                       "right": True}
        ax2.tick_params(**tick_params)

        legend = {"fancybox": False, "loc": 'upper right',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax2.legend(**legend)

        dic = {"left": 0.12, "bottom": 0.12, "top": 0.98, "right": 0.88, "hspace": 0}
        plt.subplots_adjust(**dic)
        plt.savefig(figures_dir + "unif_fs_rs_nonspread_noloss.png")
        # plt.tight_layout()
        plt.show()

    # step-func density, 2 __models Gamma, rho
    def task_plot_dyn_evol_step_dens_prof_steep(self):
        # source parameters # shock parameters
        Gamma0 = 1000
        theta0 = np.pi / 180. * 5  # np.pi/2.
        E0 = 1e55 * (1 - np.cos(theta0)) / 2.
        # Grids
        Rstart, Rend, nR = 1e2, 1e32, 1001
        r_grid = np.logspace(np.log10(Rstart), np.log10(Rend), nR)
        R_EJ = 1e16
        R_ISM = 1e22
        # CBM
        A0 = 1e62
        s = 3.5
        nCM = 1e-1
        # Radiation
        p = 2.2
        eB = 10 ** (-3)

        aa = -1

        epsilone = 0.2  # -0.33#np.log10(2.5e-1) #Fraction of energy in the electrons

        # --- run dynamics
        o_peer = evolove_driver(
            driver=Driver_Peer_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(None, A0, s, R_EJ, R_ISM),
            # kwargs
            useSpread=False, aa=-1., ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )

        o_nava = evolove_driver(
            driver=Driver_Nava_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(None, A0, s, R_EJ, R_ISM),
            # kwargs
            useSpread=False, aa=-1., ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )

        # o_peer = evolove_driver(
        #     driver=Driver_Peer_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(None, A0, s, R_EJ, R_ISM), aa=aa, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )
        #
        # o_nava = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(None, A0, s, R_EJ, R_ISM), aa=aa, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )
        # o_peer = EvolveShellPeer(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(None, A0, s, R_EJ, R_ISM),
        #                          ncells=None, p=p, eB=eB, epsilone=epsilone, useSpread=False, aa=aa)
        #
        # o_nava = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(None, A0, s, R_EJ, R_ISM),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)

        # ---
        M0 = E0 * cgs.c ** -2 / Gamma0
        # rdec = Equations.get_Rdec(M0, A0 * (cgs.me + cgs.mp), Gamma0, s)

        subplots = {"ncols": 1, "nrows": 1, "figsize": (6.2, 4.6), "sharex": "all"}
        fig, ax = plt.subplots(**subplots)
        ax.plot(o_peer.vals["R"], o_peer.vals["Gamma"], color="green", ls='-', label=r"$\Gamma$ [P]")
        ax.plot(o_nava.vals["R"], o_nava.vals["Gamma"], color="blue", ls='-', label=r"$\Gamma$ [N]")
        ax.plot(o_peer.vals["R"], o_peer.vals["Gamma"][0] * o_peer.vals["beta"], color="green", ls='--',
                label=r"$\Gamma_0 \beta$")
        ax.plot(o_nava.vals["R"], o_nava.vals["Gamma"][0] * o_nava.vals["beta"], color="blue", ls='--',
                label=r"$\Gamma_0 \beta$")
        ax.set_xlabel(r"$R$", fontsize=14)
        ax.set_ylabel(r"$\Gamma$, $\Gamma_0\beta$", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        tick_params = {"axis": 'both', "which": 'both', "labelleft": True, 'labelright': False, "tick1On": True,
                       "tick2On": False, "labelsize": 11, "direction": 'in', "bottom": True, "top": True, "left": True,
                       "right": False}
        ax.tick_params(**tick_params)
        ax.set_ylim(1e0, 2e3)
        ax.set_xlim(1e8, 1e30)

        # ax.axvline(x=rdec, color="gray", linestyle=":", label=r"$R_{dec}$")

        legend = {"fancybox": False, "loc": 'lower left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax.legend(**legend)

        ax2 = ax.twinx()
        ax2.plot(o_nava.vals["R"], o_nava.vals["rho"], color="gray", ls='-', label=r"$\rho$ [g/cm$^{3}$]")
        # ax2.plot(o_peer.out_fs_arr["R"], o_peer.out_fs_arr["m"], color="gray", ls='--', label=r"$m$ [P]")
        ax2.set_xlabel(r"$R$", fontsize=14)
        ax2.set_ylabel(r"$\rho_{\rm CBM}$", fontsize=14)  # r"$\rho_{\rm CBM}$"
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        tick_params = {"axis": 'both', "which": 'both', "labelleft": False, 'labelright': True, "tick1On": False,
                       "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True, "left": False,
                       "right": True}
        ax2.tick_params(**tick_params)

        legend = {"fancybox": False, "loc": 'center left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax2.legend(**legend)

        # dic = {"left" : 0.16, "bottom" : 0.12, "top" : 0.98, "right" : 0.95, "hspace" : 0}
        # plt.subplots_adjust(**dic)
        # plt.tight_layout()
        # plt.savefig("./output/dynamics_rho.png")
        plt.tight_layout()
        plt.savefig(figures_dir + "stepdens_fs_comp_nonspread_noloss.png")
        plt.show()

    # wind-like density, 2 __models, Gamma and mass
    def task_plot_dyn_evol_non_uniform_cmb(self):
        # source parameters # shock parameters
        Gamma0 = 1000
        theta0 = np.pi / 180. * 5  # np.pi/2.
        E0 = 1e55 * (1 - np.cos(theta0)) / 2.
        # Grids
        Rstart, Rend, nR = 8., 30., 1001
        r_grid = np.logspace(Rstart, Rend, nR)
        # CBM
        A0 = 1e33
        s = 2
        dens_pars = (None, A0, s, 0, np.inf)

        aa = 1

        o_peer = evolove_driver(
            driver=Driver_Peer_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(None, A0, s, 0, np.inf),
            # kwargs
            useSpread=False, aa=-1., ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )

        o_nava = evolove_driver(
            driver=Driver_Nava_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(None, A0, s, 0, np.inf),
            # kwargs
            useSpread=False, aa=-1., ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )


        # --- run dynamics
        # o_peer = evolove_driver(
        #     driver=Driver_Peer_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=dens_pars, aa=aa, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )
        #
        # o_nava = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=dens_pars, aa=aa, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )
        # o_peer = EvolveShellPeer(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(None, A0, s, 0, np.inf),
        #                          ncells=None, p=p, eB=eB, epsilone=epsilone, useSpread=False, aa=aa)
        #
        # o_nava = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart, Rend, nR), r_grid=None, dens_pars=(None, A0, s, 0, np.inf),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)

        # ---
        M0 = E0 * cgs.c ** -2 / Gamma0
        rdec = get_Rdec(M0, A0 * (cgs.me + cgs.mp), Gamma0, s)

        subplots = {"ncols": 1, "nrows": 1, "figsize": (6.2, 4.6), "sharex": "all"}
        fig, ax = plt.subplots(**subplots)
        ax.plot(o_peer.vals["R"], o_peer.vals["Gamma"], color="green", ls='-', label=r"$\Gamma$ [P]")
        ax.plot(o_nava.vals["R"], o_nava.vals["Gamma"], color="blue", ls='-', label=r"$\Gamma$ [N]")
        ax.plot(o_peer.vals["R"], o_peer.vals["Gamma"][0] * o_peer.vals["beta"], color="green", ls='--',
                label=r"$\Gamma_0 \beta$")
        ax.plot(o_nava.vals["R"], o_nava.vals["Gamma"][0] * o_nava.vals["beta"], color="blue", ls='--',
                label=r"$\Gamma_0 \beta$")
        ax.set_xlabel(r"$R$", fontsize=14)
        ax.set_ylabel(r"$\Gamma$, $\Gamma_0\beta$", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        tick_params = {"axis": 'both', "which": 'both', "labelleft": True, 'labelright': False, "tick1On": True,
                       "tick2On": False, "labelsize": 11, "direction": 'in', "bottom": True, "top": True, "left": True,
                       "right": False}
        ax.tick_params(**tick_params)
        ax.set_ylim(1e0, 2e3)
        ax.set_xlim(1e8, 1e30)

        ax.axvline(x=rdec, color="gray", linestyle=":", label=r"$R_{dec}$")

        legend = {"fancybox": False, "loc": 'lower left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax.legend(**legend)

        ax2 = ax.twinx()
        ax2.plot(o_nava.vals["R"], o_nava.vals["M2"], color="gray", ls='-', label=r"$m$ [N]")
        ax2.plot(o_peer.vals["R"], o_peer.vals["M2"], color="gray", ls='--', label=r"$m$ [P]")
        ax2.set_xlabel(r"$R$", fontsize=14)
        ax2.set_ylabel(r"$m$ [g]", fontsize=14)  # r"$\rho_{\rm CBM}$"
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        tick_params = {"axis": 'both', "which": 'both', "labelleft": False, 'labelright': True, "tick1On": False,
                       "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True, "left": False,
                       "right": True}
        ax2.tick_params(**tick_params)

        legend = {"fancybox": False, "loc": 'center left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax2.legend(**legend)

        # dic = {"left" : 0.16, "bottom" : 0.12, "top" : 0.98, "right" : 0.95, "hspace" : 0}
        # plt.subplots_adjust(**dic)
        # plt.tight_layout()
        # plt.savefig("./output/dynamics_rho.png")
        plt.tight_layout()
        plt.savefig(figures_dir + "nonunif_fs_comp_nonspread_noloss.png")
        plt.show()


class TestDynUniformCBM:

    # Uniform density, 2 __models, Self-Similar solutions
    def task_plot_dyn_evol_spreading(self):
        # source parameters # shock parameters
        Gamma0 = 1000
        theta0 = np.pi / 180. * 5  # np.pi/2.
        E0 = 1e55 * (1 - np.cos(theta0)) / 2.
        # Grids
        Rstart, Rend, nR = 8., 22., 1001
        r_grid = np.logspace(Rstart, Rend, nR)
        # CBM
        s = 0
        nCM = 1e-1

        aa = -1  # Sets some equtions

        # --- run dynamics
        o_nava_sp1 = evolove_driver(
            driver=Driver_Nava_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=True, aa=aa, ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="Adi", eq_dmdr="default", eq_gammaAdi="Peer", eq_rho2="transrel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )
        # o_nava_sp1 = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=True, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_Adi, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )
        aa = 1
        o_nava_sp2 = evolove_driver(
            driver=Driver_Nava_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=True, aa=aa, ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="AA", eq_dmdr="default", eq_gammaAdi="Peer", eq_rho2="transrel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )
        # o_nava_sp2 = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=True, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_AA, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )
        # aa = 0
        # o_nava_sp3 = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0,Gamma0=Gamma0,thetaE=0.,theta0=theta0,M0=None,r_grid_pars=(),r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=True, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_AA, thetaMax=np.pi/2.
        # )

        # o_nava_nosp1 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)
        # o_nava_sp1 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=True,
        #                          adiabaticLosses=True, aa=aa)

        # aa = 1
        # o_nava_nosp2 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)
        # o_nava_sp2 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=True,
        #                          adiabaticLosses=True, aa=aa)

        # aa = 1  # Sets some equtions
        # # --- run dynamics
        # o_peer = EvolveShellPeer(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, p=p, eB=eB, epsilone=epsilone, useSpread=True, aa=aa)

        # o_nava = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)

        # --- analytic arguments
        # R = np.logspace(np.log10(Rstart), np.log10(Rend), nR)
        # M0 = E0 * cgs.c ** -2 / Gamma0
        # # rdec = Equations.get_Rdec(M0, nCM * (cgs.me + cgs.mp), Gamma0, s)
        # rdec = Equations.get_Rdec2(E0, nCM, Gamma0)
        # mask = R > rdec
        # # tdec = o_dyn.tobs[find_nearest_index(o_dyn.R, rdec)]
        # Gamma_bm = get_bm79(E0, nCM * (cgs.me + cgs.mp), s, R)
        # delta_R_bm = R[find_nearest_index(o_nava.out_fs_arr["Gamma"], 0.1 * Gamma0)] / \
        #              R[find_nearest_index(Gamma_bm, 0.1 * Gamma0)]
        #
        # beta0 = Equations.beta(Gamma0)
        # beta_st = get_sedovteylor(rdec, beta0, o_nava.out_fs_arr["R"])
        # delta_R_st = R[find_nearest_index(o_nava.out_fs_arr["beta"], 0.1 * beta0)] / \
        #              R[find_nearest_index(beta_st, 0.1 * beta0)]

        # --- plot

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6.2, 4.6), sharex="all")
        ip = 0

        # ax.plot(R[mask] * delta_R_bm, Gamma_bm[mask], color="black", ls='-', label=r"$\Gamma_{\rm BM79}$")

        # ax.plot(R[mask] * delta_R_st, Gamma0 * beta_st[mask], color="black", ls='--', label=r"$\Gamma_0 \beta_{\rm ST}$")

        # ax.plot(o_nava_nosp1.out_fs_arr["R"], o_nava_nosp1.out_fs_arr["Gamma"], color="black", ls='-')  # , label=r"$\Gamma$ [P]")
        # ax.plot(o_nava_nosp2.out_fs_arr["R"], o_nava_nosp2.out_fs_arr["Gamma"], color="gray", ls='-')  # , label=r"$\Gamma$ [P]")
        ax.plot(o_nava_sp1.vals["R"], o_nava_sp1.vals["Gamma"], color="green", ls='-')  # , label=r"$\Gamma$ [N]")
        ax.plot(o_nava_sp2.vals["R"], o_nava_sp2.vals["Gamma"], color="red", ls='-')  # , label=r"$\Gamma$ [N]")
        # # # ax.plot(o_nava_sp3.get("R"), o_nava_sp3.get("Gamma"), color="orange", ls='-')
        # ax.plot(o_peer.out_fs_arr["R"], o_peer.out_fs_arr["Gamma"], color="gray", ls='-')#, label=r"$\Gamma$ [P]")

        # ax.plot(o_nava.out_fs_arr["R"], o_nava.out_fs_arr["Gamma"], color="green", ls='-')#, label=r"$\Gamma$ [N]")
        # ax.plot(o_peer_.out_fs_arr["R"], o_peer_.out_fs_arr["Gamma"], color="red", ls='-')#, label=r"$\Gamma$ [N]")

        # ax.plot(o_nava_nosp1.out_fs_arr["R"], Gamma0 * o_nava_nosp1.out_fs_arr["beta"], color="black", ls='--')#, label=r"$\Gamma_0 \beta$ [P]")
        # ax.plot(o_nava_nosp2.out_fs_arr["R"], Gamma0 * o_nava_nosp2.out_fs_arr["beta"], color="gray", ls='--')  # , label=r"$\Gamma_0 \beta$ [P]")
        ax.plot(o_nava_sp1.vals["R"], Gamma0 * o_nava_sp1.vals["beta"], color="green", ls='--')  # , label=r"$\Gamma_0 \beta$ [N]")
        ax.plot(o_nava_sp2.vals["R"], Gamma0 * o_nava_sp2.vals["beta"], color="red", ls='--')  # , label=r"$\Gamma_0 \beta$ [N]")
        # # # ax.plot(o_nava_sp3.get("R"), Gamma0 * o_nava_sp3.get("beta"), color="orange", ls='--')#, label=r"$\Gamma_0 \beta$ [N]")

        # ax.plot(o_peer.out_fs_arr["R"], Gamma0 * o_peer.out_fs_arr["beta"], color="gray", ls='--')#, label=r"$\Gamma$ [P]")

        ax.plot([-10. - 20], [-20. - 30], color='gray', ls='-', label=r"$\Gamma$")
        ax.plot([-10. - 20], [-20. - 30], color='gray', ls=':', label=r"$\theta$")
        ax.plot([-10. - 20], [-20. - 30], color='gray', ls='--', label=r"$\Gamma_0 \beta$")
        ax.plot([-10. - 20], [-20. - 30], color='green', ls='-', label=r"$d\theta/dr \neq f(\theta)$")
        ax.plot([-10. - 20], [-20. - 30], color='red', ls='-', label=r"$d\theta/dr = f(\theta)$ $a=1$")
        # # # ax.plot([-10. - 20], [-20. - 30], color='orange', ls='-', label=r"$d\theta/dr = f(\theta)$ $a=0$")

        # ax.axvline(x=rdec, color="gray", linestyle=":", label=r"$R_{dec}$")
        ax.set_xlabel(r"$R$ [cm]", fontsize=12)
        ax.set_ylabel(r"$\Gamma$, $\Gamma_0\beta$", fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(**{"axis": 'both', "which": 'both', "labelleft": True, 'labelright': False, "tick1On": True,
                          "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True,
                          "left": True,
                          "right": True})
        legend = {"fancybox": False, "loc": 'center left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax.legend(**legend)
        ax.set_ylim(1e0, 2e3)
        ax.set_xlim(5e16, 1e21)

        ax2 = ax.twinx()
        ax2.plot(o_nava_sp1.vals["R"], o_nava_sp1.vals["theta"] * 180 / np.pi, color="green", ls=':')  # , label=r"$m$ [N]")
        ax2.plot(o_nava_sp2.vals["R"], o_nava_sp2.vals["theta"] * 180 / np.pi, color="red", ls=':')  # , label=r"$m$ [P]")
        # # # ax2.plot(o_nava_sp3.get("R"), o_nava_sp3.get("theta")*180/np.pi, color="orange", ls=':')#, label=r"$m$ [P]")

        # ax2.plot(o_peer.out_fs_arr["R"], o_peer.out_fs_arr["theta"]*180/np.pi, color="gray", ls=':')#, label=r"$\Gamma$ [P]")
        ax2.set_xlabel(r"$R$", fontsize=14)
        ax2.set_ylabel(r"$\theta$ [deg]", fontsize=14)  # r"$\rho_{\rm CBM}$"
        ax2.set_xscale("log")
        ax2.set_yscale("linear")
        tick_params = {"axis": 'both', "which": 'both', "labelleft": False, 'labelright': True, "tick1On": False,
                       "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True, "left": False,
                       "right": True}
        ax2.tick_params(**tick_params)
        ax2.minorticks_on()

        legend = {"fancybox": False, "loc": 'center left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax2.legend(**legend)

        # ax.text(0.05, 0.7, r"$\Gamma_0=1000$" + "\n" +
        #                   "$n=0.1$" + "\n" + "$E_0=10^{55}$", color="black", fontsize=12,
        #               transform=ax.transAxes)

        dic = {"left": 0.16, "bottom": 0.12, "top": 0.98, "right": 0.92, "hspace": 0}
        plt.subplots_adjust(**dic)
        plt.savefig(figures_dir + "unif_fs_comp_spread_noloss.png")
        # plt.tight_layout()
        plt.show()

    # Uniform density, 2 __models, Self-Similar solutions
    def task_plot_dyn_evol_and_bm97_st_solutions(self):
        # source parameters # shock parameters
        Gamma0 = 1000
        theta0 = np.pi / 180. * 5  # np.pi/2.
        E0 = 1e55 * (1 - np.cos(theta0)) / 2.
        # Grids
        Rstart, Rend, nR = 8., 22., 1001
        r_grid = np.logspace(Rstart, Rend, nR)
        # CBM
        s = 0
        nCM = 1e-1

        aa = -1  # Sets some equtions

        # --- run dynamics
        o_peer = evolove_driver(
            driver=Driver_Peer_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=aa, ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Peer", eq_rho2="transrel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )
        # o_peer = evolove_driver(
        #     driver=Driver_Peer_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )

        o_nava = evolove_driver(
            driver=Driver_Nava_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=aa, ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Peer", eq_rho2="transrel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )
        # o_nava = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )

        aa = 1  # Sets some equtions
        # --- run dynamics
        o_peer_ = evolove_driver(
            driver=Driver_Peer_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=aa, ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Peer", eq_rho2="transrel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )
        # o_peer_ = evolove_driver(
        #     driver=Driver_Peer_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=False, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )

        # --- analytic arguments
        R = np.logspace(Rstart, Rend, nR)
        M0 = E0 * cgs.c ** -2 / Gamma0
        # rdec = Equations.get_Rdec(M0, nCM * (cgs.me + cgs.mp), Gamma0, s)
        rdec = get_Rdec2(E0, nCM, Gamma0)
        mask = R > rdec
        # tdec = o_dyn.tobs[find_nearest_index(o_dyn.R, rdec)]
        Gamma_bm = get_bm79(E0, nCM * (cgs.me + cgs.mp), s, R)
        delta_R_bm = R[find_nearest_index(o_nava.vals["Gamma"], 0.1 * Gamma0)] / \
                     R[find_nearest_index(Gamma_bm, 0.1 * Gamma0)]

        beta0 = get_beta(Gamma0)
        beta_st = get_sedovteylor(rdec, beta0, o_nava.vals["R"])
        delta_R_st = R[find_nearest_index(o_nava.vals["beta"], 0.1 * beta0)] / \
                     R[find_nearest_index(beta_st, 0.1 * beta0)]

        # --- plot

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6.2, 4.6), sharex="all")
        ip = 0

        ax.plot(R[mask], np.full_like(R[mask], Gamma0), color="black", ls=":", lw=1.2)
        ax.plot(R[mask] * delta_R_bm, Gamma_bm[mask], color="black", ls='-', label=r"$\Gamma_{\rm BM79}$")

        ax.plot(R[mask] * delta_R_st, Gamma0 * beta_st[mask], color="black", ls='--',
                label=r"$\Gamma_0 \beta_{\rm ST}$")

        ax.plot(o_peer.vals["R"], o_peer.vals["Gamma"], color="blue", ls='-')  # , label=r"$\Gamma$ [P]")
        ax.plot(o_nava.vals["R"], o_nava.vals["Gamma"], color="green", ls='-')  # , label=r"$\Gamma$ [N]")
        ax.plot(o_peer_.vals["R"], o_peer_.vals["Gamma"], color="red", ls='-')  # , label=r"$\Gamma$ [N]")

        ax.plot(o_peer.vals["R"], Gamma0 * o_peer.vals["beta"], color="blue", ls='--')  # , label=r"$\Gamma_0 \beta$ [P]")
        ax.plot(o_nava.vals["R"], Gamma0 * o_nava.vals["beta"], color="green",
                ls='--')  # , label=r"$\Gamma_0 \beta$ [N]")
        ax.plot(o_peer_.vals["R"], Gamma0 * o_peer_.vals["beta"], color="red",
                ls='--')  # , label=r"$\Gamma_0 \beta$ [N]")

        ax.plot([-10. - 20], [-20. - 30], color='gray', ls='-', label=r"$\Gamma$")
        ax.plot([-10. - 20], [-20. - 30], color='gray', ls='--', label=r"$\Gamma_0 \beta$")
        ax.plot([-10. - 20], [-20. - 30], color='blue', ls='-', label=r"Peer+12")
        ax.plot([-10. - 20], [-20. - 30], color='green', ls='-', label=r"Nava+13")
        ax.plot([-10. - 20], [-20. - 30], color='red', ls='-', label=r"Peer+12 & $dm/dr = f(\Gamma)$")

        ax.axvline(x=rdec, color="gray", linestyle=":", label=r"$R_{dec}$")
        ax.set_xlabel(r"$R$ [cm]", fontsize=12)
        ax.set_ylabel(r"$\Gamma$, $\Gamma_0\beta$", fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(**{"axis": 'both', "which": 'both', "labelleft": True, 'labelright': False, "tick1On": True,
                          "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True,
                          "left": True,
                          "right": True})
        legend = {"fancybox": False, "loc": 'lower left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax.legend(**legend)
        ax.set_ylim(1e0, 2e3)
        ax.set_xlim(1e16, 2e21)

        # ax.text(0.05, 0.7, r"$\Gamma_0=1000$" + "\n" +
        #                   "$n=0.1$" + "\n" + "$E_0=10^{55}$", color="black", fontsize=12,
        #               transform=ax.transAxes)

        dic = {"left": 0.16, "bottom": 0.12, "top": 0.98, "right": 0.95, "hspace": 0}
        plt.subplots_adjust(**dic)
        plt.savefig(figures_dir + "unif_fs_comp_nonspread_noloss.png")
        # plt.tight_layout()
        plt.show()

    # uniform density 3 models, test other quantities
    def task_plot_dyn_evol_other_quantities(self, v_n = "U_e", v_n_RS="U_e_RS"):
        # source parameters # shock parameters
        Gamma0 = 1000.
        theta0 = np.pi / 180. * 5  # np.pi/2.
        E0 = 1e55 * (1 - np.cos(theta0)) / 2.
        # Grids
        Rstart, Rend, nR = 8., 22., 1001
        r_grid = np.logspace(Rstart, Rend, nR)
        # CBM
        s = 0
        nCM = 1.e-1

        aa = -1  # Sets some equtions

        # --- run dynamics
        o_nava_sp1 = evolove_driver(
            driver=Driver_Nava_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=aa, ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Peer", eq_rho2="transrel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )
        # o_nava_sp1 = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=True, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_Adi, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )

        o_peer_sp2 = evolove_driver(
            driver=Driver_Peer_FS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=aa, ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-5, "nsteps": 1000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Peer", eq_rho2="transrel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0.,
        )

        o_nava_sp2 = evolove_driver(
            driver=Driver_Nava_FSRS,
            E0=E0,
            Gamma0=Gamma0,
            thetaE=0.,
            theta0=theta0,
            M0=None,
            r_grid_pars=(),
            r_grid=r_grid,
            dens_pars=(nCM, None, None, None, None),
            # kwargs
            useSpread=False, aa=aa, ncells=1, ode='dop853',
            ode_pars={"rtol": 1e-8, "nsteps": 3000, "first_step": Rstart},
            eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Peer", eq_rho2="transrel",
            thetaMax=np.pi / 2.,
            adiabLoss=True, epsilon_e_rad=0., epsilon_e_rad_RS=0., tprompt=1e3, adiabLoss_RS=True
        )

        # o_nava_sp2 = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None, r_grid_pars=(),
        #     r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=True, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_AA, thetaMax=np.pi / 2.,
        #     eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel
        # )
        # aa = 0
        # o_nava_sp3 = evolove_driver(
        #     driver=Driver_Nava_FS, E0=E0,Gamma0=Gamma0,thetaE=0.,theta0=theta0,M0=None,r_grid_pars=(),r_grid=r_grid,
        #     dens_pars=(nCM, None, None, None, None), aa=aa, useSpread=True, adiabLoss=True, ncells=1,
        #     ode_rtol=1e-3, ode_nsteps=1000, epsilon_e_rad=0., eq_dthetadr=EqOpts.dthetadr_AA, thetaMax=np.pi/2.
        # )

        # o_nava_nosp1 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)
        # o_nava_sp1 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=True,
        #                          adiabaticLosses=True, aa=aa)

        # aa = 1
        # o_nava_nosp2 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)
        # o_nava_sp2 = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=True,
        #                          adiabaticLosses=True, aa=aa)

        # aa = 1  # Sets some equtions
        # # --- run dynamics
        # o_peer = EvolveShellPeer(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, p=p, eB=eB, epsilone=epsilone, useSpread=True, aa=aa)

        # o_nava = EvolveShellNava(E0=E0, Gamma0=Gamma0, thetaE=0., theta0=theta0, M0=None,
        #                          r_grid_pars=(Rstart,Rend,nR), r_grid=None, dens_pars=(nCM,None,None,None,None),
        #                          ncells=None, tprompt=None, p=p, epsilone=epsilone, eB=eB, useSpread=False,
        #                          adiabaticLosses=True, aa=aa)

        # --- analytic arguments
        # R = np.logspace(np.log10(Rstart), np.log10(Rend), nR)
        # M0 = E0 * cgs.c ** -2 / Gamma0
        # # rdec = Equations.get_Rdec(M0, nCM * (cgs.me + cgs.mp), Gamma0, s)
        # rdec = Equations.get_Rdec2(E0, nCM, Gamma0)
        # mask = R > rdec
        # # tdec = o_dyn.tobs[find_nearest_index(o_dyn.R, rdec)]
        # Gamma_bm = get_bm79(E0, nCM * (cgs.me + cgs.mp), s, R)
        # delta_R_bm = R[find_nearest_index(o_nava.out_fs_arr["Gamma"], 0.1 * Gamma0)] / \
        #              R[find_nearest_index(Gamma_bm, 0.1 * Gamma0)]
        #
        # beta0 = Equations.beta(Gamma0)
        # beta_st = get_sedovteylor(rdec, beta0, o_nava.out_fs_arr["R"])
        # delta_R_st = R[find_nearest_index(o_nava.out_fs_arr["beta"], 0.1 * beta0)] / \
        #              R[find_nearest_index(beta_st, 0.1 * beta0)]

        # --- plot

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6.2, 4.6), sharex="all")
        ip = 0

        # ax.plot(R[mask] * delta_R_bm, Gamma_bm[mask], color="black", ls='-', label=r"$\Gamma_{\rm BM79}$")

        # ax.plot(R[mask] * delta_R_st, Gamma0 * beta_st[mask], color="black", ls='--', label=r"$\Gamma_0 \beta_{\rm ST}$")

        # ax.plot(o_nava_nosp1.out_fs_arr["R"], o_nava_nosp1.out_fs_arr["Gamma"], color="black", ls='-')  # , label=r"$\Gamma$ [P]")
        # ax.plot(o_nava_nosp2.out_fs_arr["R"], o_nava_nosp2.out_fs_arr["Gamma"], color="gray", ls='-')  # , label=r"$\Gamma$ [P]")
        ax.loglog(o_nava_sp1.vals["R"], o_nava_sp1.vals[v_n], color="green", ls='-', label='Nava')  # , label=r"$\Gamma$ [N]")
        ax.loglog(o_peer_sp2.vals["R"], o_peer_sp2.vals[v_n], color="red", ls='-', label='Peer')  # , label=r"$\Gamma$ [N]")
        ax.loglog(o_nava_sp2.vals["R"], o_nava_sp2.vals[v_n_RS], color="blue", ls='--', label='Nava FSRS')  # , label=r"$\Gamma_0 \beta$ [N]")


        #
        # # ax.plot([-10. - 20], [-20. - 30], color='gray', ls='-', label=r"$\Gamma$")
        # # ax.plot([-10. - 20], [-20. - 30], color='gray', ls=':', label=r"$\theta$")
        # # ax.plot([-10. - 20], [-20. - 30], color='gray', ls='--', label=r"$\Gamma_0 \beta$")
        # # ax.plot([-10. - 20], [-20. - 30], color='green', ls='-', label=r"$d\theta/dr \neq f(\theta)$")
        # # ax.plot([-10. - 20], [-20. - 30], color='red', ls='-', label=r"$d\theta/dr = f(\theta)$ $a=1$")
        # # # # ax.plot([-10. - 20], [-20. - 30], color='orange', ls='-', label=r"$d\theta/dr = f(\theta)$ $a=0$")
        #
        # # ax.axvline(x=rdec, color="gray", linestyle=":", label=r"$R_{dec}$")
        # ax.set_xlabel(r"$R$ [cm]", fontsize=12)
        # ax.set_ylabel(r"$\Gamma$, $\Gamma_0\beta$", fontsize=12)
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.tick_params(**{"axis": 'both', "which": 'both', "labelleft": True, 'labelright': False, "tick1On": True,
        #                   "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True,
        #                   "left": True,
        #                   "right": True})
        legend = {"fancybox": False, "loc": 'center left',
                  # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing": 0.4,
                  "framealpha": 0., "borderaxespad": 0., "frameon": False}
        ax.legend(**legend)
        # ax.set_ylim(1e0, 2e3)
        # ax.set_xlim(5e16, 1e21)
        #
        # ax2 = ax.twinx()
        # ax2.plot(o_nava_sp1.vals["R"], o_nava_sp1.vals["theta"] * 180 / np.pi, color="green", ls=':')  # , label=r"$m$ [N]")
        # ax2.plot(o_nava_sp2.vals["R"], o_nava_sp2.vals["theta"] * 180 / np.pi, color="red", ls=':')  # , label=r"$m$ [P]")
        # # # # ax2.plot(o_nava_sp3.get("R"), o_nava_sp3.get("theta")*180/np.pi, color="orange", ls=':')#, label=r"$m$ [P]")
        #
        # # ax2.plot(o_peer.out_fs_arr["R"], o_peer.out_fs_arr["theta"]*180/np.pi, color="gray", ls=':')#, label=r"$\Gamma$ [P]")
        # ax2.set_xlabel(r"$R$", fontsize=14)
        # ax2.set_ylabel(r"$\theta$ [deg]", fontsize=14)  # r"$\rho_{\rm CBM}$"
        # ax2.set_xscale("log")
        # ax2.set_yscale("linear")
        # tick_params = {"axis": 'both', "which": 'both', "labelleft": False, 'labelright': True, "tick1On": False,
        #                "tick2On": True, "labelsize": 11, "direction": 'in', "bottom": True, "top": True, "left": False,
        #                "right": True}
        # ax2.tick_params(**tick_params)
        # ax2.minorticks_on()
        #
        # legend = {"fancybox": False, "loc": 'center left',
        #           # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #           "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
        #           "framealpha": 0., "borderaxespad": 0., "frameon": False}
        # ax2.legend(**legend)
        #
        # # ax.text(0.05, 0.7, r"$\Gamma_0=1000$" + "\n" +
        # #                   "$n=0.1$" + "\n" + "$E_0=10^{55}$", color="black", fontsize=12,
        # #               transform=ax.transAxes)
        #
        dic = {"left": 0.16, "bottom": 0.12, "top": 0.98, "right": 0.92, "hspace": 0}
        plt.subplots_adjust(**dic)
        # plt.savefig(figures_dir + "unif_fs_comp_spread_noloss.png")
        # plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # o_test = TestDynWithRadLoss()
    # o_test.test_plot_dyn_evol_nonspread_radloss()
    #
    # o_test = TestDynNonUniformCBM()
    # o_test.task_plot_evold_unif_rs()
    # o_test.task_plot_dyn_evol_step_dens_prof_steep()
    # o_test.task_plot_dyn_evol_non_uniform_cmb()

    o_test = TestDynUniformCBM()
    # o_test.task_plot_dyn_evol_spreading()
    # o_test.task_plot_dyn_evol_and_bm97_st_solutions()
    o_test.task_plot_dyn_evol_other_quantities(v_n='U_e', v_n_RS='U_e_RS')