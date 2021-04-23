"""

"""
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from PyBlastAfterglow.dynamics_old import Driver_Peer_FS, Driver_Nava_FS, Driver_Nava_FSRS, rho_dlnrho1dR, EqOpts
from PyBlastAfterglow.electrons_old import Electron_BPL, Electron_BPL_Accurate
from PyBlastAfterglow.uutils import cgs

package_dir = Path(__file__).parent.parent.parent
# where to read sampled files
data_dir = f"{package_dir}/data"
# where to save figures
figures_dir = f"{data_dir}/crosscheck_figures/electrons/"

class SimpleShell:

    def __init__(self):
        pass

class TestElectronDistEvol:

    def test_plot_evolution_of_electron_crit_lfs(self):
        RR = np.logspace(10., 22., 1000)
        n0 = 1e-1

        # print(63*1000 * cgs.c ** 2)
        # exit(1)
        shell = SimpleShell()
        shell.E0 = 1.e52
        shell.Gamma0 = 1000.
        shell.M0 = shell.E0 / (cgs.c ** 2 * shell.Gamma0)
        shell.thetaE = 0.
        shell.theta0 = np.pi / 2
        shell.Rstart = RR[0]
        shell.rho0 = n0 * cgs.mppme
        shell.dlnrho1dR_0 = 0.

        shell.eps_e = 1e-1
        shell.eps_b = 1e-2
        shell.p = 2.2

        shell.p_rs = 2.3
        shell.eps_e_rs = 1.5e-1
        shell.eps_b_rs = 1.5e-2

        shell.dyn = Driver_Nava_FS.from_obj(shell, aa=-1, useSpread=False, epsilon_e_rad=0, adiabLoss=True, tprompt=1e3,
                                            epsilon_e_rad_RS=0., adiabLoss_RS=True,
                                            ode_rtol=1e-4, ode_nsteps=3000, eq_dthetadr=EqOpts.dthetadr_None,
                                            thetaMax=np.pi / 2.,
                                            eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel,
                                            ncells=1)
        shell.dyn_rs = Driver_Nava_FSRS.from_obj(shell, aa=-1, useSpread=False, epsilon_e_rad=0, adiabLoss=True,
                                                 tprompt=1e3,
                                                 epsilon_e_rad_RS=0., adiabLoss_RS=True,
                                                 ode_rtol=1e-8, ode_nsteps=1000, eq_dthetadr=EqOpts.dthetadr_None,
                                                 thetaMax=np.pi / 2.,
                                                 eq_gammaAdi=EqOpts.gamma_adi_peer, eq_rhoprime=EqOpts.rho2_transrel,
                                                 ncells=1)
        # FS
        shell.electrons = Electron_BPL.from_obj_fs(shell.dyn, p=shell.p, eps_e=shell.eps_e, eps_b=shell.eps_b)
        shell.electrons2 = Electron_BPL_Accurate.from_obj_fs(shell.dyn, p=shell.p, eps_e=shell.eps_e, eps_b=shell.eps_b)
        # RS
        shell.electrons_rs = Electron_BPL.from_obj_rs(shell.dyn_rs, p=shell.p_rs, eps_e=shell.eps_e_rs,
                                                      eps_b=shell.eps_b_rs)
        shell.electrons_rs2 = Electron_BPL_Accurate.from_obj_rs(shell.dyn_rs, p=shell.p_rs, eps_e=shell.eps_e_rs,
                                                                eps_b=shell.eps_b_rs)

        spectra = []

        for i in range(1, len(RR)):
            rho, dlnrho1dR = rho_dlnrho1dR(RR[i], n0, None, None, None, None)

            ''' FS '''
            shell.dyn.evolove(RR[i], rho, dlnrho1dR)

            shell.electrons.compute_char_lfs_fs(shell.dyn)
            shell.electrons2.compute_char_lfs_fs(shell.dyn)

            ''' RS '''
            shell.dyn_rs.evolove(RR[i], rho, dlnrho1dR)

            shell.electrons_rs.compute_char_lfs_rs(shell.dyn_rs)
            shell.electrons_rs2.compute_char_lfs_rs(shell.dyn_rs)


        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        ''' --- | FS | --- '''

        ax = axes[0]
        ax.set_title("Forward Shock")
        # ax.plot(shell.dyn.get("R"), np.log10(shell.dyn.get("M2")), label="BRS")

        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons.get("gamma_min")), color="blue", label="gamma_min")
        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons.get("gamma_c")), color="green", label="gamma_c")
        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons.get("gamma_max")), color="red", label="gamma_max")
        #
        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons2.get("gamma_min")), color="blue", label="gamma_min 2",
                ls='--')
        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons2.get("gamma_c")), color="green", label="gamma_c 2",
                ls='--')

        # ax.plot(shell.dyn.get("R"), np.log10(shell.electrons2.get("gamma_max")), color="red", label="gamma_max 2",ls='--')

        # ax.set_yscale()
        ax.set_xscale("log")
        ax.set_xlabel("R [cm]")
        ax.set_ylabel(r"$\log(\gamma)$")
        ax.legend()
        ax.grid(1)
        ax.set_ylim(-0.05, 8.1)

        ''' --- | RS | --- '''

        ax = axes[1]
        ax.set_title("Reverse Shock")
        # ax.plot(shell.dyn_rs.get("R"), np.log10(shell.dyn_rs.get("rho4")), label="BRS")

        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons_rs.get("gamma_min")), color="blue", label="gamma_min")
        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons_rs.get("gamma_c")), color="green", label="gamma_c")
        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons_rs.get("gamma_max")), color="red", label="gamma_max")

        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons_rs2.get("gamma_min")), color="blue", label="gamma_min",
                ls='--')
        ax.plot(shell.dyn.get("R"), np.log10(shell.electrons_rs2.get("gamma_c")), color="green", label="gamma_c",
                ls='--')
        # ax.plot(shell.dyn.get("R"), np.log10(shell.electrons_rs2.get("gamma_max")), color="red", label="gamma_max",ls='--')

        ax.set_xscale("log")
        ax.set_xlabel("R [cm]")
        ax.grid(1)
        ax.set_ylim(-0.05, 8.1)

        plt.savefig(figures_dir + "electron_evolution_for_fsrs_system.png")
        plt.show()

if __name__ == '__main__':
    o_test = TestElectronDistEvol()
    o_test.test_plot_evolution_of_electron_crit_lfs()