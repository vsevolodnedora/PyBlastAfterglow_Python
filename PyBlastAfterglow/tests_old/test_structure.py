"""

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PyBlastAfterglow.uutils import cgs
from PyBlastAfterglow.structure import Structure_Angular
from PyBlastAfterglow.dynamics import Driver_Nava_FS, Driver_Peer_FS, EqOpts

package_dir = Path(__file__).parent.parent.parent
# where to read sampled files
data_dir = f"{package_dir}/data"
# where to save figures
figures_dir = f"{data_dir}/crosscheck_figures/structure/"

class TestStruct():
    def uniform_jet_layer_structure(self):
        EE_l = 10 ** (52.4)  # Peak core energy
        GamC_l = 666.  # Peak core Lorentz factor
        thetaCore_l = 0.09  # np.pi/2. np.pi / 2.
        thetaJet_l = 15. * cgs.pi / 180  # * .5 # joAngle -> jet theta openning

        gauss_struct = Structure_Angular.from_analytic_pars(
            nlayers=100,
            EEc0=EE_l,
            Gamc0=GamC_l,
            theta0j=thetaJet_l,
            theta0c=thetaCore_l,
            kk=0.,
            structure='gaussian'
        )

        pl_struct = Structure_Angular.from_analytic_pars(
            nlayers=100,
            EEc0=EE_l,
            Gamc0=GamC_l,
            theta0j=thetaJet_l,
            theta0c=thetaCore_l,
            kk=2.,
            structure='power-law'
        )

        fig, axes = plt.subplots(nrows=1, ncols=1)  # , figsize=(5,8))
        ax = axes

        # ax.plot(np.full_like(jet.struct.theta0c, jet.RRs[0]), jet.struct.theta0c, marker='x', color='black')
        ax.axvline(x=thetaCore_l, linestyle="dotted", color="green", label=r"$\theta_с$")
        ax.axvline(x=thetaJet_l, linestyle="dotted", color="orange", label=r"$\theta_j$")

        ax.set_title("Gaussian jet initial structure")

        ax.plot(gauss_struct.cthetas0, gauss_struct.cell_Gam0s, color="blue", ls='-')
        ax.plot(pl_struct.cthetas0, pl_struct.cell_Gam0s, color="blue", ls='--')
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\Gamma_0$")
        ax.tick_params(axis='y', labelcolor="blue")
        ax.minorticks_on()

        ax.plot([0, 0], [0, 0], color='gray', ls='-', label=r"Gaussia")
        ax.plot([0, 0], [0, 0], color='gray', ls='--', label=r"Power-law $k=2$")

        # ax = axes[1]
        ax1 = ax.twinx()

        # ax.plot(np.full_like(jet.struct.theta0c, jet.RRs[0]), jet.struct.theta0c, marker='x', color='black')
        # ax1.axvline(x=thetaCore_l, linestyle="dashed", color="green", label=r"$\theta_с$", ls='--')
        # ax1.axvline(x=thetaJet_l, linestyle="dashed", color="orange", label=r"$\theta_j$", ls='--')

        # ax1.set_title("Gaussian jet initial structure")

        ax1.plot(gauss_struct.cthetas0, gauss_struct.cell_EEs, color="red", ls='-')
        ax1.plot(pl_struct.cthetas0, pl_struct.cell_EEs, color="red", ls='--')
        # ax.set_xscale("log")
        ax1.set_yscale("log")
        # ax1.set_xlabel(r"$\theta_0$")
        # ax1.set_ylabel(r"$\Gamma_0$")
        ax1.minorticks_on()

        # f1 = lambda gam_to_e: interpolate.interp1d(jet.struct.cell_Gam0s, np.log10(jet.struct.cell_EEs),
        #                                            fill_value="extrapolate")(gam_to_e)
        # f2 = lambda loge_to_gam: interpolate.interp1d(np.log10(jet.struct.cell_EEs), jet.struct.cell_Gam0s,
        #                                               fill_value="extrapolate")(loge_to_gam)
        # secaxy = ax.secondary_yaxis('right', functions=(f1, f2))

        # ax2 = ax.twinx()
        # ax2.plot(jet.struct.cthetas0, np.log10(jet.struct.cell_EEs), color="red")
        # ax2.set_xscale("log")
        ax1.set_yscale("log")
        # secaxy.set_xlabel("R [cm]")
        ax1.set_ylabel(r"$E_0$ [erg]")
        ax1.tick_params(axis='y', labelcolor="red")
        # secaxy.minorticks_on()

        ax.legend()

        plt.savefig(figures_dir + "test_jet_init_struct.png")
        plt.show()


# def uniform_jet_layer_structure_spread(
#         save = "test_jet_struct_evol_stread.png"
# ):
#
#     thetaC = 0.1 # Half-opening angle in radians
#     thetaW = 0.1 # Truncation angle, unused for top-hat
#     E0 = 1e52 # Isotropic-equivalent energy in erg
#     n0 = 1e-3 # circumburst density in cm^{-3}
#
#     jet = StructuredJet.from_analytic_pars(
#         nlayers=100,
#         EEc0=E0,
#         Gamc0=150.,
#         theta0j=thetaW,
#         thetaCo=thetaC,
#         kk=1.,
#         structure='uniform',
#         r_pars=(8., 22., 1000),
#         dens_pars=(n0, None, None, None, None),
#         driver=(Driver_Peer_FS,{"aa": 1, "useSpread": True, "epsilon_e_rad": 0, "adiabLoss": True,
#                        "ode_rtol": 1e-3, "ode_nsteps": 3000,
#                        "eq_dthetadr":EqOpts.dthetadr_AA, "thetaMax":np.pi/2}),
#         electrons=(None, {}),
#         synchrotron=(None, {}),
#         eats=(None, {})
#     )
#
#     layers = [0, 25, 50, 75, 99]
#     colors = ["blue", "red", "green", "orange", "black"]
#
#     fig, axes = plt.subplots(nrows=1, ncols=1)
#     ax = axes
#
#     # ax.plot(np.full_like(jet.struct.theta0c, jet.RRs[0]), jet.struct.theta0c, marker='x', color='black')
#
#     ax.axhline(y = thetaW, linestyle="dotted", color="gray", label=r"$\theta_w$={:.2f}".format(thetaW))
#
#     for ll, cc in zip(layers, colors):
#         # ax.plot(jet.shells[ll].dyn.get("R"), jet.shells[ll].dyn.get("theta"), label="ll={}".format(ll))
#         ax.plot(jet.shells[ll].dyn.get("R")[1:], jet.cthetas[ll], label=r"layer={:d} $\theta_0$={:.2f}".format(ll, jet.cthetas[ll][0]))
#         # ax.plot(jet.shells[ll].dyn.get("R"), jet.shells[ll].dyn.get("Gamma"), label="ll={}".format(ll))
#
#     ax.set_xscale("log")
#     ax.set_xlabel("R [cm]")
#     ax.set_ylabel(r"$\theta$")
#     ax.legend()
#     ax.minorticks_on()
#     ax.grid(1)
#     if not save is None:
#         plt.savefig(__figpath__ + save)
#     plt.show()


if __name__ == '__main__':
    o_test = TestStruct()
    o_test.uniform_jet_layer_structure()


