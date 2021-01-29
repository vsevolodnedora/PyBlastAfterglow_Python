"""

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PyBlastAfterglow.uutils import cgs

from PyBlastAfterglow.electrons import BrokenPowerLaw
from PyBlastAfterglow.synchrotron import \
    Synchrotron_Joh06, Synchrotron_WSPN99, Synchrotron_DM06, freq_to_integrate, PhiPint
from PyBlastAfterglow.dynamics import Driver_Nava_FS, Driver_Peer_FS, EqOpts

package_dir = Path(__file__).parent.parent.parent
# where to read sampled files
data_dir = f"{package_dir}/data"
# where to save figures
figures_dir = f"{data_dir}/crosscheck_figures/synchrotron/"

class TestSyncrotron:

    def phi_p(self):
        p = 2.2
        phi_p_wspn = np.sqrt(3) * PhiPint(p)
        phi_p_joh_s = 2.234 * (0.54 + 0.08 * p)
        phi_p_joh_f = 11.17 * (p - 1) / (3 * p - 1) * (1.89 - 0.935 * p + 0.17 * p ** 2)
        print("p:{} WSPN phiP:{} Joh:{}[f] {} [s]".format(p, phi_p_wspn, phi_p_joh_s, phi_p_joh_f))
        print("!! Two models are different by 1.5 !!")

    def test_sed(self):
        ''' '''
        ncells = 1
        theta = np.pi
        R = 1e12
        Gamma = 120
        m2 = 1e20
        rho = 1e-1 * cgs.mppme
        rho2 = 4 * Gamma * rho
        delta_shock = m2 / (8. * np.pi * (1. - np.cos(theta) / ncells) * Gamma ** 2. * rho * R ** 2.)

        B = 1
        p = 2.2
        ''' --- sc --- '''

        gamma_min = 1e2
        gamma_c = 1e4

        ### Widges and SPN
        o_WSPN = Synchrotron_WSPN99(
            p,
            gamma_min,
            gamma_c,
            B,
            m2 / cgs.mppme,
            delta_shock,
            False
        )
        sc_nu_m_WSPN = o_WSPN.get("nu_m")
        sc_nu_c_WSPN = o_WSPN.get("nu_c")
        sc_pmax_WSPN = o_WSPN.get("pprimemax")
        sc_power_WSPN = o_WSPN.get_spectrum()

        ### Johanosn
        o_JOH = Synchrotron_Joh06(
            p,
            gamma_min,
            gamma_c,
            B,
            R,
            Gamma,
            rho2 / cgs.mppme,
            delta_shock,
            False
        )
        sc_nu_m_JOH = o_JOH.get("nu_m")
        sc_nu_c_JOH = o_JOH.get("nu_c")
        sc_pmax_JOH = o_JOH.get("pprimemax")
        sc_power_JOH = o_JOH.get_spectrum()

        ### Dermer
        bpl = BrokenPowerLaw.from_normalised_density(
            n_e_tot=rho2 / cgs.mppme,
            p1=p if gamma_min < gamma_c else 2.,
            p2=p + 1,
            gamma_b=gamma_c if gamma_min < gamma_c else gamma_min,
            gamma_min=gamma_min if gamma_min < gamma_c else gamma_c,
            gamma_max=1e6)

        o_DER = Synchrotron_DM06(
            bpl,
            bpl.parameters,
            delta_shock,
            B,
            R,
            Gamma,
            False,
            np.trapz
        )
        sc_power_DER = o_DER.get_spectrum()

        ''' --- fc --- '''

        gamma_min = 1e4
        gamma_c = 1e3

        ### Widges and SPN
        o_WSPN = Synchrotron_WSPN99(
            p,
            gamma_min,
            gamma_c,
            B,
            m2 / cgs.mppme,
            delta_shock,
            False
        )
        fc_nu_m_WSPN = o_WSPN.get("nu_m")
        fc_nu_c_WSPN = o_WSPN.get("nu_c")
        fc_pmax_WSPN = o_WSPN.get("pprimemax")
        fc_power_WSPN = o_WSPN.get_spectrum()

        ### Johanosn
        o_JOH = Synchrotron_Joh06(
            p,
            gamma_min,
            gamma_c,
            B,
            R,
            Gamma,
            rho2 / cgs.mppme,
            delta_shock,
            False
        )
        fc_nu_m_JOH = o_JOH.get("nu_m")
        fc_nu_c_JOH = o_JOH.get("nu_c")
        fc_pmax_JOH = o_JOH.get("pprimemax")
        fc_power_JOH = o_JOH.get_spectrum()

        ### Gao
        # fc_power_GAO = np.zeros(len(nu_arr))
        # for inu, nu in enumerate(nu_arr):
        #     fc_nu_m_GAO, fc_nu_c_GAO, fc_nu_a_GAO, fc_pmax_GAO, fc_power_GAO[inu] = \
        #         spectrum_GAO(nu, Ne, delta, gamma_min, gamma_c, B, p)

        ### Dermer
        ### Dermer
        bpl = BrokenPowerLaw.from_normalised_density(
            n_e_tot=rho2 / cgs.mppme,
            p1=p if gamma_min < gamma_c else 2.,
            p2=p + 1,
            gamma_b=gamma_c if gamma_min < gamma_c else gamma_min,
            gamma_min=gamma_min if gamma_min < gamma_c else gamma_c,
            gamma_max=1e6)

        o_DER = Synchrotron_DM06(
            bpl,
            bpl.parameters,
            delta_shock,
            B,
            R,
            Gamma,
            False,
            np.trapz
        )
        fc_power_DER = o_DER.get_spectrum()

        ''' --- '''

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey='all')

        # sc
        axes[0].set_title(r"Slow Cooling, $\gamma_m < \gamma_c$")
        axes[0].plot(freq_to_integrate, sc_power_JOH, color='red', label='JOH')
        axes[0].axvline(x=sc_nu_m_JOH, linestyle='dashed', color="red")
        axes[0].axvline(x=sc_nu_c_JOH, linestyle="dotted", color="red")
        # axes[0].axhline(y=sc_pmax_JOH, linestyle='dotted', color="red")

        axes[0].plot(freq_to_integrate, sc_power_WSPN, color='green', label='WSPN')
        axes[0].axvline(x=sc_nu_m_WSPN, linestyle='dashed', color="green")
        axes[0].axvline(x=sc_nu_c_WSPN, linestyle="dotted", color="green")
        # axes[0].axhline(y=sc_pmax_JOH, linestyle='dotted', color="green")

        axes[0].plot(freq_to_integrate, sc_power_DER, color='black', label='DER')

        axes[0].set_ylabel(r"$F'_{\nu'}$", fontsize=12)
        axes[0].set_xlabel(r"$\nu'$", fontsize=12)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlim(1e9, 1e19)
        # axes[0].set_ylim(5e-5, 8e-0)
        axes[0].set_ylim(1e14, 1e21)
        axes[0].grid(1)

        # fc
        axes[1].set_title(r"Fast Cooling, $\gamma_m > \gamma_c$")
        axes[1].plot(freq_to_integrate, fc_power_JOH, color='red', label='JOH')
        axes[1].axvline(x=fc_nu_m_JOH, linestyle='dashed', color="red")
        axes[1].axvline(x=fc_nu_c_JOH, linestyle="dotted", color="red")
        # axes[0].axhline(y=sc_pmax_JOH, linestyle='dotted', color="red")

        axes[1].plot(freq_to_integrate, fc_power_WSPN, color='green', label='WSPN')
        axes[1].axvline(x=fc_nu_m_WSPN, linestyle='dashed', color="green")
        axes[1].axvline(x=fc_nu_c_WSPN, linestyle="dotted", color="green")
        # axes[0].axhline(y=sc_pmax_JOH, linestyle='dotted', color="green")

        axes[1].plot(freq_to_integrate, fc_power_DER, color='black', label='DER')

        # axes[1].set_ylabel(r"$P'_{\nu}'$", fontsize=12)
        axes[1].set_xlabel(r"$\nu'$", fontsize=12)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlim(1e9, 1e19)
        # axes[1].set_ylim(5e-5, 8e-0)
        axes[1].set_ylim(1e14, 1e21)
        axes[1].grid(1)
        axes[1].legend()

        plt.subplots_adjust(hspace=0., wspace=0.)

        #if not (save is None):
        plt.savefig(figures_dir + "compare_sed_for_diff_formulisms.png")

        plt.show()

        # plt.loglog(nu_arr, nu_arr*power_GAO, color='gray', label='GAO')
        # plt.axvline(x=nu_a_GAO, linestyle='solid', color="gray")
        # plt.axvline(x=nu_m_GAO, linestyle='dashed', color="gray")
        # plt.axvline(x=nu_c_GAO, linestyle="dotted", color="gray")
        # plt.axhline(y=pmax_GAO, linestyle='dotted', color="gray")

    def test_attenuation(self):
        ''' '''
        ncells = 1
        theta = np.pi
        R = 1e12
        Gamma = 120
        m2 = 1e20
        rho = 1e-1 * cgs.mppme
        rho2 = 4 * Gamma * rho
        delta_shock = m2 / (8. * np.pi * (1. - np.cos(theta) / ncells) * Gamma ** 2. * rho * R ** 2.)

        B = 1
        p = 2.2
        ''' --- sc --- '''

        gamma_min = 1e1
        gamma_c = 1e3

        ### Johanosn
        o_JOH = Synchrotron_Joh06(
            p,
            gamma_min,
            gamma_c,
            B,
            R,
            Gamma,
            rho2 / cgs.mppme,
            delta_shock,
            False
        )
        sc_atten_JOH = o_JOH.tau_attenuation(freq_to_integrate)

        ### Dermer
        bpl = BrokenPowerLaw.from_normalised_density(
            n_e_tot=rho2 / cgs.mppme,
            p1=p if gamma_min < gamma_c else 2.,
            p2=p + 1,
            gamma_b=gamma_c if gamma_min < gamma_c else gamma_min,
            gamma_min=gamma_min if gamma_min < gamma_c else gamma_c,
            gamma_max=1e6)

        o_DER = Synchrotron_DM06(
            bpl,
            bpl.parameters,
            delta_shock,
            B,
            False,
            np.trapz
        )
        sc_atten_DER = o_DER.tau_attenuation(freq_to_integrate)

        ''' --- fc --- '''

        gamma_min = 1e4
        gamma_c = 1e3

        ### Johanosn
        o_JOH = Synchrotron_Joh06(
            p,
            gamma_min,
            gamma_c,
            B,
            R,
            Gamma,
            rho2 / cgs.mppme,
            delta_shock,
            False
        )
        fc_atten_JOH = o_JOH.tau_attenuation(freq_to_integrate)

        ### Dermer
        bpl = BrokenPowerLaw.from_normalised_density(
            n_e_tot=rho2 / cgs.mppme,
            p1=p if gamma_min < gamma_c else 2.,
            p2=p + 1,
            gamma_b=gamma_c if gamma_min < gamma_c else gamma_min,
            gamma_min=gamma_min if gamma_min < gamma_c else gamma_c,
            gamma_max=1e6)

        o_DER = Synchrotron_DM06(
            bpl,
            bpl.parameters,
            delta_shock,
            B,
            False,
            np.trapz
        )
        fc_atten_DER = o_DER.tau_attenuation(freq_to_integrate)

        ''' --- '''

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey='all')

        # sc
        axes[0].set_title(r"Slow Cooling, $\gamma_m < \gamma_c$")
        axes[0].plot(freq_to_integrate, sc_atten_JOH, color='red', label='JOH')
        # axes[0].axvline(x=sc_nu_m_JOH, linestyle='dashed', color="red")
        # axes[0].axvline(x=sc_nu_c_JOH, linestyle="dotted", color="red")
        # axes[0].axhline(y=sc_pmax_JOH, linestyle='dotted', color="red")

        axes[0].plot(freq_to_integrate, sc_atten_DER, color='black', label='DER')

        axes[0].set_ylabel(r"SSA Attenuation", fontsize=12)
        axes[0].set_xlabel(r"$\nu'$", fontsize=12)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlim(1e6, 1e13)
        axes[0].set_ylim(1e-6, 2e-0)
        axes[0].grid(1)

        # fc
        axes[1].set_title(r"Fast Cooling, $\gamma_m > \gamma_c$")
        axes[1].plot(freq_to_integrate, fc_atten_JOH, color='red', label='JOH')
        # axes[1].axvline(x=fc_nu_m_JOH, linestyle='dashed', color="red")
        # axes[1].axvline(x=fc_nu_c_JOH, linestyle="dotted", color="red")
        # axes[0].axhline(y=sc_pmax_JOH, linestyle='dotted', color="red")

        axes[1].plot(freq_to_integrate, fc_atten_DER, color='black', label='DER')

        # axes[1].set_ylabel(r"SSA Attenuation", fontsize=12)
        axes[1].set_xlabel(r"$\nu'$", fontsize=12)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlim(1e6, 1e13)
        axes[1].set_ylim(1e-6, 2e-0)
        axes[1].grid(1)
        axes[1].legend()

        plt.subplots_adjust(hspace=0., wspace=0.)

        #if not (save is None):
        plt.savefig(figures_dir + "compare_ssa_atten_for_diff_formulisms.png")

        # plt.loglog(nu_arr, nu_arr*power_GAO, color='gray', label='GAO')
        # plt.axvline(x=nu_a_GAO, linestyle='solid', color="gray")
        # plt.axvline(x=nu_m_GAO, linestyle='dashed', color="gray")
        # plt.axvline(x=nu_c_GAO, linestyle="dotted", color="gray")
        # plt.axhline(y=pmax_GAO, linestyle='dotted', color="gray")

        plt.show()

if __name__ == '__main__':
    o_test = TestSyncrotron()
    o_test.test_sed()
    o_test.test_attenuation()