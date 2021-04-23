"""
    pass
"""

import numpy as np
from PyBlastAfterglow.uutils import cgs, axes_reshaper
from scipy import interpolate

from .synchrotron import Synchrotron

gamma_to_integrate = np.logspace(1., 9., 200) # default arrays to be used for integration

def R(x):
    """
    Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(x, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)

def nu_synch_peak(B, gamma):
    """
    observed peak frequency for monoenergetic electrons
    Eq. 7.19 in [DermerMenon2009]_
    :B: float
    magnetic field strength in Gauss
    """
    # B = B_to_cgs(B)
    nu_peak = (cgs.qe * B / (2 * np.pi * cgs.me * cgs.c)) * np.power(gamma, 2)
    return nu_peak#.to("Hz")

def calc_x(B, epsilon, gamma):
    """
    ratio of the frequency to the critical synchrotron frequency from
    Eq. 7.34 in [DermerMenon2009]_, argument of R(x),
    note B has to be in cgs Gauss units
    """
    x = (
        4
        * np.pi
        * epsilon
        * np.power(cgs.me, 2)
        * np.power(cgs.c, 3)
        / (3 * cgs.qe * B * cgs.h * np.power(gamma, 2))
    )
    return x#.to_value("")

def single_electron_synch_power(B, epsilon, gamma):
    """
    angle-averaged synchrotron power for a single electron,
    to be folded with the electron distribution
    :B: float
        magnetic field strength in Gauss
    """
    x = calc_x(B, epsilon, gamma)
    prefactor = np.sqrt(3) * np.power(cgs.qe, 3) * B / cgs.h
    return prefactor * R(x)

def tau_to_attenuation(tau):
    """
    Converts the synchrotron self-absorption optical depth to an attenuation
    Eq. 7.122 in [DermerMenon2009]_.
    """
    u = 1 / 2 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
    return np.where(tau < 1e-3, 1., 3 * u / tau)

class Synchrotron_DM06(Synchrotron):
    """Class for synchrotron radiation computation

    Parameters
    ----------
    ssa : bool
        whether or not to consider synchrotron self absorption (SSA).
    integrator : (`~uutils.trapz_loglog`, `~numpy.trapz`)
        function to be used for the integration
	"""

    def __init__(self,
                 electrons,
                 electron_pars,
                 thickness,
                 B,
                 R,
                 Gamma,
                 ssa=False,
                 integrator=np.trapz
                 ):

        self.B = B
        self.R = R
        self.Gamma = Gamma
        self.electrons = electrons
        self.electron_pars = electron_pars
        self.thickness = thickness
        self.ssa = ssa
        self.integrator = integrator

        super(Synchrotron_DM06, self).__init__()


    def __call__(self, nupime):
        return self.sed(nupime)

    @staticmethod
    def evaluate_tau_ssa(
            nuprim, # Hz
            B, # Gauss
            length, # cm ; 2 * R_b for the blob
            n_e, # object
            *args, # for electron distribution
            integrator=np.trapz,
            gamma=gamma_to_integrate, # integration limits
    ):
        """ Computes the syncrotron self-absorption opacity for a general set
            of model parameters, see
            :func:`~agnpy:sycnhrotron.Synchrotron.evaluate_sed_flux`
            for parameters defintion.
            Eq. before 7.122 in [DermerMenon2009]_.
        """
        # conversions
        epsilon = nuprim * cgs.h / cgs.mec2 # = nu_to_epsilon_prime(nu, z, delta_D)
        # multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        SSA_integrand = n_e.evaluate_SSA_integrand(_gamma, *args)
        integrand = SSA_integrand * single_electron_synch_power(B, _epsilon, _gamma)
        integral = integrator(integrand, gamma, axis=0)
        prefactor_k_epsilon = (
                -1 / (8 * np.pi * cgs.me * np.power(epsilon, 2)) * np.power(cgs.lambda_c / cgs.c, 3)
        )
        k_epsilon = (prefactor_k_epsilon * integral)#.to("cm-1")
        return (k_epsilon * length) #.to_value("") # dimensionless

    @staticmethod
    def evaluate_sed(
            nuprim,
            B,
            R,
            Gamma,
            length,
            n_e_obj,
            *args,
            ssa=False,
            integrator=np.trapz,
            gamma=gamma_to_integrate,
    ):
        # conversions
        epsilon = nuprim * cgs.h / cgs.mec2
        # reshape for multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        N_e = 1. * n_e_obj.evaluate(_gamma, *args) # 1. is volume * n_e which is number of electrons
        integrand = N_e * single_electron_synch_power(B, _epsilon, _gamma)
        emissivity = integrator(integrand, gamma, axis=0)
        sed = epsilon * emissivity * (R ** 2 * length * Gamma)

        if ssa:
            tau = Synchrotron_DM06.evaluate_tau_ssa(
                nuprim,
                B,
                R,
                Gamma,
                length,
                n_e_obj,
                *args,
                integrator=integrator,
                gamma=gamma,
            )
            attenuation = tau_to_attenuation(tau)
            sed *= attenuation

        return sed

    def sed(self, nuprime):
        return self.evaluate_sed(
            nuprime,
            self.B,
            self.R,
            self.Gamma,
            self.thickness,
            self.electrons,
            *self.electron_pars,
            ssa=False,
            integrator=np.trapz,
            gamma=gamma_to_integrate,
        )

    def tau_attenuation(self, nuprim):
        tau = Synchrotron_DM06.evaluate_tau_ssa(
            nuprim,
            self.B,
            self.thickness,
            self.electrons,
            *self.electron_pars,
            integrator=np.trapz,
            gamma=gamma_to_integrate,
        )
        return tau_to_attenuation(tau)