"""
    pass
"""

import numpy as np
from PyBlastAfterglow.uutils import cgs, axes_reshaper
from scipy import interpolate

from .synchrotron import Synchrotron

def SSA_joh06(nu, nu_m, nu_c, p, alpha0F, alpha0S):
    """
            See description of the synchrotron self-absorption in afterglow context:
            https://iopscience.iop.org/article/10.1086/308052/fulltext/39011.text.html

            See description also in PhD thesis (p.19)
            https://www.imprs-astro.mpg.de/sites/default/files/varela_karla.pdf

            Soruce of these exact formulas:
            N/A

    """
    if hasattr(nu, '__len__'):  # array treatment
        assert len(nu) > 0
        alpha_out = np.zeros(len(nu))
        sc = np.array(nu_m <= nu_c, dtype=bool)  # slow cooling
        fc = np.array(nu_m > nu_c, dtype=bool)  # fast cooling
        if any(sc) > 0:  # Slow cooling
            fill1 = sc & (nu <= nu_m)
            fill2 = sc & ((nu_m < nu) & (nu <= nu_c))
            fill3 = sc & (nu_c < nu)
            alpha_out[fill1] = (nu[fill1] / nu_m[fill1]) ** (-5 / 3.) * alpha0S[fill1]
            alpha_out[fill2] = (nu[fill2] / nu_m[fill2]) ** (-(p + 4) / 2) * alpha0S[fill2]
            alpha_out[fill3] = (nu_c[fill3] / nu_m[fill3]) ** (-(p + 4) / 2) * \
                               (nu[fill3] / nu_c[fill3]) ** (-(p + 5) / 2) * alpha0S[fill3]
        if any(fc) > 0:  # fast cooling
            fill1f = fc & (nu <= nu_c)
            fill2f = fc & ((nu_c < nu) & (nu <= nu_m))
            fill3f = fc & (nu_m < nu)
            alpha_out[fill1f] = (nu[fill1f] / nu_c[fill1f]) ** (-5 / 3.) * alpha0F[fill1f]
            alpha_out[fill2f] = (nu[fill2f] / nu_c[fill2f]) ** (-3) * alpha0F[fill2f]
            alpha_out[fill3f] = (nu_m[fill3f] / nu_c[fill3f]) ** (-3) * \
                                (nu[fill3f] / nu_m[fill3f]) ** (-(p + 5) / 2) * alpha0F[fill3f]
        return alpha_out
    else:  # float treatment
        if nu_m <= nu_c:  # slow cooling
            if (nu <= nu_m):
                alpha_out = (nu / nu_m) ** (-5 / 3.) * alpha0S
            elif (nu_m < nu) and (nu <= nu_c):
                alpha_out = (nu / nu_m) ** (-(p + 4) / 2) * alpha0S
            elif (nu_c < nu):
                alpha_out = (nu_c / nu_m) ** (-(p + 4) / 2) * (nu / nu_c) ** (-(p + 5) / 2) * alpha0S
            else:
                raise ValueError()
        else:  # fast cooling
            if (nu <= nu_c):
                alpha_out = (nu / nu_c) ** (-5 / 3.) * alpha0F
            elif (nu_c < nu) and (nu <= nu_m):
                alpha_out = (nu / nu_c) ** (-3) * alpha0F
            elif (nu_m < nu):
                alpha_out = (nu_m / nu_c) ** (-3) * (nu / nu_m) ** (-(p + 5) / 2) * alpha0F
            else:
                raise ValueError()
        return alpha_out

def SED_joh06(nu, nu_m, nu_c, p, PmaxF, PmaxS):
    kappa1 = 2.37 - 0.3 * p
    kappa2 = 14.7 - 8.68 * p + 1.4 * p ** 2
    kappa3 = 6.94 - 3.844 * p + 0.62 * p ** 2
    kappa4 = 3.5 - 0.2 * p

    kappa13 = -kappa1 / 3.
    kappa12 = kappa1 / 2.
    kappa11 = -1. / kappa1
    kappa2p = kappa2 * (p - 1) / 2.
    kappa12inv = -1. / kappa2
    kappa33 = -kappa3 / 3
    kappa3p = kappa3 * (p - 1) / 2.
    kappa13inv = -1. / kappa3
    kappa42 = kappa4 / 2
    kappa14 = -1. / kappa4

    if hasattr(nu, '__len__'):  # array

        fc = nu_m > nu_c  # fast cooling
        sc = nu_m <= nu_c  # slow cooling

        P_out = np.zeros(len(nu_m))

        P_out[fc] = PmaxF[fc] * \
                    ((nu[fc] / nu_c[fc]) ** (kappa13) + (nu[fc] / nu_c[fc]) ** (kappa12)) ** (kappa11) * \
                    (1 + (nu[fc] / nu_m[fc]) ** (kappa2p)) ** (kappa12inv)

        P_out[sc] = PmaxS[sc] * \
                    ((nu[sc] / nu_m[sc]) ** (kappa33) + (nu[sc] / nu_m[sc]) ** (kappa3p)) ** (kappa13inv) * \
                    (1 + (nu[sc] / nu_c[sc]) ** (kappa42)) ** (kappa14)
        return P_out
    else:  # float
        sc = bool(nu_m <= nu_c)
        fc = bool(nu_m > nu_c)

        if fc:  # fast cooling
            P_out = PmaxF * \
                    ((nu / nu_c) ** (kappa13) + (nu / nu_c) ** (kappa12)) ** (kappa11) * \
                    (1 + (nu / nu_m) ** (kappa2p)) ** (kappa12inv)
        elif sc:  # slow cooling
            P_out = PmaxS * \
                    ((nu / nu_m) ** (kappa33) + (nu / nu_m) ** (kappa3p)) ** (kappa13inv) * \
                    (1 + (nu / nu_c) ** (kappa42)) ** (kappa14)
        else:
            raise ValueError("Neither fast nor slow cooling")
        return P_out

def get_num_nuc_fmax(
        p,
        gamma_min,
        gamma_c,
        B,
        ne,
):
    phipF = 1.89 - 0.935 * p + 0.17 * p ** 2
    phipS = 0.54 + 0.08 * p
    XpF = 0.455 + 0.08 * p  # for self-absorption
    XpS = 0.06 + 0.28 * p  # for self-absorption

    gamToNuFactor = (3. / (4. * np.pi)) * (cgs.qe * B) / (cgs.me * cgs.c)

    rhoprim = ne * cgs.mppme

    if gamma_min < gamma_c:
        # slow cooling
        nu_m = XpS * gamma_min ** 2 * gamToNuFactor
        nu_c = XpS * gamma_c ** 2 * gamToNuFactor
        _phip = 11.17 * (p - 1) / (3 * p - 1) * phipS
        pprimemax = _phip * cgs.qe ** 3 * ne * B / (cgs.me * cgs.c ** 2)
        # for synch. self absorption
        _alpha = 7.8 * phipS * XpS ** (-(4 + p) / 2.) * (p + 2) * (p - 1) * cgs.qe / cgs.mp / (p + 2 / 3.)
        alpha = _alpha * rhoprim * gamma_min ** (-5) / B

    else:
        # fast cooling
        nu_m = XpF * gamma_min ** 2 * gamToNuFactor
        nu_c = XpF * gamma_c ** 2 * gamToNuFactor
        _phip = 2.234 * phipF
        pprimemax = _phip * cgs.qe ** 3 * ne * B / (cgs.me * cgs.c ** 2)
        # for synch. self absorption
        _alpha = 11.7 * phipF * XpF ** (-3) * cgs.qe / cgs.mp
        alpha = _alpha * rhoprim * gamma_c ** (-5) / B

    return (nu_m, nu_c, pprimemax, alpha)

class Synchrotron_Joh06(Synchrotron):

    def __init__(
            self,
            p,
            gamma_min,
            gamma_c,
            B,
            R,
            Gamma,
            ne,
            delta_shock,
            ssa=False
    ):
        """
        :ne: is the comoving number density of electrons
        """

        # self.gamma_min =gamma_min
        # self.gamma_c = gamma_c
        self.p = p
        self.B = B
        self.ne = ne
        self.delta_shock = delta_shock
        self.ssa = ssa

        # phipF = 1.89 - 0.935 * p + 0.17 * p ** 2
        # phipS = 0.54 + 0.08 * p
        # XpF = 0.455 + 0.08 * p  # for self-absorption
        # XpS = 0.06 + 0.28 * p  # for self-absorption
        #
        # gamToNuFactor = (3. / (4. * np.pi)) * (cgs.qe * B) / (cgs.me * cgs.c)
        #
        # rhoprim = ne * cgs.mppme
        #
        # if gamma_min < gamma_c:
        #     # slow cooling
        #     self.nu_m = XpS * gamma_min ** 2 * gamToNuFactor
        #     self.nu_c = XpS * gamma_c ** 2 * gamToNuFactor
        #     _phip = 11.17 * (p - 1) / (3 * p - 1) * phipS
        #     self.pprimemax = _phip * cgs.qe ** 3 * ne * B / (cgs.me * cgs.c ** 2)
        #     # for synch. self absorption
        #     _alpha = 7.8 * phipS * XpS ** (-(4 + p) / 2.) * (p + 2) * (p - 1) * cgs.qe / cgs.mp / (p + 2 / 3.)
        #     self.alpha = _alpha * rhoprim * gamma_min ** (-5) / B
        #
        # else:
        #     # fast cooling
        #     self.nu_m = XpF * gamma_min ** 2 * gamToNuFactor
        #     self.nu_c = XpF * gamma_c ** 2 * gamToNuFactor
        #     _phip = 2.234 * phipF
        #     self.pprimemax = _phip * cgs.qe ** 3 * ne * B / (cgs.me * cgs.c ** 2)
        #     # for synch. self absorption
        #     _alpha = 11.7 * phipF * XpF ** (-3) * cgs.qe / cgs.mp
        #     self.alpha = _alpha * rhoprim * gamma_c ** (-5) / B

        self.nu_m, self.nu_c, self.pprimemax, self.alpha = get_num_nuc_fmax(
            p,
            gamma_min,
            gamma_c,
            B,
            ne
        )
        self.Fmax = self.pprimemax * R ** 2 * self.delta_shock * Gamma

        super(Synchrotron_Joh06, self).__init__()

    @classmethod
    def from_obj_fs(cls, R, electons, dynamics, ssa=False):
        idx = int(np.where(dynamics.vals["R"] == R)[0][0])
        assert idx > 0
        return cls(
            electons.p1,
            electons.vals["gamma_min"][idx],
            electons.vals["gamma_c"][idx],
            electons.vals["B"][idx],
            dynamics.vals["R"][idx],
            dynamics.vals["Gamma"][idx],
            dynamics.vals["rho2"][idx] / cgs.mppme,
            dynamics.vals["thickness"][idx],
            ssa
        )

    def __call__(self, nupime):
        return self.sed(nupime)

    @staticmethod
    def evaluate_tau_ssa(
            nuprime, # Hz
            nu_m, # Gauss
            nu_c, # cm ; 2 * R_b for the blob
            alpha, # object
            delta_shock,
            p
    ):

        if hasattr(nuprime, '__len__') and ( (not hasattr(nu_m, '__len__')) or (not hasattr(alpha, '__len__')) ):
            nu_m = np.full_like(nuprime, nu_m)
            nu_c = np.full_like(nuprime, nu_c)
            alpha = np.full_like(nuprime, alpha)

        alpha_sed = SSA_joh06(nuprime, nu_m, nu_c, p, alpha, alpha)
        alpha_sed *= delta_shock / 2.

        if hasattr(alpha_sed, '__len__'):
            atten=np.ones_like(alpha_sed)
            msc = alpha_sed > 1e-2
            atten[msc] = (1. - np.exp(-alpha_sed[msc])) / alpha_sed[msc]
            msc2 = (1e-8<alpha_sed) & (alpha_sed<1e-2)
            atten[msc2] = \
                (alpha_sed[msc2] - alpha_sed[msc2] ** 2 / 2 + alpha_sed[msc2] ** 4 / 4 - alpha_sed[msc2] ** 6 / 6) / \
                alpha_sed[msc2]
        else:
            if alpha_sed > 1e-2:
                atten = (1. - np.exp(-alpha_sed)) / alpha_sed
            elif 1e-8 < alpha_sed <= 1e-2:
                atten = (alpha_sed - alpha_sed ** 2 / 2 + alpha_sed ** 4 / 4 - alpha_sed ** 6 / 6) / alpha_sed
            else:
                atten = 1.

        return atten

    @staticmethod
    def evaluate_sed(
            nuprime,
            nu_m,
            nu_c,
            Fmax,
            p,
            alpha=None,
            delta_shock=0.,
            ssa=False
    ):
        """
            Based on eqs from https://arxiv.org/abs/astro-ph/0605299
            and https://arxiv.org/abs/1805.05875
            by Johanneson et al
        """
        if hasattr(nuprime, '__len__') and (not hasattr(nu_m, '__len__')):
            nu_m = np.full_like(nuprime, nu_m)
            nu_c = np.full_like(nuprime, nu_c)
            Fmax = np.full_like(nuprime, Fmax)

        spectrum = SED_joh06(nuprime, nu_m, nu_c, p, Fmax, Fmax)

        if ssa:
            atten = Synchrotron_Joh06.evaluate_tau_ssa(
                nuprime,
                nu_m,
                nu_c,
                alpha,
                delta_shock,
                p
            )

            spectrum *= atten

        return nuprime * spectrum

    def sed(self, nuprime):
        return self.evaluate_sed(
            nuprime,
            self.nu_m,
            self.nu_c,
            self.Fmax,
            self.p,
            self.alpha,
            self.delta_shock,
            self.ssa
        )

    def tau_attenuation(self, nuprime):
        return self.evaluate_tau_ssa(
            nuprime,  # Hz
            self.nu_m,  # Gauss
            self.nu_c,  # cm ; 2 * R_b for the blob
            self.alpha,  # object
            self.delta_shock,
            self.p
        )

