"""
    pass
"""

import numpy as np
import numba as nb

from PyBlastAfterglow.uutils import cgs

def get_beta(Gamma):
    return np.sqrt(1. - np.power(float(Gamma), -2))

def get_Gamma(beta):
    return np.float64(np.sqrt(1. / (1. - np.float64(beta) ** 2.)))

def rho_dlnrho1dR(R, nn, A0, s, R_EJ, R_ISM):
    if not nn is None:
        rho = nn * cgs.mppme
        dlnrho1dR = 0.
    else:
        if R < R_EJ:
            rho = A0 * R_EJ ** (-s) * cgs.mppme
            dlnrho1dR = 0.
        elif R >= R_EJ and R < R_ISM:
            rho = A0 * R ** (-s) * cgs.mppme
            dlnrho1dR = -s / R
        else:
            rho = A0 * R_ISM ** (-s) * cgs.mppme
            # rho = pars.A0 / pars.M0 * pars.R_ISM ** (-pars.s) * mppme
            dlnrho1dR = 0.
    # if scalebyM0: rho = rho / M0
    return (rho, dlnrho1dR)

def get_Rdec2(E, nn, Gamma):
    rdec = (3. / (4. * cgs.pi) * 1. / (cgs.c ** 2. * cgs.mp) * E / (nn * Gamma ** 2.)) ** (1. / 3.)
    return rdec

def get_Rdec(M0, A0, Gamma0, s):
    """
    From Nava et al 2013
    :param M0:
    :param A0:
    :param Gamma0:
    :param s:
    :return:
    """
    Rdec = ((3. - s) * M0 / (4. * np.pi * A0 * Gamma0)) ** (1 / (3. - s))
    return Rdec

def get_bm79(E0, A0, s, R):
    """

    :param E0:
    :param A0:
    :param s:
    :param R:
    :return:
    """
    Gamma = np.sqrt((17. - 4.*s) * E0 / (16 * np.pi * A0 * (cgs.c ** 2) * R ** (3 - s))) # FIXED
    beta = (1 - 1. / Gamma ** 2) ** (1 / 2.)
    # return R[Gamma * beta > 1.], Gamma[Gamma * beta > 1.], beta[Gamma * beta > 1.]

    return Gamma

def get_sedovteylor(Rdec, beta0, R):
    # Rdec = float(((3 - s) * M0 / (4 * np.pi * A0 * Gamma0)) ** (1 / (3. - s)))

    beta = np.zeros(len(R))
    beta.fill(beta0)
    beta[R > Rdec] = beta0 * (R[R > Rdec] / Rdec) ** (-3 / 2)
    Gamma_st = 1 / (1 - beta ** 2) ** (1 / 2)

    # return R[R > Rdec] + Rdec, Gamma_st[R > Rdec], beta[R > Rdec]
    return beta


