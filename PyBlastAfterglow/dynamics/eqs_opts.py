"""
    pass
"""
import numpy as np
import numba as nb
from PyBlastAfterglow.uutils import cgs


def dthetadr_None(*args):
    return 0.

def dthetadr_Adi(gammaAdi, Gamma, beta, R, theta, aa, useSpread, Rd, thetamax=np.pi / 2.):
    """ basic method """
    vperp = np.sqrt(gammaAdi * (gammaAdi - 1) * (Gamma - 1) / (1 + gammaAdi * (Gamma - 1))) * cgs.c
    return vperp / R / Gamma / beta / cgs.c if (theta < thetamax) & useSpread else 0.

def dthetadr_Adi_Rd(gammaAdi, Gamma, beta, R, theta, aa, useSpread, Rd, thetamax=np.pi / 2.):
    """ basic method that starts working after Rd """
    return dthetadr_Adi(gammaAdi, Gamma, beta, R, theta, aa, useSpread, Rd, thetamax) if (R > Rd) else 0.

def dthetadr_AA(gammaAdi, Gamma, beta, R, theta, aa, useSpread, Rd, thetamax=np.pi / 2.):
    """ Method of Granot & Piran 2012 with 'a' parameter """
    return 1. / (R * Gamma ** (1. + aa) * theta ** (aa)) if (theta < thetamax) & useSpread else 0.

def dthetadr_AA_Rd(gammaAdi, Gamma, beta, R, theta, aa, useSpread, Rd, thetamax=np.pi / 2.):
    """ Method of Granot & Piran 2012 with 'a' parameter that starts working after Rd """
    return dthetadr_AA(gammaAdi, Gamma, beta, R, theta, aa, useSpread, Rd, thetamax) if (R > Rd) else 0.

def opts_dthetadr(method):
    if method == "None":
        return dthetadr_None
    elif method == "Adi":
        return dthetadr_Adi
    elif method == "Adi_Rd":
        return dthetadr_Adi_Rd
    elif method == "AA":
        return dthetadr_AA
    elif method == "AA_Rd":
        return dthetadr_AA_Rd
    else:
        raise NameError("Method {} does not exist for dthetadr ".format(method))

#
# @staticmethod
# def dthetadr(gammaAdi, Gamma, R, theta, aa=np.nan):
#     """
#     Source:
#     1. / (R * Gamma ** (1. + aa) * theta ** (aa))
#     is from https://academic.oup.com/mnras/article/421/1/570/990386
#     vperp / R / Gamma * one_over_beta / cgs.c
#     is from ???
#     :param gammaAdi:
#     :param Gamma:
#     :param R:
#     :param one_over_beta:
#     :param theta:
#     :param aa:
#     :return:
#     """
#
#     if theta < np.pi:
#         if np.isfinite(aa):
#             return 1. / (R * Gamma ** (1. + aa) * theta ** (aa))
#         else:
#             vperp = np.sqrt(gammaAdi * (gammaAdi - 1) * (Gamma - 1) / (1 + gammaAdi * (Gamma - 1))) * cgs.c
#             one_over_beta = 1. / np.power(1 - np.power(Gamma, -2.), 0.5)
#             return vperp / R / Gamma * one_over_beta / cgs.c
#     else:
#         return 0.


def dmdr(Gamma, RR, thetaE, theta, rho, aa=-1.):
    """
    https://arxiv.org/pdf/1203.5797.pdf
    https://academic.oup.com/mnras/article/421/1/570/990386 (for introduction of 'a' parameter)

    basic form was dmdr = 2 * cgs.pi * R ** 2. * rho * one_minus_costheta / pars["m_scale"]
    """
    # First term: change in swept-up mass due to the change in solid angle
    t1 = 0. if (aa < 0) else (1. / 3.) * np.sin(theta) / (Gamma ** (1. + aa) * theta ** (aa))
    # Second term: change in swept-up mass due to radial expansion
    t2 = (np.cos(thetaE) - np.cos(theta))  # -> always (1 - cos(theta))
    return 2. * np.pi * rho * (t1 + t2) * RR ** 2.

def opts_dmdr(method):
    if method == "default":
        return dmdr
    else:
        raise NameError("Method {} does not exist for dmdr ".format(method))
# --- Adiabatic index (EOS) ---


def gamma_adi_nava(Gamma, beta):
    """ From Nava 2013 paper """
    return (4 + 1 / Gamma) / 3.

def gamma_adi_peer(Gamma, beta):
    """ From Peer 2012 arxiv:1203.5797 """
    mom = Gamma * beta
    theta = mom / 3. * (mom + 1.07 * mom ** 2.) / (1 + mom + 1.07 * mom ** 2.)
    zz = theta / (0.24 + theta)
    gamma_adi = (5. - 1.21937 * zz + 0.18203 * zz ** 2. - 0.96583 * zz ** 3. + 2.32513 * zz ** 4. -
                 2.39332 * zz ** 5. + 1.07136 * zz ** 6.) / 3.
    return gamma_adi

def opts_gammaAdi(method):
    if method == "Nava":
        return gamma_adi_nava
    elif method == "Peer":
        return gamma_adi_peer
    else:
        raise NameError("Method {} does not exist for gammaAdi ".format(method))
# --- rho2 (comoving density) ---


def rho2_rel(Gamma, beta, rho, gammaAdi):
    """ Eq. 36 in Rumar + 2014 arXiv:1410.0679 if Gamma >> 1"""
    return 4. * rho * Gamma

def rho2_transrel(Gamma, beta, rho, gammaAdi):
    """ Eq. 36 in Rumar + 2014 arXiv:1410.0679 """
    return (gammaAdi * Gamma + 1.) / (gammaAdi - 1.)

def opts_rho2(method):
    if method == "rel":
        return rho2_rel
    elif method == "transrel":
        return rho2_transrel
    else:
        raise NameError("Method {} does not exist for rho2 ".format(method))

# --- thickness ---


def shock_thickness(mass, rhoprime, theta, Gamma, R, ncells):
    """ eq. from Johannesson 2006 : astro-ph/0605299 """
    one_min_costheta = (1 - np.cos(theta) / ncells)
    # rhoprime = 4. * rho * Gamma
    delta = mass / (2 * np.pi * one_min_costheta * rhoprime * Gamma * R ** 2)
    return delta

def opts_shock_thickness(method):
    if method == "default":
        return shock_thickness
    else:
        raise NameError("Method {} does not exist for shock_thickness ".format(method))