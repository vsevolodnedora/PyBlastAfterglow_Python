import numpy as np
import numba as nb
from PyBlastAfterglow.uutils import cgs
from .dynamics import Driver
from .eqs import (get_beta, get_Rdec2, rho_dlnrho1dR)
from .eqs_opts import (opts_dthetadr, opts_dmdr, opts_gammaAdi, opts_rho2, opts_shock_thickness)

from .model_nava import (get_GammaEff, get_dGammaEffdGamma, get_U_e)

def get_gamma43_minus_one(Gamma, Gamma0, beta, beta0, beta_switch=0.9999):
    # if Gamma > 0.999995 * Gamma0:
    #     return 0
    # else:
    #     return Gamma * Gamma0 - Gamma * Gamma0 * beta * beta0
    # return np.sqrt(Gamma**2 * Gamma0**2 - Gamma**2 - Gamma0**2 + 1)

    # print("1:{} 2:{}".format(Gamma * Gamma0 - Gamma * Gamma0 * beta * beta0,
    #                          Gamma*Gamma0 - np.sqrt(Gamma**2 * Gamma0**2 - Gamma0**2 - Gamma**2 + 1)))
    # return Gamma*Gamma0 - np.sqrt(Gamma**2 * Gamma0**2 - Gamma0**2 - Gamma**2 + 1)

    # gamma43_minus_one = Gamma * Gamma0 * (1 - beta * beta0)
    # if gamma43_minus_one == 1.:
    #     gamma43_minus_one = Gamma * Gamma0 * \
    #                         (1 / Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / Gamma0 ** 2) / \
    #                         (1 + beta * beta0) - 1
    #
    # return gamma43_minus_one

    if Gamma > 0.999995 * Gamma0:
        gamma43_minus_one = 1. - 1.
    else:
        gamma43_minus_one = Gamma * Gamma0 * (1 / Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / Gamma0 ** 2) / (
                    1 + beta * beta0) - 1.
        # gamma43_minus_one = Gamma*Gamma0 - np.sqrt(Gamma0**2 - 1) * np.sqrt(Gamma**2 - 1) # -- wolfram. Creats jump

        # if beta < 0.999:
        # gamma43_minus_one = Gamma * Gamma0 * (1 - beta * beta0)
        # gamma43_minus_one = Gamma * Gamma0 * (1 / Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / Gamma0 ** 2) / (1 + beta * beta0) - 1.
        #
        # print(gamma43_minus_one)
        # else:
        # gamma43_minus_one = Gamma * Gamma0 * (1 / Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / Gamma0 ** 2) / (1 + beta * beta0) - 1.
        # print(gamma43_minus_one, Gamma * Gamma0 * (1 - beta * beta0)); exit(1)
    # if Gamma > 0.999995 * pars.Gamma0:
    #     gamma43_minus_one = 0.
    # else:
    #     gamma43_minus_one = Gamma * pars.Gamma0 * (
    #                 1 / pars.Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / pars.Gamma0 ** 2) / (
    #                                     1 + beta * beta0) - 1
    # # else:
    # # gamma43_minus_one = Gamma*ModVar.Gamma0*(1-beta*beta0) - 1
    return gamma43_minus_one


def get_dgamma43dGamma(Gamma0, Gamma):
    """ these two should be equivalent but they are not exactly the same at Gamma=>1 """
    return Gamma0 - (Gamma0 ** 2 * Gamma - Gamma) / (Gamma ** 2 * Gamma0 ** 2 - Gamma0 ** 2 - Gamma ** 2 + 1) ** (
                1. / 2.)  # wolfram (simple) [works]
    # return Gamma0 - Gamma * (np.sqrt(Gamma0**2-1)/np.sqrt(Gamma**2-1)) # wolfram alpha [does no work]
    # return 0.5 / Gamma0 + 0.5 / Gamma ** 2 * (0.5 / Gamma0 - Gamma0 - 3. * Gamma0 / 8. / Gamma ** 2) # from the source  with beta assympotoe


def get_dGammaEff3dGamma(Gamma, gammaAdi3, dgamma43dGamma, gamma43):
    """ derived """
    return gammaAdi3 * (1. + Gamma ** -2) - dgamma43dGamma / 3. / gamma43 ** 2 * (Gamma - 1. / Gamma) - Gamma ** -2


def get_dGammadR_fs_rs(Gamma, Gamma0, gammaAdi, dlnrho1dR, M2, dM2dR,
                       dlnrho4dR, M3, dM3dR, Eint2, Eint3, gammaAdi3, gamma43_m1):
    # gamma43_minus_one = NavaEqs.gamma43_minus_one(Gamma, Gamma0, beta, beta0)
    # gammaAdi3 = (4 + 1 / (gamma43_minus_one + 1)) / 3.
    dgamma43dGamma = get_dgamma43dGamma(Gamma0, Gamma)  # 0.5 / Gamma0 + 0.5 / Gamma ** 2 * (0.5 / Gamma0 - Gamma0 - 3. * Gamma0 / 8. / Gamma ** 2)
    # . Using asymptotic approximation of beta
    GammaEff3 = get_GammaEff(Gamma, gammaAdi3)  # (gammaAdi3 * Gamma ** 2 - gammaAdi3 + 1) / Gamma # (gammaAdi * Gamma ** 2. - gammaAdi + 1.) / Gamma
    dGammaEff3dGamma = get_dGammaEff3dGamma(Gamma, gammaAdi3, dgamma43dGamma, gamma43_m1 + 1.)  # gammaAdi3 * (1. + Gamma ** -2) - dgamma43dGamma / 3. / (gamma43_m1 + 1.) ** 2 * (Gamma - 1. / Gamma) - Gamma ** -2

    dGammaEffdGamma = get_dGammaEffdGamma(Gamma, gammaAdi)  # 4. / 3. + 1. / Gamma ** 2. / 3. + 2. / Gamma ** 3. / 3.
    GammaEff = get_GammaEff(Gamma, gammaAdi)  # (gammaAdi * Gamma ** 2. - gammaAdi + 1.) / Gamma

    f_2 = GammaEff * (gammaAdi - 1.) * Eint2 / Gamma
    h_2 = GammaEff * (gammaAdi - 1.) * Eint2 * (dM2dR / M2 - dlnrho1dR)

    fh_factor3 = GammaEff3 * (gammaAdi3 - 1.) * Eint3
    f_3 = fh_factor3 / (gamma43_m1 + 1.) * dgamma43dGamma
    if Eint3 != 0:
        h_3 = fh_factor3 * (dM3dR / M3 - dlnrho4dR)
    else:
        h_3 = 0.
    dGammadR = -((Gamma - 1.) * (GammaEff + 1.) * dM2dR + (
            Gamma - Gamma0 + GammaEff3 * gamma43_m1) * dM3dR - h_2 - h_3) / (
                       (M2 + M3) + Eint2 * dGammaEffdGamma + Eint3 * dGammaEff3dGamma + f_2 + f_3)

    return dGammadR


def rhs(R, solution_array, pars_dict,
        eq_gammaAdi, eq_dmdr, eq_dthetadr):
    """
    [0]   1 / beta / cgs.c,
    [1]   1 / beta / Gamma / cgs.c,
    [2]   dGammadR,
    [3]   dEint2dR,
    [4]   dEint3dR,
    [5]   dthetadR,
    [6]   dErad2dR,
    [7]   dErad3dR,
    [8]   dEsh2dR,
    [9]   dEsh3dR,
    [10]  dEad2dR,
    [11]  dEad3dR,
    [12]  dM2dR,
    [13]  dM3dR,
    [14]  ddeltaR4dR
    :param R:
    :param solution_array:
    :param pars_dict:
    :return:
    """

    tburst = solution_array[0]
    tcomoving = solution_array[1]
    Gamma = solution_array[2]
    Eint2 = solution_array[3]
    Eint3 = solution_array[4]
    theta = solution_array[5]
    M2 = solution_array[12]
    M3 = solution_array[13]
    deltaR4 = solution_array[14]
    #
    M0 = pars_dict["M0"]
    Gamma0 = pars_dict["Gamma0"]
    theta0 = pars_dict["theta0"]
    rho = pars_dict["rho"] / M0
    dlnrho1dR = pars_dict["dlnrho1dR"]
    #
    if Eint2 < 0:
        beta = np.sqrt(1. - Gamma ** -2)
        print("ERROR! Eint2 < 0 Gamma < 1 [Gamma:{} beta:{} Eint2:{} Eint3:{} M2:{} M3:{}] Resetting to 0".format(
            Gamma, beta, Eint2, Eint3, M2, M3
        ))
        Eint2 = 0
    if Eint3 < 0:
        # logger.error("Warning: Eint3 = 0")
        Eint3 = 0.
    if Gamma < 1:
        Gamma = 1.0001
        print("ERROR Gamma < 1 [Gamma:{} Eint2:{} Eint3:{} M2:{} M3:{}] Resetting to 1.0001".format(
            Gamma, Eint2, Eint3, M2, M3
        ))
        # raise ValueError("Gamma < 1")
    if Gamma > Gamma0:
        print("ERROR! Gamma({}) > Gamma0({}) -- resetting to Gamma0"
              .format(Gamma, Gamma0))
        Gamma = Gamma0

    beta = np.sqrt(1. - np.power(Gamma, -2))
    beta0 = np.sqrt(1. - np.power(Gamma0, -2))

    # forward shock EOS
    # gammaAdi = Nava_fs_rhs.gammaAdi(Gamma, beta)  # (4 + 1 / Gamma) / 3.
    gammaAdi = eq_gammaAdi(Gamma, beta)

    # reverseShock EOS [ADIABASTIC]
    gamma43_minus_one = get_gamma43_minus_one(Gamma, Gamma0, beta, beta0, beta_switch=0.9999)
    # gammaAdi3 = Nava_fs_rhs.gammaAdi(gamma43_minus_one + 1.,
    #                                  np.sqrt(1. - np.power(float(gamma43_minus_one + 1.), -2)))
    gammaAdi3 = eq_gammaAdi(gamma43_minus_one + 1., get_beta(gamma43_minus_one + 1.))

    # Spreading
    dthetadR = eq_dthetadr(gammaAdi, Gamma, beta, R, theta,
                                  pars_dict["aa"], pars_dict["useSpread"], pars_dict["Rd"], pars_dict["thetaMax"])
    # dthetadR = Nava_fs_rhs.dthetadr(gammaAdi, Gamma, R, theta, pars["aa"]) * int(pars["useSpread"])

    # Densities and volumes
    # rho2 = 4. * Gamma * rho
    # V2 = M2 / rho2
    dM2dR = eq_dmdr(Gamma, R, pars_dict["thetaE"], theta, rho, aa=pars_dict["aa"]) / pars_dict["ncells"]
    # B = EquationsNava.B_func(Eint2, V2 / M0, pars["eB"])  # * M0 ** 2

    # Reverse Shock [DENSITY]
    if (not pars_dict["shutOff"]) and (
            Gamma < Gamma0):  # and (not M3 > 1.):# and (deltaR4 > 0): # the last assures that jet is decelerating
        alpha_of = pars_dict["tprompt"] * beta0 * cgs.c
        ddeltaR4dR = (1 / beta ** 4 - 1 / beta0 ** 4) / \
                     (1 / beta + 1 / beta0) / \
                     (1 / beta ** 2 + 1 / beta0 ** 2) * beta0
        rho4_scale_factor = np.float64(-deltaR4 / alpha_of)
        # rho4_factor = 1 / alpha_of * 1 / (2 * np.pi * R ** 2 * (1 - np.cos(theta0)))  # * pars["m_scale"] -- NO EFFECT
        # rho4 = rho4_factor * np.exp(rho4_scale_factor)  # Here exp() does not really make a difference
        dM3dR = (1 / alpha_of) * ddeltaR4dR * np.exp(rho4_scale_factor)  # [not / pars["m_scale"] -- changes everything]
        # rho3prim = 4 * Gamma * rho4
        # if not pars["shutOff"] and (M3 > 0) and (Eint3 > 0):
        # V3 = M3 / rho3prim
        # BRS = EquationsNava.B_func(Eint3 * M0 ** 2, V3, pars["eB3"])  # np.sqrt(8 * pi * pars.eB3) * np.sqrt(Eint3) / np.sqrt(V3) * c * np.sqrt(pars.M0)
        # else:
        # BRS = 0.
        dlnrho4dR = -2 / R - ddeltaR4dR / alpha_of
        # print('\t', np.exp(rho4_scale_factor))
        # print("Gamma:{:.1f} ddeltaR4dR:{:.3e} deltaR4:{:.3e} alpha_of:{:.3e} dM3dR:{:.3e} np.exp(rho4_scale_factor):{:.3e}".format(Gamma, ddeltaR4dR, deltaR4, alpha_of, dM3dR, np.exp(rho4_scale_factor)))
        # if np.exp(rho4_scale_factor) == 0.: exit(1)

    # elif False and sets["reverseShock"] and (Gamma < Gamma0) and (not M3 > 1.):
    #     alpha_of = pars["tprompt"] * beta0 * cgs.c
    #     ddeltaR4dR = (1 / beta ** 4 - 1 / beta0 ** 4) / \
    #                  (1 / beta + 1 / beta0) / \
    #                  (1 / beta ** 2 + 1 / beta0 ** 2) * beta0
    #
    #     bb0m1 = 0.5 / Gamma ** 2 + 0.5 / Gamma0 ** 2 + Gamma ** (-4) / 8 + Gamma0 ** (-4) / 8 - 0.25 / (Gamma0 ** 2) / (Gamma ** 2)
    #     dM3dR = 1 * (beta0 ** 2 - beta ** 2) / (beta0 + beta) / beta0 / pars["tprompt"] / beta / cgs.c / bb0m1
    #     rho4 = 1 / (2 * cgs.pi * R ** 2 *  pars["tprompt"] * (1-np.cos(theta)) * beta * cgs.c)
    #     rho3prim = 4 * Gamma * rho4
    #     if not sets["shutOff"] and (M3 > 0) and (Eint3 > 0):
    #         V3 = M3 / rho3prim
    #         BRS = EquationsNava.B_func(Eint3 * M0 ** 2, V3, pars["eB3"])  # np.sqrt(8 * pi * pars.eB3) * np.sqrt(Eint3) / np.sqrt(V3) * c * np.sqrt(pars.M0)
    #     else:
    #         BRS = 0.
    #     dlnrho4dR = -ddeltaR4dR / alpha_of - 2 / R
    else:
        ddeltaR4dR = 0.
        dM3dR = 0.
        rho4 = 0.
        BRS = 0.
        dlnrho4dR = 0.

    # dGammadR
    dGammadR = get_dGammadR_fs_rs(Gamma, Gamma0, gammaAdi, dlnrho1dR, M2, dM2dR,
                                       dlnrho4dR, M3, dM3dR, Eint2, Eint3, gammaAdi3, gamma43_minus_one)

    if dGammadR > 0:
        if Gamma > 0.95 * Gamma0:
            # raise ValueError("Gamma > 0.95 Gamma0 after RSISING")
            dGammadR = 0.

    # Energies
    dEsh2dR = (Gamma - 1.) * dM2dR  # Shocked energy

    # Expansion energy
    dlnV2dR = dM2dR / M2 - dlnrho1dR - dGammadR / Gamma
    if pars_dict["adiabLoss"]:
        dEad2dR = -(gammaAdi - 1.) * Eint2 * dlnV2dR
    else:
        dEad2dR = 0.

    # Radiative losses
    dErad2dR = pars_dict["epsilon_e_rad"] * dEsh2dR
    dErad3dR = pars_dict["epsilon_e_rad_RS"] * gamma43_minus_one * dM3dR

    assert np.isfinite(dErad2dR)

    if (not pars_dict["shutOff"]) and M3 > 0:  ### rho4 becomes 0 when RS injecta is cut-off
        ### Shocked energy
        dEsh3dR = gamma43_minus_one * dM3dR

        ### Expansion energy
        # dlnGamma43dR = dGammadR * dgamma43dGamma / (gamma43_minus_one+1)
        dlnV3dR = dM3dR / M3 - dlnrho4dR - dGammadR / Gamma

        if pars_dict["adiabLoss_RS"]:
            dEad3dR = -(gammaAdi3 - 1) * Eint3 * dlnV3dR  # dgamma43dGamma - h_3 / GammaEff3
        else:
            dEad3dR = 0.
    else:
        dEsh3dR = 0.
        dEad3dR = 0.

    dEint2dR = dEsh2dR + dEad2dR - dErad2dR  # / (pars.M0 * c ** 2)
    dEint3dR = dEsh3dR + dEad3dR - dErad3dR  # / (pars.M0 * c ** 2)

    return np.array([
        1 / beta / cgs.c,
        1 / beta / Gamma / cgs.c,
        dGammadR,
        dEint2dR,
        dEint3dR,
        dthetadR,
        dErad2dR,
        dErad3dR,
        dEsh2dR,
        dEsh3dR,
        dEad2dR,
        dEad3dR,
        dM2dR,
        dM3dR,
        ddeltaR4dR
    ])


class Driver_Nava_FSRS(Driver):

    def __init__(
            self,
            E0=1.e53,
            Gamma0=1000.,
            thetaE=0.,
            theta0=np.pi / 2.,
            M0=1.e53 * cgs.c ** -2 / 1000.,
            r_grid=np.logspace(18., 22., 500),
            rho0=1.e-2 * cgs.mppme,
            dlnrho1dR_0=0.,
            **kwargs
    ):

        assert "tprompt" in kwargs.keys()
        self.rs_shutOff_criterion_rho = 1e-50  # when density in RS falls below this, the RS computation stops

        # set equations to use in model
        eq_gammaAdi = opts_gammaAdi(kwargs["eq_gammaAdi"])
        eq_dmdr = opts_dmdr(kwargs["eq_dmdr"])
        eq_dthetadr = opts_dthetadr(kwargs["eq_dthetadr"])
        eq_rho2 = opts_rho2(kwargs["eq_rho2"])

        assert (not M0 is None)
        assert np.isfinite(kwargs["aa"])
        beta0 = get_beta(Gamma0)
        Rstart = r_grid[0]
        gammaAdi0 = eq_gammaAdi(Gamma0, beta0)

        # initial data for ODEs
        Rd = get_Rdec2(E0, rho0 / cgs.mppme, Gamma0)
        # (3. / (4. * cgs.pi) * 1. / (cgs.c ** 2. * cgs.mp) * E0 / (rho0 / cgs.mppme * Gamma0 ** 2.)) ** (1. / 3.)
        # m20 = (2. / 3.) * cgs.pi * (1. - np.cos(theta0)) * rho0 * Rstart ** 3. / kwargs["ncells"]
        M20 = (2 / 3.) * np.pi * Rstart ** 3. * (1 - np.cos(theta0)) * rho0 / kwargs["ncells"]

        # filling constant parameters for ODEs
        pars_ode_rhs = {
            "M0": M0,
            "theta0": theta0,
            "Gamma0": Gamma0,
            "rho": rho0,
            "dlnrho1dR": dlnrho1dR_0,
            "epsilon_e_rad": kwargs["epsilon_e_rad"],
            "useSpread": kwargs["useSpread"],
            "adiabLoss": kwargs["adiabLoss"],
            "ncells": kwargs["ncells"],
            "aa": kwargs["aa"],
            # reverse shock
            "tprompt": kwargs["tprompt"],
            "epsilon_e_rad_RS": kwargs["epsilon_e_rad_RS"],
            "adiabLoss_RS": kwargs["adiabLoss_RS"],
            "shutOff": False,
            "thetaMax":kwargs["thetaMax"],
            "Rd": Rd,
            "thetaE": 0.
        }

        # pars_ode_rhs = {
        #     "M0": M0,
        #     "aa": kwargs["aa"],
        #     "Rd": Rd,
        #     "ncells": kwargs["ncells"],
        #     "thetaE": thetaE,
        #     "rho": rho0,
        #     "dlnrho1dR": dlnrho1dR_0,
        #     "useSpread": kwargs["useSpread"],
        #     "thetaMax": kwargs["thetaMax"]
        # }

        # init_v_ns = ["tburst", "tcomoving", "Gamma", "Eint2", "theta", "Erad2", "Esh2", "Ead2", "M2"]
        # all_v_ns = init_v_ns + ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2", "gammaAdi"]
        init_v_ns = ["tburst", "tcomoving", "Gamma", "Eint2", "Eint3", "theta", "Erad2", "Erad3",
                     "Esh2", "Esh3", "Ead2", "Ead3", "M2", "M3", "deltaR4"]
        all_v_ns = init_v_ns + ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2",
                                "rho4", "Gamma43", "U_e_RS", "thickness_RS", "rho3", "gammaAdi", "gammaAdi3"]
        # storage for solutions
        dtypes = []
        for v_n in all_v_ns:
            dtypes.append((v_n, 'f8'))
        vals = np.zeros(len(r_grid), dtype=dtypes)

        # filling initial data
        vals['tburst'][0] = Rstart / (get_beta(Gamma0) * cgs.c)  # 0
        vals['tcomoving'][0] = Rstart / (get_beta(Gamma0) * cgs.c) / Gamma0  # 1
        vals['Gamma'][0] = Gamma0  # 2
        vals['Eint2'][0] = (Gamma0 - 1) * M20 / M0  # 3
        vals['Eint3'][0] = 0.  # 4
        vals['theta'][0] = theta0  # 5
        vals['Erad2'][0] = 0.  # 6
        vals['Erad3'][0] = 0.  # 7
        vals['Esh2'][0] = 0.  # 8
        vals['Esh3'][0] = 0.  # 9
        vals['Ead2'][0] = 0.  # 10
        vals['Ead3'][0] = 0.  # 11
        vals['M2'][0] = M20 / M0  # 12
        vals['M3'][0] = 0.  # 13
        vals['deltaR4'][0] = 0.  # 14

        # filling non-ODE initial values
        vals['rho'][0] = rho0
        vals['tt'][0] = self.init_elapsed_time(Rstart, beta0, Gamma0, kwargs['useSpread'])  # tt0
        vals['R'][0] = Rstart
        vals['thickness'][0] = 0.
        vals['U_e'][0] = get_U_e(rho0, Gamma0, M20 / M0, (Gamma0 - 1) * M20 / M0) * cgs.c**2 # get_U_e(rho0, Gamma0, M20 / M0, (Gamma0 - 1) * M20 / M0)
        vals['beta'][0] = beta0
        vals['rho2'][0] = eq_rho2(Gamma0, beta0, rho0, gammaAdi0) #eq_rho2(Gamma0, beta0, rho0, gammaAdi0)
        vals['rho4'][0] = 0.
        vals['Gamma43'][0] = 0.
        vals['U_e_RS'][0] = 0.
        vals['thickness_RS'][0] = 0.
        vals['gammaAdi'][0] = gammaAdi0
        vals['gammaAdi3'][0] = 0.

        # all_v_ns = init_v_ns + ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2",
        #                         "rho4", "Gamma43", "U_e_RS", "thickness_RS", "rho3", "gammaAdi", "gammaAdi3"]
        # np.array([rho0, tt0, Rstart, 0., 0., beta0, rho20,
        #           0., 0., 0., 0., 0.,
        #           gammaAdi0, 0.])
        # Rescale the values to 'cgs'
        # self.vals[0, :] = self.apply_units(self.vals[0, :])

        super(Driver_Nava_FSRS, self).__init__(r_grid, rhs, vals, pars_ode_rhs, init_v_ns, all_v_ns, **kwargs)





















        # assert np.isfinite(kwargs["aa"])
        # self.rs_shutOff_criterion_rho = 1e-50 # when density in RS falls below this, the RS computation stops
        #
        # # some equations for ODEs
        # self.eqs_ode_rhs = {"eq_dthetadr": kwargs["eq_dthetadr"],
        #                     "eq_gammaAdi": kwargs["eq_gammaAdi"]}
        #
        #
        # assert (not M0 is None)
        # beta0 = get_beta(Gamma0)
        #
        # Rd = get_Rdec2(E0, rho0/cgs.mppme, Gamma0)
        # M20 = (2 / 3.) * np.pi * Rstart ** 3. * (1 - np.cos(theta0)) * rho0 / kwargs["ncells"]
        #
        # self.v_ns_init_vals = [
        #     "tburst", "tcomoving", "Gamma",
        #     "Eint2", "Eint3", "theta", "Erad2", "Erad3", "Esh2", "Esh3", "Ead2", "Ead3",
        #     "M2", "M3", "deltaR4"
        # ]
        #
        # init_data = {
        #     "tburst": Rstart / (get_beta(Gamma0) * cgs.c),  # 0
        #     "tcomoving": Rstart / (get_beta(Gamma0) * cgs.c) / Gamma0,  # 1
        #     "Gamma": Gamma0,  # 2
        #     "Eint2": (Gamma0 - 1) * M20 / M0,  # 3
        #     "Eint3": 0.,  # 4
        #     "theta": theta0,  # 5
        #     "Erad2": 0.,  # 6
        #     "Erad3": 0.,  # 7
        #     "Esh2": 0.,  # 8
        #     "Esh3": 0.,  # 9
        #     "Ead2": 0.,  # 10
        #     "Ead3": 0.,  # 11
        #     "M2": M20 / M0,  # 12
        #     "M3": 0.,  # 13
        #     "deltaR4": 0.  # 14
        # }
        # self.initial_data = self._set_ode_inits(init_data)
        #
        # self.pars_ode_rhs = {
        #     "M0": M0, "theta0": theta0, "Gamma0": Gamma0, "rho": rho0, "dlnrho1dR": dlnrho1dR_0,
        #     "epsilon_e_rad": kwargs["epsilon_e_rad"], "useSpread": kwargs["useSpread"],
        #     "adiabLoss": kwargs["adiabLoss"],
        #     "ncells": kwargs["ncells"], "aa": kwargs["aa"],
        #     # reverse shock
        #     "tprompt": kwargs["tprompt"],
        #     "epsilon_e_rad_RS": kwargs["epsilon_e_rad_RS"],
        #     "adiabLoss_RS": kwargs["adiabLoss_RS"],
        #     "shutOff": False,
        #     "thetaMax":kwargs["thetaMax"],
        #     "Rd": Rd,
        #     "thetaE": 0.
        # }
        #
        # # initialize time elapsed in the comoving frame
        # gammaAdi0 = kwargs["eq_gammaAdi"](Gamma0, beta0)
        # rho20 = kwargs["eq_rhoprime"](Gamma0, beta0, rho0, gammaAdi0)  # self.get_rhoprime(rho0, Gamma0)
        # tt0 = self.init_elapsed_time(Rstart, beta0, Gamma0, **kwargs)
        # self.all_v_ns = self.v_ns_init_vals + \
        #                 ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2",
        #                  "rho4", "Gamma43", "U_e_RS", "thickness_RS", "rho3",
        #                  "gammaAdi", "gammaAdi3"]
        # self.vals = np.zeros((1, len(self.all_v_ns)))
        # self.vals[0, :len(self.initial_data)] = self.initial_data
        # self.vals[0, len(self.initial_data):] = np.array([rho0, tt0, Rstart, 0., 0., beta0, rho20,
        #                                                       0., 0., 0., 0., 0.,
        #                                                       gammaAdi0, 0.])
        # self.vals[0, self.i_nv("U_e")] = self.get_U_e(idx=0) * cgs.c**2
        # self.vals[0] = self.apply_units(self.vals[0])
        # # self.vals = np.hstack((self.apply_units(np.copy(self.initial_data)),
        # #                            np.array([rho0, tt0, Rstart, 0., 0., beta0, 0., 0., 0., 0.])))
        #
        # self.rhs = Nava_fs_rs_rhs()
        #
        # self.kwargs = kwargs
        #
        # super(Driver_Nava_FSRS, self).__init__(Rstart, kwargs["ode_rtol"], kwargs["ode_nsteps"])

    # def get_U_e(self,idx=-1):
    #     return get_U_e(self.)
    #
    #     rho = self.get("rho")[idx]
    #     Gamma = self.get("Gamma")[idx]
    #     M2 = self.get("M2")[idx]
    #     Eint2 = self.get("Eint2")[idx]
    #
    #     rhoprim = 4. * rho * Gamma  # comoving density
    #     V2 = M2 / rhoprim  # comoving volume
    #     U_e = Eint2 / V2  # comoving energy density (electrons)
    #     # U_b = eps_b * U_e  # comoving energy density (MF)
    #     return U_e

    def _additional_quantities(self, idx):

        super(Driver_Nava_FSRS, self)._additional_quantities(idx)

        if (not self.pars_ode_rhs["shutOff"]):

            alpha_of = self.pars_ode_rhs["tprompt"] * self.vals["beta"][0] * cgs.c
            rho4_fac_1 = self.pars_ode_rhs["M0"] / (2 * alpha_of * np.pi * (1. - np.cos(self.vals["theta"][0])))
            rho4_fac = rho4_fac_1 / np.power(self.vals["R"][idx], 2)
            rho4 = rho4_fac * np.exp(-self.vals["deltaR4"][idx] / alpha_of)
            self.vals["rho4"][idx] = rho4

            gamma43 = get_gamma43_minus_one(self.vals["Gamma"][idx],
                                            self.vals["Gamma"][0],
                                            self.vals["beta"][idx],
                                            self.vals["beta"][0],
                                            beta_switch=0.9999)
            self.vals["Gamma43"][idx] = gamma43 + np.float64(1.)
            self.vals["U_e_RS"][idx] = self.get_U_e_rs(idx)

            gammaAdi3 = self.eq_gammaAdi(gamma43 + 1., get_beta(gamma43 + 1.))
            self.vals["gammaAdi3"][idx] = np.float64(gammaAdi3)

            rho3prim = self.eq_rho2(self.vals["Gamma"][idx],
                                    self.vals["beta"][idx],
                                    self.vals["rho4"][idx],
                                    self.vals["gammaAdi3"][idx])
            self.vals["rho3"][idx] = rho3prim

            thickness_RS = self.eq_delta(self.vals["M3"][idx],
                                         self.vals["rho3"][idx],
                                         self.vals["theta"][idx],
                                         self.vals["Gamma"][idx],
                                         self.vals["R"][idx],
                                         self.pars_ode_rhs["ncells"])
            self.vals["thickness_RS"][idx] = thickness_RS

            # alpha_of = self.kwargs["tprompt"] * self.get("beta")[0] * cgs.c
            # rho4_fac_1 = self.pars_ode_rhs["M0"] / (2 * alpha_of * np.pi * (1.-np.cos(self.get("theta")[0])))
            # rho4_fac = rho4_fac_1 / np.power(self.get("R")[-1], 2)
            # rho4 = rho4_fac * np.exp(-self.get("deltaR4")[-1] / alpha_of)

            # self.set_last("rho4", rho4)

            # gamma43 = Nava_fs_rs_rhs.gamma43_minus_one(self.get("Gamma")[-1],self.get("Gamma")[0],
            #                                            self.get("beta")[-1], self.get("beta")[0], beta_switch=0.9999)
            #
            # self.set_last("Gamma43", np.float64(gamma43+1.))
            # self.set_last("U_e_RS", self.get_U_e_rs())

            # gammaAdi3 = self.kwargs["eq_gammaAdi"](gamma43+1., get_beta(gamma43+1.))
            # self.set_last("gammaAdi3", np.float64(gammaAdi3))

            # rho3prim = self.kwargs["eq_rhoprime"](self.get_last("Gamma"), self.get_last("beta"),
            #                                       self.get_last("rho4"), self.get_last("gammaAdi3")) ## SURE it is gammaAdi3 ????
            # self.set_last("rho3", rho3prim)
            #
            # # rho3prim = self.get_rhoprime(self.get_last("rho4"), self.get_last("Gamma"))
            # # nprim3 = rho3prim / cgs.mp
            # # thickness_RS = self.get("M3")[-1] / (8 * np.pi * (1. - np.cos(self.get("theta")[-1])) *
            # #                                      self.get("Gamma")[-1] ** 2 * rho4 * self.get("R")[-1] ** 2)
            # thickness_RS = EqOpts.shock_thickness(self.get_last("M3"),
            #                                       self.get_last("rho3"),
            #                                       self.get_last("theta"),
            #                                       self.get_last("Gamma"),
            #                                       self.get_last("R"),
            #                                       self.kwargs["ncells"])
            # self.set_last("thickness_RS", thickness_RS)

        # if self.pars_ode_rhs["shutOff"]: self.set("rho3", 0., -1)
        # if self.pars_ode_rhs["shutOff"]: self.set("Eint3", 0., -1)
        # if self.pars_ode_rhs["shutOff"]: self.set("Ead3", 0., -1)
        # if self.pars_ode_rhs["shutOff"]: self.set("Erad3", 0., -1)
        # if self.pars_ode_rhs["shutOff"]: self.set("Esh3", 0., -1)
        # if self.pars_ode_rhs["shutOff"]: self.set("M3", 0., -1)
        # if self.pars_ode_rhs["shutOff"]: self.set("deltaR4", 0., -1)

    def evolove(self, R, rho, dlnrho1dR):

        super(Driver_Nava_FSRS, self).evolove(R, rho, dlnrho1dR)

        idx = int(np.where(self.r_grid == R)[0][0])

        if (self.vals["Gamma"][idx] < self.vals["Gamma"][0]) & \
                (self.vals["rho4"][idx] < self.rs_shutOff_criterion_rho):
            self.pars_ode_rhs["shutOff"] = True

        # self.odeinstance.set_f_params(self.pars_ode_rhs)

    def get_U_e(self, idx=-1):

        return get_U_e(self.vals["rho"][idx],
                       self.vals["Gamma"][idx],
                       self.vals["M2"][idx],
                       self.vals["Eint2"][idx])

    def get_U_e_rs(self, idx):
        rho3 = 4. * self.vals["Gamma"][idx] * self.vals["rho4"][idx] # IS it correct iwth Gamma and not Gamma43?
        V3 = self.vals["M3"][idx] / rho3      # comoving volume
        U_e = self.vals["Eint3"][idx] / V3    # comoving energy density (electrons)
        # U_b = eps_b * U_e  # comoving energy density (MF)
        return U_e

    def apply_units(self, idx):

        self.vals["M2"][idx] *= self.pars_ode_rhs["M0"]
        self.vals["Eint2"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2
        self.vals["Erad2"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2
        self.vals["Esh2"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2
        self.vals["Ead2"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2

        self.vals["M3"][idx] *= self.pars_ode_rhs["M0"]
        self.vals["Eint3"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2
        self.vals["Erad3"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2
        self.vals["Esh3"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2
        self.vals["Ead3"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2



        # self.units_dic = {
        #     "M2": M0,
        #     "Eint2": M0 * cgs.c**2,
        #     "Erad2": M0 * cgs.c**2,
        #     "Esh2": M0 * cgs.c**2,
        #     "Ead2": M0 * cgs.c**2,
        #     "M3": M0,
        #     "Eint3": M0 * cgs.c ** 2,
        #     "Erad3": M0 * cgs.c ** 2,
        #     "Esh3": M0 * cgs.c ** 2,
        #     "Ead3": M0 * cgs.c ** 2,
        # }
        # i_res[self.i_nv("M2")] *= self.pars_ode_rhs["M0"]
        # i_res[self.i_nv("Eint2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # i_res[self.i_nv("Erad2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # i_res[self.i_nv("Esh2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # i_res[self.i_nv("Ead2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        #
        # i_res[self.i_nv("M3")] *= self.pars_ode_rhs["M0"]
        # i_res[self.i_nv("Eint3")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # i_res[self.i_nv("Erad3")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # i_res[self.i_nv("Esh3")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # i_res[self.i_nv("Ead3")] *= self.pars_ode_rhs["M0"] * cgs.c**2

        # return i_res

if __name__ == '__main__':
    rho = 1e-2 * cgs.mppme
    RR = np.logspace(14., 23., 1000)
    o = Driver_Nava_FSRS(
        E0=1.e53,
        Gamma0=1000.,
        thetaE=0.,
        theta0=np.pi / 2.,
        M0=1.e53 * cgs.c ** -2 / 1000.,
        r_grid=RR,
        rho0=rho,
        dlnrho1dR_0=0.,
        # kwargs
        useSpread=False, aa=-1., ncells=1, ode='dop853', ode_pars={"rtol": 1e-8, "nsteps": 1000, "first_step": RR[0]},
        eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
        thetaMax=np.pi / 2.,
        adiabLoss=True, epsilon_e_rad=0., epsilon_e_rad_RS=0., tprompt=1e3, adiabLoss_RS=True
    )

    # dyn = Driver_Peer_FS(E0=1e53, M0=1e53 / (cgs.c ** 2 * 1000), Rstart=RR[0], rho0=rho,
    #                      useSpread=False, aa=-1., ncells=1,
    #                      ode_rtol=1e-4, ode_nsteps=1000, eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi/2.)
    for i in range(1, len(RR)):
        # dyn.update_ode_pars(rho=rho)
        rho, dlnrho1dR = rho_dlnrho1dR(RR[i], 1e-2, None, None, None, None)
        o.evolove(RR[i], rho, dlnrho1dR)

    import matplotlib.pyplot as plt

    plt.loglog(o.vals["R"], o.vals["Gamma"], label="P")
    # plt.loglog(dyn2.get("R"), dyn2.get("Gamma"), label="N1")
    # plt.loglog(dyn3.get("R"), dyn3.get("Gamma"), label="N2")
    plt.legend()
    plt.show()