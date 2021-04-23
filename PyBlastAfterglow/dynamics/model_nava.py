import numpy as np
import numba as nb
from PyBlastAfterglow.uutils import cgs
from .dynamics import Driver
from .eqs import (get_beta, get_Rdec2, rho_dlnrho1dR)
from .eqs_opts import (opts_dthetadr, opts_dmdr, opts_gammaAdi, opts_rho2, opts_shock_thickness)




# @staticmethod
# def gammaAdi(Gamma, beta):
#     """ From Peer 2012 arxiv:1203.5797 """
#     mom = Gamma * beta
#     theta = mom / 3. * (mom + 1.07 * mom ** 2.) / (1 + mom + 1.07 * mom ** 2.)
#     zz = theta / (0.24 + theta)
#     gamma_adi = (5. - 1.21937 * zz + 0.18203 * zz ** 2. - 0.96583 * zz ** 3. + 2.32513 * zz ** 4. -
#             2.39332 * zz ** 5. + 1.07136 * zz ** 6.) / 3.
#     return gamma_adi

# @staticmethod
# def gammaAdi(Gamma):
#     return (4 + 1 / Gamma) / 3.

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


def get_GammaEff(Gamma, gammaAdi):
    return (gammaAdi * Gamma ** 2. - gammaAdi + 1.) / Gamma

def get_dGammaEffdGamma(Gamma, gammaAdi):
    return (gammaAdi * Gamma ** 2. + gammaAdi - 1.) / Gamma ** 2
    # return 4. / 3. + 1. / Gamma ** 2. / 3. + 2. / Gamma ** 3. / 3.

def dGammadR_fs(Gamma, gammaAdi, dlnrho1dR, M2, dM2dR, Eint2):
    GammaEff = get_GammaEff(Gamma, gammaAdi)  # , gammaAdi) #(gammaAdi * Gamma ** 2. - gammaAdi + 1.) / Gamma
    dGammaEffdGamma = get_dGammaEffdGamma(Gamma, gammaAdi)  # 4. / 3. + 1. / Gamma ** 2. / 3. + 2. / Gamma ** 3. / 3.

    f_2 = GammaEff * (gammaAdi - 1.) * Eint2 / Gamma
    h_2 = GammaEff * (gammaAdi - 1.) * Eint2 * (dM2dR / M2 - dlnrho1dR)

    dGammadR = -((Gamma - 1.) * (GammaEff + 1.) * dM2dR - h_2) / ((1. + M2) + Eint2 * dGammaEffdGamma + f_2)

    return dGammadR


def rhs(R, solution_array, pars_dict,
        eq_gammaAdi, eq_dmdr, eq_dthetadr):
    """
    [0] 1 / beta / cgs.c,
    [1] 1 / beta / Gamma / cgs.c,
    [2] dGammadR,
    [3] dEint2dR,
    [4] dthetadR,
    [5] dErad2dR,
    [6] dEsh2dR,
    [7] dEad2dR,
    [8] dM2dR
    :param R:
    :param solution_array:
    :param pars_dict:
    :return:
    """

    tburst = solution_array[0]
    tcomoving = solution_array[1]
    Gamma = solution_array[2]
    Eint2 = solution_array[3]
    theta = solution_array[4]
    M2 = solution_array[8]
    #
    M0 = pars_dict["M0"]
    Gamma0 = pars_dict["Gamma0"]
    theta0 = pars_dict["theta0"]
    rho = pars_dict["rho"] / M0
    dlnrho1dR = pars_dict["dlnrho1dR"]
    #
    # if Eint2 < 0:
    #     beta = np.sqrt(1. - Gamma ** -2)
    #     print("ERROR! Eint2 < 0 Gamma < 1 [Gamma:{} beta:{} Eint2:{} M2:{} ] Resetting to 0".format(
    #         Gamma, beta, Eint2, M2,
    #     ))
    #     Eint2 = 0
    # if Gamma < 1:
    #     Gamma = 1.0001
    #     print("ERROR Gamma < 1 [Gamma:{} Eint2:{} M2:{} ] Resetting to 1.0001".format(
    #         Gamma, Eint2,  M2,
    #     ))
    #     # raise ValueError("Gamma < 1")
    # if Gamma > Gamma0:
    #     print("ERROR! Gamma({}) > Gamma0({}) -- resetting to Gamma0"
    #           .format(Gamma, Gamma0))
    #     Gamma = Gamma0

    #  one_over_beta = 1. / np.power(1 - np.power(Gamma, -2.), 0.5)
    beta = np.sqrt(1. - np.power(Gamma, -2))
    # beta0 = np.sqrt(1. - Gamma0 ** -2)

    gammaAdi = eq_gammaAdi(Gamma, beta)
    # gammaAdi = self.gammaAdi(Gamma, beta)  # (4 + 1 / Gamma) / 3.

    # one_minus_costheta = EquationsNava.one_minus_costheta(theta)
    # one_minus_costheta0 = 1. - np.cos(theta0)

    # Spreading
    # dthetadR = self.dthetadr(gammaAdi, Gamma, R, theta, pars["aa"]) * int(pars["useSpread"])
    dthetadR = eq_dthetadr(gammaAdi, Gamma, beta, R, theta, pars_dict["aa"],
                                  pars_dict["useSpread"], pars_dict["Rd"], pars_dict["thetaMax"])
    # Densities and volumes

    # rho2 = 4. * Gamma * rho
    # V2 = M2 / rho2
    dM2dR = eq_dmdr(Gamma, R, pars_dict["thetaE"], theta, rho, aa=pars_dict["aa"]) / pars_dict[
        "ncells"]  # 2 * cgs.pi * R ** 2. * rho * one_minus_costheta / (pars["m_scale"])

    # # # # dGammadR
    dGammadR = dGammadR_fs(Gamma, gammaAdi, dlnrho1dR, M2, dM2dR, Eint2)

    if dGammadR > 0:
        if Gamma > 0.95 * Gamma0:
            # raise ValueError("Gamma > 0.95 Gamma0 after RSISING")
            dGammadR = 0.

    # # # # Energies # # # #
    dEsh2dR = (Gamma - 1.) * dM2dR  # Shocked energy

    # --- Expansion energy
    dlnV2dR = dM2dR / M2 - dlnrho1dR - dGammadR / Gamma
    if pars_dict["adiabLoss"]:
        dEad2dR = -(gammaAdi - 1.) * Eint2 * dlnV2dR
    else:
        dEad2dR = 0.

    # -- Radiative losses
    dErad2dR = pars_dict["epsilon_e_rad"] * dEsh2dR

    ### -- Energy equation
    dEint2dR = dEsh2dR + dEad2dR - dErad2dR  # / (pars.M0 * c ** 2)

    return np.array([1 / beta / cgs.c,
                     1 / beta / Gamma / cgs.c,
                     dGammadR,
                     dEint2dR,
                     dthetadR,
                     dErad2dR,
                     dEsh2dR,
                     dEad2dR,
                     dM2dR
                     ])

def get_U_e(rho, Gamma, M2, Eint2):

    rhoprim = 4. * rho * Gamma  # comoving density
    V2 = M2 / rhoprim  # comoving volume
    U_e = Eint2 / V2  # comoving energy density (electrons)
    # U_b = eps_b * U_e  # comoving energy density (MF)
    return U_e

class Driver_Nava_FS(Driver):

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
            "Rd": Rd,
            "thetaMax":kwargs["thetaMax"],
            "thetaE":0.
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

        init_v_ns = ["tburst", "tcomoving", "Gamma", "Eint2", "theta", "Erad2", "Esh2", "Ead2", "M2"] # For ODE
        all_v_ns = init_v_ns + ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2", "gammaAdi"] # all
        # storage for solutions
        dtypes = []
        for v_n in all_v_ns:
            dtypes.append((v_n, 'f8'))
        vals = np.zeros(len(r_grid), dtype=dtypes)

        # filling initial data
        vals['tburst'][0] = Rstart / (beta0 * cgs.c)  # 0
        vals['tcomoving'][0] = Rstart / (beta0 * Gamma0 * cgs.c)  # 1
        vals['Gamma'][0] = Gamma0
        vals['Eint2'][0] = (Gamma0 - 1) * M20 / M0
        vals['theta'][0] = theta0
        vals['Erad2'][0] = 0.
        vals['Esh2'][0] = 0.
        vals['Ead2'][0] = 0.
        vals['M2'][0] = M20 / M0

        # filling non-ODE initial values
        vals['rho'][0] = rho0
        vals['tt'][0] = self.init_elapsed_time(Rstart, beta0, Gamma0, kwargs['useSpread'])  # tt0
        vals['R'][0] = Rstart
        vals['thickness'][0] = 0.
        vals['beta'][0] = beta0
        vals['gammaAdi'][0] = gammaAdi0
        vals['rho2'][0] = eq_rho2(Gamma0, beta0, rho0, gammaAdi0)
        vals['U_e'][0] = get_U_e(rho0, Gamma0, M20 / M0, (Gamma0 - 1) * M20 / M0) * cgs.c**2 # get_U_e(rho0, Gamma0, M20 / M0, (Gamma0 - 1) * M20 / M0)
            # get_U_e(rho0, Gamma0, M20 / M0, (Gamma0 - 1) * M20) * cgs.c ** 2  # self.get_U_e(idx=0)

        # Rescale the values to 'cgs'
        # self.vals[0, :] = self.apply_units(self.vals[0, :])

        super(Driver_Nava_FS, self).__init__(r_grid, rhs, vals, pars_ode_rhs, init_v_ns, all_v_ns, **kwargs)


        # assert (not M0 is None)
        # assert np.isfinite(kwargs["aa"])
        # beta0 = get_beta(Gamma0)
        #
        # # some equations for ODEs
        # self.eqs_ode_rhs = {"eq_dthetadr": kwargs["eq_dthetadr"],
        #                     "eq_gammaAdi": kwargs["eq_gammaAdi"]}
        #
        # Rd = get_Rdec2(E0, rho0 / cgs.mppme, Gamma0)
        # M20 = (2 / 3.) * np.pi * Rstart ** 3. * (1 - np.cos(theta0)) * rho0 / kwargs["ncells"]
        #
        # self.v_ns_init_vals = [
        #     "tburst", "tcomoving", "Gamma", "Eint2", "theta", "Erad2", "Esh2", "Ead2", "M2"
        # ]
        # init_data = {
        #     "tburst": Rstart / (beta0 * cgs.c),  # 0
        #     "tcomoving": Rstart / (beta0 * Gamma0 * cgs.c),  # 1
        #     "Gamma": Gamma0,  # 2
        #     "Eint2": (Gamma0 - 1) * M20 / M0,  # 3 # 0.75 *
        #     "theta": theta0,  # 4
        #     "Erad2": 0.,  # 5
        #     "Esh2": 0.,  # 6
        #     "Ead2": 0.,  # 7
        #     "M2": M20 / M0,  # 8
        # }
        # self.initial_data = self._set_ode_inits(init_data)
        #
        # self.pars_ode_rhs = {
        #     "M0": M0, "theta0": theta0, "Gamma0": Gamma0, "rho": rho0, "dlnrho1dR": dlnrho1dR_0,
        #     "epsilon_e_rad": kwargs["epsilon_e_rad"], "useSpread": kwargs["useSpread"],
        #     "adiabLoss": kwargs["adiabLoss"],
        #     "ncells": kwargs["ncells"], "aa": kwargs["aa"],
        #     "Rd": Rd, "thetaMax":kwargs["thetaMax"], "thetaE":0.
        # }
        #
        # # initialize time elapsed in the comoving frame
        # tt0 = self.init_elapsed_time(Rstart, beta0, Gamma0, **kwargs)
        # gammaAdi0 = kwargs["eq_gammaAdi"](Gamma0, beta0)
        # rho20 = kwargs["eq_rhoprime"](Gamma0, beta0, rho0, gammaAdi0)#self.get_rhoprime(rho0, Gamma0)
        # self.all_v_ns = self.v_ns_init_vals + ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2", "gammaAdi"]
        # self.vals = np.zeros((1, len(self.all_v_ns)))
        # self.vals[0, :len(self.initial_data)] = self.initial_data
        # self.vals[0, len(self.initial_data):] = np.array([rho0, tt0, Rstart, 0., 0., beta0, rho20, gammaAdi0])
        # self.vals[0, self.i_nv("U_e")] = self.get_U_e(idx=0) * cgs.c**2
        # self.vals[0] = self.apply_units(self.vals[0])
        # # self.vals = np.hstack((self.apply_units(np.copy(self.initial_data)),
        # #                            np.array([rho0, tt0, Rstart, 0., 0., beta0])))
        #
        # self.rhs = Nava_fs_rhs()
        #
        # super(Driver_Nava_FS, self).__init__(Rstart, kwargs["ode_rtol"], kwargs["ode_nsteps"])
        #
        # self.kwargs = kwargs

    def get_U_e(self, idx=-1):

        return get_U_e(self.vals["rho"][idx],
                       self.vals["Gamma"][idx],
                       self.vals["M2"][idx],
                       self.vals["Eint2"][idx])


        # rhoprim = 4. * rho * Gamma  # comoving density
        # V2 = M2 / rhoprim  # comoving volume
        # U_e = Eint2 / V2  # comoving energy density (electrons)
        # # U_b = eps_b * U_e  # comoving energy density (MF)
        # return U_e

    def apply_units(self, idx):

        self.vals["M2"][idx] *= self.pars_ode_rhs["M0"]
        self.vals["Eint2"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2
        self.vals["Erad2"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2
        self.vals["Esh2"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2
        self.vals["Ead2"][idx] *= self.pars_ode_rhs["M0"] * cgs.c**2

        # self.units_dic = {
        #     "M2": M0,
        #     "Eint2": M0 * cgs.c**2,
        #     "Erad2": M0 * cgs.c**2,
        #     "Esh2": M0 * cgs.c**2,
        #     "Ead2": M0 * cgs.c**2,
        # }
        # i_res[self.i_nv("M2")] *= self.pars_ode_rhs["M0"]
        # i_res[self.i_nv("Eint2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # i_res[self.i_nv("Erad2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # i_res[self.i_nv("Esh2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # i_res[self.i_nv("Ead2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
        # return i_res


if __name__ == '__main__':
    rho = 1e-2 * cgs.mppme
    RR = np.logspace(14., 23., 1000)
    o = Driver_Nava_FS(
        E0=1.e53,
        Gamma0=1000.,
        thetaE=0.,
        theta0=np.pi / 2.,
        M0=1.e53 * cgs.c ** -2 / 1000.,
        r_grid=RR,
        rho0=rho,
        dlnrho1dR_0=0.,
        # kwargs
        useSpread=False, aa=-1., ncells=1, ode='dop853', ode_pars={"rtol":1e-5, "nsteps":1000, "first_step":RR[0]},
        eq_delta="default", eq_dthetadr="None", eq_dmdr="default", eq_gammaAdi="Nava", eq_rho2="rel",
        thetaMax=np.pi / 2.,
        adiabLoss=True, epsilon_e_rad=0.,
    )


    # dyn = Driver_Peer_FS(E0=1e53, M0=1e53 / (cgs.c ** 2 * 1000), Rstart=RR[0], rho0=rho,
    #                      useSpread=False, aa=-1., ncells=1,
    #                      ode_rtol=1e-4, ode_nsteps=1000, eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi/2.)
    for i in range(1,len(RR)):
        # dyn.update_ode_pars(rho=rho)
        rho, dlnrho1dR = rho_dlnrho1dR(RR[i], 1e-2, None, None, None, None)
        o.evolove(RR[i],rho, dlnrho1dR)

    import matplotlib.pyplot as plt
    plt.semilogx(o.vals["R"], o.vals["Gamma"], label="P")
    # plt.loglog(dyn2.get("R"), dyn2.get("Gamma"), label="N1")
    # plt.loglog(dyn3.get("R"), dyn3.get("Gamma"), label="N2")
    plt.legend()
    plt.show()