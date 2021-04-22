"""
    Dynamical modulus for blast wave
"""

import numpy as np
import numba as nb
from scipy.integrate import ode

from PyBlastAfterglow.uutils import cgs
from .eqs import get_beta, get_Rdec2, rho_dlnrho1dR
from .eqs_opts import (opts_dthetadr, opts_dmdr, opts_gammaAdi, opts_rho2, opts_shock_thickness)

# TODO 1. check 'solver_GP12()' in joelib
# TODO see 'get_shock_front_thickness' -- fix the cos(theta). It is 2pi by default but it SHOULD follow the jet openning
# TODO check if 'shock_thickness' is correctly computed in Gamma >> 1 and Gamma ~ 1 and for theta different.
# TODO assure that rho2 changes between Gamma >> 1 and Gamma ~ 1 regimes (compresion ration changes)

class Driver:

    def __init__(
            self, r_grid, rhs, dynamics, pars_ode_rhs, init_v_ns, all_v_ns, **kwargs
        ):

        # setting for all inheriting classes
        self.r_grid = r_grid
        self.Rstart, self.Rend, self.nR = r_grid[0], r_grid[1], r_grid[2]
        self.dynamics = dynamics
        self.pars_ode_rhs = pars_ode_rhs
        self.init_v_ns = init_v_ns
        self.all_v_ns = all_v_ns
        self.odeinstance = ode(rhs)

        # set equations to use in model
        self.eq_gammaAdi = opts_gammaAdi(kwargs["eq_gammaAdi"])
        self.eq_dmdr = opts_dmdr(kwargs["eq_dmdr"])
        self.eq_dthetadr = opts_dthetadr(kwargs["eq_dthetadr"])
        self.eq_rho2 = opts_rho2(kwargs["eq_rho2"])
        self.eq_delta = opts_shock_thickness(kwargs["eq_delta"])


        self.odeinstance.set_integrator(kwargs['ode'], **kwargs['ode_pars'])
        self.odeinstance.set_f_params(self.pars_ode_rhs,
                                      self.eq_gammaAdi, self.eq_dmdr, self.eq_dthetadr)

        init_vals = np.zeros(len(init_v_ns))
        for i_v_n, v_n in enumerate(self.init_v_ns):
            init_vals[i_v_n] = self.dynamics[v_n][0]
        self.odeinstance.set_initial_value(init_vals, self.Rstart * 0.9999)


        #
        # self.r_grid = r_grid
        # self.Rstart, self.Rend, self.nR = r_grid[0], r_grid[1], r_grid[2]
        #
        # self.rhs = rhs
        # self.pars_ode_rhs = pars_dict
        # self.initial_data = init_vals_ode
        #
        # # set initial data and method to integrate ODE
        # method = method_opts["method"]
        # del method_opts["method"]
        # method_opts["first_step"] = self.Rstart
        # self.odeinstance = ode(self.rhs)
        # self.odeinstance.set_integrator(method, **method_opts)
        # self.odeinstance.set_f_params(self.pars_ode_rhs,
        #                               self.eq_gammaAdi, self.eq_dmdr, self.eq_dthetadr)
        # self.odeinstance.set_initial_value(self.initial_data, self.Rstart*0.9999)
        #
        # # container for solutions
        # # self.dynamics = np.zeros((self.nR, len(self.initial_data)))
        # # self.dynamics[0, :] = all_init_vals

    # @classmethod
    # def from_obj(cls, shell, **kwargs):
    #     return cls(
    #         shell.E0,
    #         shell.Gamma0,
    #         shell.thetaE,
    #         shell.theta0,
    #         shell.M0,
    #         shell.Rstart,
    #         shell.rho0,
    #         shell.dlnrho1dR_0,
    #         **kwargs
    #     )

    # def update_ode_pars(self, **kwargs):
    #
    #     if len(kwargs.keys()) > 0:
    #         for key in kwargs.keys():
    #             # if key.__contains__("eq_"):
    #             #     self.eqs_ode_rhs[key] = kwargs[key]
    #             # else:
    #             self.pars_ode_rhs[key] = kwargs[key]
    #
    #     self.odeinstance.set_f_params(self.pars_ode_rhs, self.eqs_ode_rhs)
    #
    # def update_ode_par(self, key, val):
    #     self.pars_ode_rhs[key] = val
    #     self.odeinstance.set_f_params(self.pars_ode_rhs, self.eqs_ode_rhs)

    def apply_units(self, idx):
        """ method dependent """

    def get_U_e(self, *args):
        """ method dependent """
        return 0.

    def evolove(self, R, rho, dlnrho1dR):
        # update
        self.pars_ode_rhs["rho"] = rho
        self.pars_ode_rhs["dlnrho1dR"] = dlnrho1dR
        self.odeinstance.set_f_params(self.pars_ode_rhs,
                                      self.eq_gammaAdi, self.eq_dmdr, self.eq_dthetadr)

        i_res = np.copy(self.odeinstance.integrate(R))

        if R > self.Rstart:
            assert np.isfinite(self.odeinstance.t)
            assert self.odeinstance.t > self.Rstart

        idx = int(np.where(self.r_grid == R)[0][0])
        assert idx > 0

        # collect solution into the main array
        for i_v_n, v_n in enumerate(self.init_v_ns):
            self.dynamics[v_n][idx] = i_res[i_v_n]

        self.apply_units(idx)

        self._additional_quantities(idx)


    def _additional_quantities(self, idx):

        self.dynamics["beta"][idx] = get_beta(self.dynamics["Gamma"][idx])
        self.dynamics["rho"][idx] = self.pars_ode_rhs["rho"] # the one used for evolution
        self.dynamics["gammaAdi"][idx] = self.eq_gammaAdi(self.dynamics["Gamma"][idx],
                                                          self.dynamics["beta"][idx])
        self.dynamics["rho2"][idx] = self.eq_rho2(self.dynamics["Gamma"][idx],
                                                  self.dynamics["beta"][idx],
                                                  self.dynamics["rho"][idx],
                                                  self.dynamics["gammaAdi"][idx])

        self.dynamics["R"][idx] = self.odeinstance.t
        self.dynamics["tt"][idx] = self.get_elapsed_time(self.dynamics["R"][:idx],
                                                         self.dynamics["Gamma"][:idx],
                                                         self.dynamics["beta"][:idx],
                                                         self.dynamics["theta"][:idx],
                                                         self.pars_ode_rhs["useSpread"])

        self.dynamics["thickness"][idx] = self.eq_delta(self.dynamics["M2"][idx],
                                                        self.dynamics["rho2"][idx],
                                                        self.dynamics["theta"][idx],
                                                        self.dynamics["Gamma"][idx],
                                                        self.dynamics["R"][idx],
                                                        self.pars_ode_rhs["ncells"])

        self.dynamics["U_e"][idx] = self.get_U_e(idx)



        # self.set_last("beta", get_beta(self.get_last("Gamma")))
        # self.set_last("rho", self.pars_ode_rhs["rho"]) # the one used for evolution
        # self.set_last("gammaAdi", self.kwargs["eq_gammaAdi"](self.get_last("Gamma"), self.get_last("beta")))
        # self.set_last("rho2", self.kwargs["eq_rhoprime"](self.get_last("Gamma"),
        #                                                  self.get_last("beta"),
        #                                                  self.get_last("rho"),
        #                                                  self.get_last("gammaAdi")))
        # #self.get_rhoprime(self.get_last("rho"), self.get_last("Gamma")))
        # self.set_last("R", self.odeinstance.t)
        # self.set_last("tt", self.get_elapsed_time(self.get("R"),
        #                                           self.get("Gamma"),
        #                                           self.get("theta")))
        # # mass, rhoprime, theta, Gamma, R
        # self.set_last("thickness", EqOpts.shock_thickness(self.get_last("M2"),
        #                                                   self.get_last("rho2"),
        #                                                   self.get_last("theta"),
        #                                                   self.get_last("Gamma"),
        #                                                   self.get_last("R"),
        #                                                   self.kwargs["ncells"]))
        # self.set_last("U_e", self.get_U_e())

        #
        # self.set("rho", beta(self.get("Gamma")[-1]), -1)
        # self.set("rho", self.pars_ode_rhs["rho"], -1)
        # self.set("R", self.odeinstance.t, -1)
        # self.set("tt", self.get_elapsed_time(self.get("R"), self.get("Gamma"), self.get("theta")), -1)
        # delta = self.get_shock_front_thickness(self.get("R")[-1], self.get("rho")[-1], self.get("Gamma")[-1],
        #                                        self.get("theta")[-1], self.get("M2")[-1])
        # self.set("delta", delta, -1)
        # self.set("U_e", self.get_U_e(), -1)

    # def _set_ode_inits(self, v_ns_dict):
    #     vals = [0. for v_n in self.v_ns_init_vals]
    #     for key in v_ns_dict.keys():
    #         if key in v_ns_dict:
    #             i = self.v_ns_init_vals.index(key)
    #         else:
    #             raise NameError("Key : {} : is not in init.data dict:"
    #                             "{}".format(key, v_ns_dict.keys()))
    #         vals[i] = v_ns_dict[key]
    #     assert len(vals) == len(self.v_ns_init_vals)
    #     return (vals)

    # def i_nv(self, v_n):
    #     return self.all_v_ns.index(v_n)
    #
    # def set(self, v_n, array):
    #     self.dynamics[:, self.i_nv(v_n)] = array
    #
    # def set_last(self, v_n, value):
    #     self.dynamics[-1, self.i_nv(v_n)] = value
    #
    # def set_first(self, v_n, value):
    #     self.dynamics[0, self.i_nv(v_n)] = value
    #
    # def get(self, v_n):
    #     return self.dynamics[:, self.i_nv(v_n)]
    #
    # def get_last(self, v_n):
    #     return self.dynamics[-1, self.i_nv(v_n)]
    #
    # def get_first(self, v_n):
    #     return self.dynamics[0, self.i_nv(v_n)]
    #
    # def get_init_val(self, v_n):
    #     if self.dynamics.ndim == 2:
    #         return self.dynamics[0, self.i_nv(v_n)]
    #     else:
    #         return self.dynamics[self.i_nv(v_n)]

    def init_elapsed_time(self, Rstart, beta0, Gamma0, use_spread):
        if use_spread:
            dThetadr0 = 0
            tt = Rstart / (cgs.c * beta0) * (np.sqrt(1. + Rstart ** 2. * dThetadr0 ** 2.)) - Rstart / cgs.c
        else:
            tt = Rstart / (cgs.c * Gamma0 ** 2. * beta0 * (1. + beta0))
        return tt

    def get_elapsed_time(self, Rs, Gammas, betas, thetas, use_spread):
        # beta = get_beta(Gammas) #np.sqrt(1. - np.power(Gammas, -2))
        if use_spread:
            dThetadr = np.concatenate([np.zeros(1), np.diff(thetas) / np.diff(Rs)])
            integrand = 1. / (cgs.c * betas) * np.sqrt(1. + Rs ** 2. * dThetadr ** 2.) - 1. / (cgs.c)
        else:
            integrand = 1. / (cgs.c * Gammas ** 2. * betas * (1. + betas))
        tti = np.trapz(integrand, Rs) + self.dynamics["tt"][0]
        return tti

    # def get_shock_front_thickness(self, R, rhoprime, Gamma, theta, m):
    #     """
    #     assuming compresion ration '4' by default. as rho2 = 4 * Gamma * rho is embedded here
    #
    #     delta = M2 / (8 * pi * (1-cos(theta)) * Gamma^2 * rho * r^2)
    #     where rho2 = 4 * rho * Gamma
    #     which obays that M2 / (4 * pi) = rho2 * Gamma * r^2 * delta
    #
    #     """
    #
    #     one_min_costheta = 2.#1. - np.cos(theta)/self.kwargs["ncells"]
    #     # delta = m / (8. * np.pi * one_min_costheta * Gamma ** 2. * rho * R ** 2.)
    #     delta = m / (2 * np.pi * one_min_costheta * rhoprime * Gamma * R ** 2)
    #     return delta

    # def get_rhoprime(self, rho, Gamma):
    #     """ 4. is the compression ratio """
    #     return 4. * rho * Gamma
    #     return






# class Driver_Peer_FS(Driver):
#
#     def __init__(
#             self,
#             E0=1.e53,
#             Gamma0=1000.,
#             thetaE=0.,
#             theta0=np.pi / 2.,
#             M0=1.e53 * cgs.c ** -2 / 1000.,
#             Rstart=1.e18,
#             rho0=1.e-2*cgs.mppme,
#             dlnrho1dR_0=0.,
#             **kwargs
#     ):
#
#         assert (not M0 is None)
#         assert np.isfinite(kwargs["aa"])
#         beta0 = get_beta(Gamma0)
#
#         # some equations for ODEs
#         self.eqs_ode_rhs = {"eq_dthetadr": kwargs["eq_dthetadr"],
#                             "eq_gammaAdi": kwargs["eq_gammaAdi"]}
#
#         # initial data for ODEs
#         Rd = get_Rdec2(E0, rho0/cgs.mppme, Gamma0)
#         #(3. / (4. * cgs.pi) * 1. / (cgs.c ** 2. * cgs.mp) * E0 / (rho0 / cgs.mppme * Gamma0 ** 2.)) ** (1. / 3.)
#         m20 = (2. / 3.) * cgs.pi * (1. - np.cos(theta0)) * rho0 * Rstart ** 3. / kwargs["ncells"]
#
#         self.v_ns_init_vals = ["tburst", "tcomoving", "Gamma", "theta", "M2"]
#         init_vals_dic = {
#             "tburst": Rstart / (beta0 * cgs.c),
#             "tcomoving": Rstart / (beta0 * Gamma0 * cgs.c),
#             "Gamma": Gamma0,  # dg/dr
#             "theta": theta0,  # dthet/dr
#             "M2": m20 / M0  # dm/dr
#         }
#         self.initial_data = self._set_ode_inits(init_vals_dic)
#
#         # set additional parameters for ODE
#         self.pars_ode_rhs = {
#             "M0": M0,
#             "aa": kwargs["aa"],
#             # "to_dens": (self.dens),
#             "Rd": Rd,
#             "ncells": kwargs["ncells"],
#             "thetaE": thetaE,
#             "rho": rho0,
#             "dlnrho1dR": dlnrho1dR_0,
#             "useSpread": kwargs["useSpread"],
#             "thetaMax": kwargs["thetaMax"]
#         }
#
#         # initialize time elapsed in the comoving frame
#         tt0 = self.init_elapsed_time(Rstart, beta0, Gamma0, **kwargs)
#         gammaAdi0 = kwargs["eq_gammaAdi"](Gamma0, beta0)
#         rho20 = kwargs["eq_rhoprime"](Gamma0, beta0, rho0, gammaAdi0)  # self.get_rhoprime(rho0, Gamma0)
#         self.all_v_ns = self.v_ns_init_vals + ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2", "gammaAdi"]
#         self.dynamics = np.zeros((1, len(self.all_v_ns)))
#         self.dynamics[0, :len(self.initial_data)] = self.initial_data
#         self.dynamics[0, len(self.initial_data):] = np.array([rho0, tt0, Rstart, 0., 0., beta0, rho20, gammaAdi0])
#         self.dynamics[0, self.i_nv("U_e")] = self.get_U_e(idx=0)
#         self.dynamics[0] = self.apply_units(self.dynamics[0])
#         # self.dynamics = np.hstack((self.apply_units(np.copy(self.initial_data)),
#         #                            np.array([rho0, tt0, Rstart, 0., 0., beta0])))
#
#         # set the RHS
#         self.rhs = Peer_rhs()
#
#         self.kwargs = kwargs
#
#         super(Driver_Peer_FS, self).__init__(Rstart, kwargs["ode_rtol"], kwargs["ode_nsteps"])
#
#     def get_U_e(self,idx=-1):
#
#         Gamma = self.get("Gamma")[idx]
#         rho = self.get("rho")[idx]
#
#         nn = rho / cgs.mppme
#         beta = np.sqrt(1. - np.power(float(Gamma), -2))
#         TT = Peer_rhs.normT(Gamma, beta)
#         ada = Peer_rhs.adabatic_index(TT)
#         eT = (ada * Gamma + 1.) / (ada - 1) * (Gamma - 1.) * nn * cgs.mp * cgs.c ** 2.
#
#         U_e = eT  # assumption
#
#         return U_e
#
#     def apply_units(self, i_res):
#         i_res[self.i_nv("M2")] *= self.pars_ode_rhs["M0"]
#         return i_res
#
#

#
# class Nava_fs_rhs:
#
#     def __init__(self):
#         pass
#
#     def __call__(self, R, params, pars, Eqs):
#         """
#         [0] 1 / beta / cgs.c,
#         [1] 1 / beta / Gamma / cgs.c,
#         [2] dGammadR,
#         [3] dEint2dR,
#         [4] dthetadR,
#         [5] dErad2dR,
#         [6] dEsh2dR,
#         [7] dEad2dR,
#         [8] dM2dR
#         :param R:
#         :param params:
#         :param pars:
#         :return:
#         """
#
#         tburst = params[0]
#         tcomoving = params[1]
#         Gamma = params[2]
#         Eint2 = params[3]
#         theta = params[4]
#         M2 = params[8]
#         #
#         M0 = pars["M0"]
#         Gamma0 = pars["Gamma0"]
#         theta0 = pars["theta0"]
#         rho = pars["rho"] / M0
#         dlnrho1dR = pars["dlnrho1dR"]
#         #
#         # if Eint2 < 0:
#         #     beta = np.sqrt(1. - Gamma ** -2)
#         #     print("ERROR! Eint2 < 0 Gamma < 1 [Gamma:{} beta:{} Eint2:{} M2:{} ] Resetting to 0".format(
#         #         Gamma, beta, Eint2, M2,
#         #     ))
#         #     Eint2 = 0
#         # if Gamma < 1:
#         #     Gamma = 1.0001
#         #     print("ERROR Gamma < 1 [Gamma:{} Eint2:{} M2:{} ] Resetting to 1.0001".format(
#         #         Gamma, Eint2,  M2,
#         #     ))
#         #     # raise ValueError("Gamma < 1")
#         # if Gamma > Gamma0:
#         #     print("ERROR! Gamma({}) > Gamma0({}) -- resetting to Gamma0"
#         #           .format(Gamma, Gamma0))
#         #     Gamma = Gamma0
#
#
#         #  one_over_beta = 1. / np.power(1 - np.power(Gamma, -2.), 0.5)
#         beta = np.sqrt(1. - np.power(Gamma,-2))
#         # beta0 = np.sqrt(1. - Gamma0 ** -2)
#
#         gammaAdi = Eqs["eq_gammaAdi"](Gamma, beta)
#         # gammaAdi = self.gammaAdi(Gamma, beta)  # (4 + 1 / Gamma) / 3.
#
#         # one_minus_costheta = EquationsNava.one_minus_costheta(theta)
#         # one_minus_costheta0 = 1. - np.cos(theta0)
#
#         # Spreading
#         # dthetadR = self.dthetadr(gammaAdi, Gamma, R, theta, pars["aa"]) * int(pars["useSpread"])
#         dthetadR = Eqs["eq_dthetadr"](gammaAdi, Gamma, beta, R, theta, pars["aa"],
#                                       pars["useSpread"], pars["Rd"], pars["thetaMax"])
#         # Densities and volumes
#
#         # rho2 = 4. * Gamma * rho
#         # V2 = M2 / rho2
#         dM2dR = EqOpts.dmdr(Gamma, R, pars["thetaE"], theta, rho, aa=pars["aa"]) / pars["ncells"] # 2 * cgs.pi * R ** 2. * rho * one_minus_costheta / (pars["m_scale"])
#
#         # # # # dGammadR
#         dGammadR = self.dGammadR_fs(Gamma, gammaAdi, dlnrho1dR, M2, dM2dR, Eint2)
#
#         if dGammadR > 0:
#             if Gamma > 0.95 * Gamma0:
#                 # raise ValueError("Gamma > 0.95 Gamma0 after RSISING")
#                 dGammadR = 0.
#
#         # # # # Energies # # # #
#         dEsh2dR = (Gamma - 1.) * dM2dR  # Shocked energy
#
#         # --- Expansion energy
#         dlnV2dR = dM2dR / M2 - dlnrho1dR - dGammadR / Gamma
#         if pars["adiabLoss"]:
#             dEad2dR = -(gammaAdi - 1.) * Eint2 * dlnV2dR
#         else:
#             dEad2dR = 0.
#
#         # -- Radiative losses
#         dErad2dR = pars["epsilon_e_rad"] * dEsh2dR
#
#         ### -- Energy equation
#         dEint2dR = dEsh2dR + dEad2dR - dErad2dR  # / (pars.M0 * c ** 2)
#
#
#         return np.array([1 / beta / cgs.c,
#                          1 / beta / Gamma / cgs.c,
#                          dGammadR,
#                          dEint2dR,
#                          dthetadR,
#                          dErad2dR,
#                          dEsh2dR,
#                          dEad2dR,
#                          dM2dR
#                          ])
#
#     # @staticmethod
#     # def gammaAdi(Gamma, beta):
#     #     """ From Peer 2012 arxiv:1203.5797 """
#     #     mom = Gamma * beta
#     #     theta = mom / 3. * (mom + 1.07 * mom ** 2.) / (1 + mom + 1.07 * mom ** 2.)
#     #     zz = theta / (0.24 + theta)
#     #     gamma_adi = (5. - 1.21937 * zz + 0.18203 * zz ** 2. - 0.96583 * zz ** 3. + 2.32513 * zz ** 4. -
#     #             2.39332 * zz ** 5. + 1.07136 * zz ** 6.) / 3.
#     #     return gamma_adi
#
#     # @staticmethod
#     # def gammaAdi(Gamma):
#     #     return (4 + 1 / Gamma) / 3.
#
#     # @staticmethod
#     # def dthetadr(gammaAdi, Gamma, R, theta, aa=np.nan):
#     #     """
#     #     Source:
#     #     1. / (R * Gamma ** (1. + aa) * theta ** (aa))
#     #     is from https://academic.oup.com/mnras/article/421/1/570/990386
#     #     vperp / R / Gamma * one_over_beta / cgs.c
#     #     is from ???
#     #     :param gammaAdi:
#     #     :param Gamma:
#     #     :param R:
#     #     :param one_over_beta:
#     #     :param theta:
#     #     :param aa:
#     #     :return:
#     #     """
#     #
#     #     if theta < np.pi:
#     #         if np.isfinite(aa):
#     #             return 1. / (R * Gamma ** (1. + aa) * theta ** (aa))
#     #         else:
#     #             vperp = np.sqrt(gammaAdi * (gammaAdi - 1) * (Gamma - 1) / (1 + gammaAdi * (Gamma - 1))) * cgs.c
#     #             one_over_beta = 1. / np.power(1 - np.power(Gamma, -2.), 0.5)
#     #             return vperp / R / Gamma * one_over_beta / cgs.c
#     #     else:
#     #         return 0.
#
#     @staticmethod
#     def GammaEff(Gamma, gammaAdi):
#         return (gammaAdi * Gamma ** 2. - gammaAdi + 1.) / Gamma
#
#     @staticmethod
#     def dGammaEffdGamma(Gamma, gammaAdi):
#         return (gammaAdi * Gamma ** 2. + gammaAdi - 1.) / Gamma ** 2
#         #return 4. / 3. + 1. / Gamma ** 2. / 3. + 2. / Gamma ** 3. / 3.
#
#     @staticmethod
#     def dGammadR_fs(Gamma, gammaAdi, dlnrho1dR, M2, dM2dR, Eint2):
#
#         GammaEff = Nava_fs_rhs.GammaEff(Gamma, gammaAdi)#, gammaAdi) #(gammaAdi * Gamma ** 2. - gammaAdi + 1.) / Gamma
#         dGammaEffdGamma = Nava_fs_rhs.dGammaEffdGamma(Gamma, gammaAdi) #4. / 3. + 1. / Gamma ** 2. / 3. + 2. / Gamma ** 3. / 3.
#
#         f_2 = GammaEff * (gammaAdi - 1.) * Eint2 / Gamma
#         h_2 = GammaEff * (gammaAdi - 1.) * Eint2 * (dM2dR / M2 - dlnrho1dR)
#
#         dGammadR = -((Gamma - 1.) * (GammaEff + 1.) * dM2dR - h_2) / ((1. + M2) + Eint2 * dGammaEffdGamma + f_2)
#
#         return dGammadR
#
# class Driver_Nava_FS(Driver):
#
#     def __init__(
#             self,
#             E0=1.e53,
#             Gamma0=1000.,
#             thetaE=0.,
#             theta0=np.pi / 2.,
#             M0=1.e53 * cgs.c ** -2 / 1000.,
#             Rstart=1.e18,
#             rho0=1.e-2 * cgs.mppme,
#             dlnrho1dR_0=0.,
#             **kwargs
#     ):
#         assert (not M0 is None)
#         assert np.isfinite(kwargs["aa"])
#         beta0 = get_beta(Gamma0)
#
#         # some equations for ODEs
#         self.eqs_ode_rhs = {"eq_dthetadr": kwargs["eq_dthetadr"],
#                             "eq_gammaAdi": kwargs["eq_gammaAdi"]}
#
#         Rd = get_Rdec2(E0, rho0 / cgs.mppme, Gamma0)
#         M20 = (2 / 3.) * np.pi * Rstart ** 3. * (1 - np.cos(theta0)) * rho0 / kwargs["ncells"]
#
#         self.v_ns_init_vals = [
#             "tburst", "tcomoving", "Gamma", "Eint2", "theta", "Erad2", "Esh2", "Ead2", "M2"
#         ]
#         init_data = {
#             "tburst": Rstart / (beta0 * cgs.c),  # 0
#             "tcomoving": Rstart / (beta0 * Gamma0 * cgs.c),  # 1
#             "Gamma": Gamma0,  # 2
#             "Eint2": (Gamma0 - 1) * M20 / M0,  # 3 # 0.75 *
#             "theta": theta0,  # 4
#             "Erad2": 0.,  # 5
#             "Esh2": 0.,  # 6
#             "Ead2": 0.,  # 7
#             "M2": M20 / M0,  # 8
#         }
#         self.initial_data = self._set_ode_inits(init_data)
#
#         self.pars_ode_rhs = {
#             "M0": M0, "theta0": theta0, "Gamma0": Gamma0, "rho": rho0, "dlnrho1dR": dlnrho1dR_0,
#             "epsilon_e_rad": kwargs["epsilon_e_rad"], "useSpread": kwargs["useSpread"],
#             "adiabLoss": kwargs["adiabLoss"],
#             "ncells": kwargs["ncells"], "aa": kwargs["aa"],
#             "Rd": Rd, "thetaMax":kwargs["thetaMax"], "thetaE":0.
#         }
#
#         # initialize time elapsed in the comoving frame
#         tt0 = self.init_elapsed_time(Rstart, beta0, Gamma0, **kwargs)
#         gammaAdi0 = kwargs["eq_gammaAdi"](Gamma0, beta0)
#         rho20 = kwargs["eq_rhoprime"](Gamma0, beta0, rho0, gammaAdi0)#self.get_rhoprime(rho0, Gamma0)
#         self.all_v_ns = self.v_ns_init_vals + ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2", "gammaAdi"]
#         self.dynamics = np.zeros((1, len(self.all_v_ns)))
#         self.dynamics[0, :len(self.initial_data)] = self.initial_data
#         self.dynamics[0, len(self.initial_data):] = np.array([rho0, tt0, Rstart, 0., 0., beta0, rho20, gammaAdi0])
#         self.dynamics[0, self.i_nv("U_e")] = self.get_U_e(idx=0) * cgs.c**2
#         self.dynamics[0] = self.apply_units(self.dynamics[0])
#         # self.dynamics = np.hstack((self.apply_units(np.copy(self.initial_data)),
#         #                            np.array([rho0, tt0, Rstart, 0., 0., beta0])))
#
#         self.rhs = Nava_fs_rhs()
#
#         super(Driver_Nava_FS, self).__init__(Rstart, kwargs["ode_rtol"], kwargs["ode_nsteps"])
#
#         self.kwargs = kwargs
#
#     def get_U_e(self, idx=-1):
#
#         rho = self.get("rho")[idx]
#         Gamma = self.get("Gamma")[idx]
#         M2 = self.get("M2")[idx]
#         Eint2 = self.get("Eint2")[idx]
#
#
#
#         rhoprim = 4. * rho * Gamma  # comoving density
#         V2 = M2 / rhoprim  # comoving volume
#         U_e = Eint2 / V2  # comoving energy density (electrons)
#         # U_b = eps_b * U_e  # comoving energy density (MF)
#         return U_e
#
#     def apply_units(self, i_res):
#         # self.units_dic = {
#         #     "M2": M0,
#         #     "Eint2": M0 * cgs.c**2,
#         #     "Erad2": M0 * cgs.c**2,
#         #     "Esh2": M0 * cgs.c**2,
#         #     "Ead2": M0 * cgs.c**2,
#         # }
#         i_res[self.i_nv("M2")] *= self.pars_ode_rhs["M0"]
#         i_res[self.i_nv("Eint2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         i_res[self.i_nv("Erad2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         i_res[self.i_nv("Esh2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         i_res[self.i_nv("Ead2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         return i_res
#
#
#
# class Nava_fs_rs_rhs:
#
#     def __init__(self):
#         pass
#
#     def __call__(self, R, params, pars, Eqs):
#         """
#         [0]   1 / beta / cgs.c,
#         [1]   1 / beta / Gamma / cgs.c,
#         [2]   dGammadR,
#         [3]   dEint2dR,
#         [4]   dEint3dR,
#         [5]   dthetadR,
#         [6]   dErad2dR,
#         [7]   dErad3dR,
#         [8]   dEsh2dR,
#         [9]   dEsh3dR,
#         [10]  dEad2dR,
#         [11]  dEad3dR,
#         [12]  dM2dR,
#         [13]  dM3dR,
#         [14]  ddeltaR4dR
#         :param R:
#         :param params:
#         :param pars:
#         :return:
#         """
#
#         tburst = params[0]
#         tcomoving = params[1]
#         Gamma = params[2]
#         Eint2 = params[3]
#         Eint3 = params[4]
#         theta = params[5]
#         M2 = params[12]
#         M3 = params[13]
#         deltaR4 = params[14]
#         #
#         M0 = pars["M0"]
#         Gamma0 = pars["Gamma0"]
#         theta0 = pars["theta0"]
#         rho = pars["rho"] / M0
#         dlnrho1dR = pars["dlnrho1dR"]
#         #
#         if Eint2 < 0:
#             beta = np.sqrt(1. - Gamma ** -2)
#             print("ERROR! Eint2 < 0 Gamma < 1 [Gamma:{} beta:{} Eint2:{} Eint3:{} M2:{} M3:{}] Resetting to 0".format(
#                 Gamma, beta, Eint2, Eint3, M2, M3
#             ))
#             Eint2 = 0
#         if Eint3 < 0:
#             # logger.error("Warning: Eint3 = 0")
#             Eint3 = 0.
#         if Gamma < 1:
#             Gamma = 1.0001
#             print("ERROR Gamma < 1 [Gamma:{} Eint2:{} Eint3:{} M2:{} M3:{}] Resetting to 1.0001".format(
#                 Gamma, Eint2, Eint3, M2, M3
#             ))
#             # raise ValueError("Gamma < 1")
#         if Gamma > Gamma0:
#             print("ERROR! Gamma({}) > Gamma0({}) -- resetting to Gamma0"
#                   .format(Gamma, Gamma0))
#             Gamma = Gamma0
#
#         beta = np.sqrt(1. - np.power(Gamma, -2))
#         beta0 = np.sqrt(1. - np.power(Gamma0, -2))
#
#         # forward shock EOS
#         # gammaAdi = Nava_fs_rhs.gammaAdi(Gamma, beta)  # (4 + 1 / Gamma) / 3.
#         gammaAdi = Eqs["eq_gammaAdi"](Gamma, beta)
#
#         # reverseShock EOS [ADIABASTIC]
#         gamma43_minus_one = self.gamma43_minus_one(Gamma, Gamma0, beta, beta0, beta_switch=0.9999)
#         # gammaAdi3 = Nava_fs_rhs.gammaAdi(gamma43_minus_one + 1.,
#         #                                  np.sqrt(1. - np.power(float(gamma43_minus_one + 1.), -2)))
#         gammaAdi3 = Eqs["eq_gammaAdi"](gamma43_minus_one+1., get_beta(gamma43_minus_one + 1.))
#
#         # Spreading
#         dthetadR = Eqs["eq_dthetadr"](gammaAdi, Gamma, beta, R, theta,
#                                       pars["aa"], pars["useSpread"], pars["Rd"], pars["thetaMax"])
#         # dthetadR = Nava_fs_rhs.dthetadr(gammaAdi, Gamma, R, theta, pars["aa"]) * int(pars["useSpread"])
#
#         # Densities and volumes
#         # rho2 = 4. * Gamma * rho
#         # V2 = M2 / rho2
#         dM2dR = EqOpts.dmdr(Gamma, R, pars["thetaE"], theta, rho, aa=pars["aa"]) / pars["ncells"]
#         # B = EquationsNava.B_func(Eint2, V2 / M0, pars["eB"])  # * M0 ** 2
#
#         # Reverse Shock [DENSITY]
#         if (not pars["shutOff"]) and (Gamma < Gamma0):  # and (not M3 > 1.):# and (deltaR4 > 0): # the last assures that jet is decelerating
#             alpha_of = pars["tprompt"] * beta0 * cgs.c
#             ddeltaR4dR = (1 / beta ** 4 - 1 / beta0 ** 4) / \
#                          (1 / beta + 1 / beta0) / \
#                          (1 / beta ** 2 + 1 / beta0 ** 2) * beta0
#             rho4_scale_factor = np.float64(-deltaR4 / alpha_of)
#             # rho4_factor = 1 / alpha_of * 1 / (2 * np.pi * R ** 2 * (1 - np.cos(theta0)))  # * pars["m_scale"] -- NO EFFECT
#             # rho4 = rho4_factor * np.exp(rho4_scale_factor)  # Here exp() does not really make a difference
#             dM3dR = (1 / alpha_of) * ddeltaR4dR * np.exp(rho4_scale_factor)  # [not / pars["m_scale"] -- changes everything]
#             # rho3prim = 4 * Gamma * rho4
#             # if not pars["shutOff"] and (M3 > 0) and (Eint3 > 0):
#                 # V3 = M3 / rho3prim
#                 # BRS = EquationsNava.B_func(Eint3 * M0 ** 2, V3, pars["eB3"])  # np.sqrt(8 * pi * pars.eB3) * np.sqrt(Eint3) / np.sqrt(V3) * c * np.sqrt(pars.M0)
#             # else:
#                 # BRS = 0.
#             dlnrho4dR = -2 / R - ddeltaR4dR / alpha_of
#             # print('\t', np.exp(rho4_scale_factor))
#             # print("Gamma:{:.1f} ddeltaR4dR:{:.3e} deltaR4:{:.3e} alpha_of:{:.3e} dM3dR:{:.3e} np.exp(rho4_scale_factor):{:.3e}".format(Gamma, ddeltaR4dR, deltaR4, alpha_of, dM3dR, np.exp(rho4_scale_factor)))
#             # if np.exp(rho4_scale_factor) == 0.: exit(1)
#
#         # elif False and sets["reverseShock"] and (Gamma < Gamma0) and (not M3 > 1.):
#         #     alpha_of = pars["tprompt"] * beta0 * cgs.c
#         #     ddeltaR4dR = (1 / beta ** 4 - 1 / beta0 ** 4) / \
#         #                  (1 / beta + 1 / beta0) / \
#         #                  (1 / beta ** 2 + 1 / beta0 ** 2) * beta0
#         #
#         #     bb0m1 = 0.5 / Gamma ** 2 + 0.5 / Gamma0 ** 2 + Gamma ** (-4) / 8 + Gamma0 ** (-4) / 8 - 0.25 / (Gamma0 ** 2) / (Gamma ** 2)
#         #     dM3dR = 1 * (beta0 ** 2 - beta ** 2) / (beta0 + beta) / beta0 / pars["tprompt"] / beta / cgs.c / bb0m1
#         #     rho4 = 1 / (2 * cgs.pi * R ** 2 *  pars["tprompt"] * (1-np.cos(theta)) * beta * cgs.c)
#         #     rho3prim = 4 * Gamma * rho4
#         #     if not sets["shutOff"] and (M3 > 0) and (Eint3 > 0):
#         #         V3 = M3 / rho3prim
#         #         BRS = EquationsNava.B_func(Eint3 * M0 ** 2, V3, pars["eB3"])  # np.sqrt(8 * pi * pars.eB3) * np.sqrt(Eint3) / np.sqrt(V3) * c * np.sqrt(pars.M0)
#         #     else:
#         #         BRS = 0.
#         #     dlnrho4dR = -ddeltaR4dR / alpha_of - 2 / R
#         else:
#             ddeltaR4dR = 0.
#             dM3dR = 0.
#             rho4 = 0.
#             BRS = 0.
#             dlnrho4dR = 0.
#
#         # dGammadR
#         dGammadR = self.dGammadR_fs_rs(Gamma, Gamma0, gammaAdi, dlnrho1dR, M2, dM2dR,
#                                        dlnrho4dR, M3, dM3dR, Eint2, Eint3, gammaAdi3, gamma43_minus_one)
#
#         if dGammadR > 0:
#             if Gamma > 0.95 * Gamma0:
#                 # raise ValueError("Gamma > 0.95 Gamma0 after RSISING")
#                 dGammadR = 0.
#
#         # Energies
#         dEsh2dR = (Gamma - 1.) * dM2dR  # Shocked energy
#
#         # Expansion energy
#         dlnV2dR = dM2dR / M2 - dlnrho1dR - dGammadR / Gamma
#         if pars["adiabLoss"]:
#             dEad2dR = -(gammaAdi - 1.) * Eint2 * dlnV2dR
#         else:
#             dEad2dR = 0.
#
#         # Radiative losses
#         dErad2dR = pars["epsilon_e_rad"] * dEsh2dR
#         dErad3dR = pars["epsilon_e_rad_RS"] * gamma43_minus_one * dM3dR
#
#         assert np.isfinite(dErad2dR)
#
#         if (not pars["shutOff"]) and M3 > 0:  ### rho4 becomes 0 when RS injecta is cut-off
#             ### Shocked energy
#             dEsh3dR = gamma43_minus_one * dM3dR
#
#             ### Expansion energy
#             # dlnGamma43dR = dGammadR * dgamma43dGamma / (gamma43_minus_one+1)
#             dlnV3dR = dM3dR / M3 - dlnrho4dR - dGammadR / Gamma
#
#             if pars["adiabLoss_RS"]:
#                 dEad3dR = -(gammaAdi3 - 1) * Eint3 * dlnV3dR  # dgamma43dGamma - h_3 / GammaEff3
#             else:
#                 dEad3dR = 0.
#         else:
#             dEsh3dR = 0.
#             dEad3dR = 0.
#
#         dEint2dR = dEsh2dR + dEad2dR - dErad2dR  # / (pars.M0 * c ** 2)
#         dEint3dR = dEsh3dR + dEad3dR - dErad3dR  # / (pars.M0 * c ** 2)
#
#         return np.array([
#             1 / beta / cgs.c,
#             1 / beta / Gamma / cgs.c,
#             dGammadR,
#             dEint2dR,
#             dEint3dR,
#             dthetadR,
#             dErad2dR,
#             dErad3dR,
#             dEsh2dR,
#             dEsh3dR,
#             dEad2dR,
#             dEad3dR,
#             dM2dR,
#             dM3dR,
#             ddeltaR4dR
#         ])
#
#
#     @staticmethod
#     def gamma43_minus_one(Gamma, Gamma0, beta, beta0, beta_switch=0.9999):
#         # if Gamma > 0.999995 * Gamma0:
#         #     return 0
#         # else:
#         #     return Gamma * Gamma0 - Gamma * Gamma0 * beta * beta0
#         # return np.sqrt(Gamma**2 * Gamma0**2 - Gamma**2 - Gamma0**2 + 1)
#
#         # print("1:{} 2:{}".format(Gamma * Gamma0 - Gamma * Gamma0 * beta * beta0,
#         #                          Gamma*Gamma0 - np.sqrt(Gamma**2 * Gamma0**2 - Gamma0**2 - Gamma**2 + 1)))
#         # return Gamma*Gamma0 - np.sqrt(Gamma**2 * Gamma0**2 - Gamma0**2 - Gamma**2 + 1)
#
#         # gamma43_minus_one = Gamma * Gamma0 * (1 - beta * beta0)
#         # if gamma43_minus_one == 1.:
#         #     gamma43_minus_one = Gamma * Gamma0 * \
#         #                         (1 / Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / Gamma0 ** 2) / \
#         #                         (1 + beta * beta0) - 1
#         #
#         # return gamma43_minus_one
#
#         if Gamma > 0.999995 * Gamma0:
#             gamma43_minus_one = 1. - 1.
#         else:
#             gamma43_minus_one = Gamma * Gamma0 * (1 / Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / Gamma0 ** 2) / (1 + beta * beta0) - 1.
#             # gamma43_minus_one = Gamma*Gamma0 - np.sqrt(Gamma0**2 - 1) * np.sqrt(Gamma**2 - 1) # -- wolfram. Creats jump
#
#             # if beta < 0.999:
#                 # gamma43_minus_one = Gamma * Gamma0 * (1 - beta * beta0)
#                 # gamma43_minus_one = Gamma * Gamma0 * (1 / Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / Gamma0 ** 2) / (1 + beta * beta0) - 1.
#             #
#             # print(gamma43_minus_one)
#             # else:
#                 # gamma43_minus_one = Gamma * Gamma0 * (1 / Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / Gamma0 ** 2) / (1 + beta * beta0) - 1.
#             # print(gamma43_minus_one, Gamma * Gamma0 * (1 - beta * beta0)); exit(1)
#         # if Gamma > 0.999995 * pars.Gamma0:
#         #     gamma43_minus_one = 0.
#         # else:
#         #     gamma43_minus_one = Gamma * pars.Gamma0 * (
#         #                 1 / pars.Gamma0 ** 2 + 1 / Gamma ** 2 - 1 / Gamma ** 2 / pars.Gamma0 ** 2) / (
#         #                                     1 + beta * beta0) - 1
#         # # else:
#         # # gamma43_minus_one = Gamma*ModVar.Gamma0*(1-beta*beta0) - 1
#         return gamma43_minus_one
#
#     @staticmethod
#     def dgamma43dGamma(Gamma0, Gamma):
#         """ these two should be equivalent but they are not exactly the same at Gamma=>1 """
#         return Gamma0 - (Gamma0**2 * Gamma - Gamma) / (Gamma**2 * Gamma0**2 - Gamma0**2 - Gamma**2 + 1)**(1./2.) # wolfram (simple) [works]
#         # return Gamma0 - Gamma * (np.sqrt(Gamma0**2-1)/np.sqrt(Gamma**2-1)) # wolfram alpha [does no work]
#         # return 0.5 / Gamma0 + 0.5 / Gamma ** 2 * (0.5 / Gamma0 - Gamma0 - 3. * Gamma0 / 8. / Gamma ** 2) # from the source  with beta assympotoe
#
#     @staticmethod
#     def dGammaEff3dGamma(Gamma, gammaAdi3, dgamma43dGamma, gamma43):
#         """ derived """
#         return gammaAdi3 * (1. + Gamma ** -2) - dgamma43dGamma / 3. / gamma43 ** 2 * (Gamma - 1. / Gamma) - Gamma ** -2
#
#
#     @staticmethod
#     def dGammadR_fs_rs(Gamma, Gamma0, gammaAdi, dlnrho1dR, M2, dM2dR,
#                        dlnrho4dR, M3, dM3dR, Eint2, Eint3, gammaAdi3, gamma43_m1):
#
#         # gamma43_minus_one = NavaEqs.gamma43_minus_one(Gamma, Gamma0, beta, beta0)
#         # gammaAdi3 = (4 + 1 / (gamma43_minus_one + 1)) / 3.
#         dgamma43dGamma = Nava_fs_rs_rhs.dgamma43dGamma(Gamma0, Gamma) #0.5 / Gamma0 + 0.5 / Gamma ** 2 * (0.5 / Gamma0 - Gamma0 - 3. * Gamma0 / 8. / Gamma ** 2)
#         # . Using asymptotic approximation of beta
#         GammaEff3 = Nava_fs_rhs.GammaEff(Gamma, gammaAdi3)# (gammaAdi3 * Gamma ** 2 - gammaAdi3 + 1) / Gamma # (gammaAdi * Gamma ** 2. - gammaAdi + 1.) / Gamma
#         dGammaEff3dGamma = Nava_fs_rs_rhs.dGammaEff3dGamma(Gamma, gammaAdi3, dgamma43dGamma, gamma43_m1+1.)#gammaAdi3 * (1. + Gamma ** -2) - dgamma43dGamma / 3. / (gamma43_m1 + 1.) ** 2 * (Gamma - 1. / Gamma) - Gamma ** -2
#
#         dGammaEffdGamma = Nava_fs_rhs.dGammaEffdGamma(Gamma, gammaAdi) #4. / 3. + 1. / Gamma ** 2. / 3. + 2. / Gamma ** 3. / 3.
#         GammaEff = Nava_fs_rhs.GammaEff(Gamma, gammaAdi) #(gammaAdi * Gamma ** 2. - gammaAdi + 1.) / Gamma
#
#         f_2 = GammaEff * (gammaAdi - 1.) * Eint2 / Gamma
#         h_2 = GammaEff * (gammaAdi - 1.) * Eint2 * (dM2dR / M2 - dlnrho1dR)
#
#         fh_factor3 = GammaEff3 * (gammaAdi3 - 1.) * Eint3
#         f_3 = fh_factor3 / (gamma43_m1 + 1.) * dgamma43dGamma
#         if Eint3 != 0: h_3 = fh_factor3 * (dM3dR / M3 - dlnrho4dR)
#         else: h_3 = 0.
#         dGammadR = -((Gamma - 1.) * (GammaEff + 1.) * dM2dR + (
#                 Gamma - Gamma0 + GammaEff3 * gamma43_m1) * dM3dR - h_2 - h_3) / (
#                            (M2 + M3) + Eint2 * dGammaEffdGamma + Eint3 * dGammaEff3dGamma + f_2 + f_3)
#
#         return dGammadR
#
# class Driver_Nava_FSRS(Driver):
#
#     def __init__(
#             self,
#             E0=1.e53,
#             Gamma0=1000.,
#             thetaE=0.,
#             theta0=np.pi / 2.,
#             M0=1.e53 * cgs.c ** -2 / 1000.,
#             Rstart=1.e18,
#             rho0=1.e-2 * cgs.mppme,
#             dlnrho1dR_0=0.,
#             **kwargs
#     ):
#         assert np.isfinite(kwargs["aa"])
#         self.rs_shutOff_criterion_rho = 1e-50 # when density in RS falls below this, the RS computation stops
#
#         # some equations for ODEs
#         self.eqs_ode_rhs = {"eq_dthetadr": kwargs["eq_dthetadr"],
#                             "eq_gammaAdi": kwargs["eq_gammaAdi"]}
#
#
#         assert (not M0 is None)
#         beta0 = get_beta(Gamma0)
#
#         Rd = get_Rdec2(E0, rho0/cgs.mppme, Gamma0)
#         M20 = (2 / 3.) * np.pi * Rstart ** 3. * (1 - np.cos(theta0)) * rho0 / kwargs["ncells"]
#
#         self.v_ns_init_vals = [
#             "tburst", "tcomoving", "Gamma",
#             "Eint2", "Eint3", "theta", "Erad2", "Erad3", "Esh2", "Esh3", "Ead2", "Ead3",
#             "M2", "M3", "deltaR4"
#         ]
#
#         init_data = {
#             "tburst": Rstart / (get_beta(Gamma0) * cgs.c),  # 0
#             "tcomoving": Rstart / (get_beta(Gamma0) * cgs.c) / Gamma0,  # 1
#             "Gamma": Gamma0,  # 2
#             "Eint2": (Gamma0 - 1) * M20 / M0,  # 3
#             "Eint3": 0.,  # 4
#             "theta": theta0,  # 5
#             "Erad2": 0.,  # 6
#             "Erad3": 0.,  # 7
#             "Esh2": 0.,  # 8
#             "Esh3": 0.,  # 9
#             "Ead2": 0.,  # 10
#             "Ead3": 0.,  # 11
#             "M2": M20 / M0,  # 12
#             "M3": 0.,  # 13
#             "deltaR4": 0.  # 14
#         }
#         self.initial_data = self._set_ode_inits(init_data)
#
#         self.pars_ode_rhs = {
#             "M0": M0, "theta0": theta0, "Gamma0": Gamma0, "rho": rho0, "dlnrho1dR": dlnrho1dR_0,
#             "epsilon_e_rad": kwargs["epsilon_e_rad"], "useSpread": kwargs["useSpread"],
#             "adiabLoss": kwargs["adiabLoss"],
#             "ncells": kwargs["ncells"], "aa": kwargs["aa"],
#             # reverse shock
#             "tprompt": kwargs["tprompt"],
#             "epsilon_e_rad_RS": kwargs["epsilon_e_rad_RS"],
#             "adiabLoss_RS": kwargs["adiabLoss_RS"],
#             "shutOff": False,
#             "thetaMax":kwargs["thetaMax"],
#             "Rd": Rd,
#             "thetaE": 0.
#         }
#
#         # initialize time elapsed in the comoving frame
#         gammaAdi0 = kwargs["eq_gammaAdi"](Gamma0, beta0)
#         rho20 = kwargs["eq_rhoprime"](Gamma0, beta0, rho0, gammaAdi0)  # self.get_rhoprime(rho0, Gamma0)
#         tt0 = self.init_elapsed_time(Rstart, beta0, Gamma0, **kwargs)
#         self.all_v_ns = self.v_ns_init_vals + \
#                         ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2",
#                          "rho4", "Gamma43", "U_e_RS", "thickness_RS", "rho3",
#                          "gammaAdi", "gammaAdi3"]
#         self.dynamics = np.zeros((1, len(self.all_v_ns)))
#         self.dynamics[0, :len(self.initial_data)] = self.initial_data
#         self.dynamics[0, len(self.initial_data):] = np.array([rho0, tt0, Rstart, 0., 0., beta0, rho20,
#                                                               0., 0., 0., 0., 0.,
#                                                               gammaAdi0, 0.])
#         self.dynamics[0, self.i_nv("U_e")] = self.get_U_e(idx=0) * cgs.c**2
#         self.dynamics[0] = self.apply_units(self.dynamics[0])
#         # self.dynamics = np.hstack((self.apply_units(np.copy(self.initial_data)),
#         #                            np.array([rho0, tt0, Rstart, 0., 0., beta0, 0., 0., 0., 0.])))
#
#         self.rhs = Nava_fs_rs_rhs()
#
#         self.kwargs = kwargs
#
#         super(Driver_Nava_FSRS, self).__init__(Rstart, kwargs["ode_rtol"], kwargs["ode_nsteps"])
#
#     def get_U_e(self,idx=-1):
#         rho = self.get("rho")[idx]
#         Gamma = self.get("Gamma")[idx]
#         M2 = self.get("M2")[idx]
#         Eint2 = self.get("Eint2")[idx]
#
#         rhoprim = 4. * rho * Gamma  # comoving density
#         V2 = M2 / rhoprim  # comoving volume
#         U_e = Eint2 / V2  # comoving energy density (electrons)
#         # U_b = eps_b * U_e  # comoving energy density (MF)
#         return U_e
#
#     def _additional_quantities(self):
#
#         super(Driver_Nava_FSRS, self)._additional_quantities()
#
#         if (not self.pars_ode_rhs["shutOff"]):
#
#             alpha_of = self.kwargs["tprompt"] * self.get("beta")[0] * cgs.c
#             rho4_fac_1 = self.pars_ode_rhs["M0"] / (2 * alpha_of * np.pi * (1.-np.cos(self.get("theta")[0])))
#             rho4_fac = rho4_fac_1 / np.power(self.get("R")[-1], 2)
#             rho4 = rho4_fac * np.exp(-self.get("deltaR4")[-1] / alpha_of)
#
#             self.set_last("rho4", rho4)
#
#             gamma43 = Nava_fs_rs_rhs.gamma43_minus_one(self.get("Gamma")[-1],self.get("Gamma")[0],
#                                                        self.get("beta")[-1], self.get("beta")[0], beta_switch=0.9999)
#
#             self.set_last("Gamma43", np.float64(gamma43+1.))
#             self.set_last("U_e_RS", self.get_U_e_rs())
#
#             gammaAdi3 = self.kwargs["eq_gammaAdi"](gamma43+1., get_beta(gamma43+1.))
#             self.set_last("gammaAdi3", np.float64(gammaAdi3))
#
#             rho3prim = self.kwargs["eq_rhoprime"](self.get_last("Gamma"), self.get_last("beta"),
#                                                   self.get_last("rho4"), self.get_last("gammaAdi3")) ## SURE it is gammaAdi3 ????
#             self.set_last("rho3", rho3prim)
#
#             # rho3prim = self.get_rhoprime(self.get_last("rho4"), self.get_last("Gamma"))
#             # nprim3 = rho3prim / cgs.mp
#             # thickness_RS = self.get("M3")[-1] / (8 * np.pi * (1. - np.cos(self.get("theta")[-1])) *
#             #                                      self.get("Gamma")[-1] ** 2 * rho4 * self.get("R")[-1] ** 2)
#             thickness_RS = EqOpts.shock_thickness(self.get_last("M3"),
#                                                   self.get_last("rho3"),
#                                                   self.get_last("theta"),
#                                                   self.get_last("Gamma"),
#                                                   self.get_last("R"),
#                                                   self.kwargs["ncells"])
#             self.set_last("thickness_RS", thickness_RS)
#
#         # if self.pars_ode_rhs["shutOff"]: self.set("rho3", 0., -1)
#         # if self.pars_ode_rhs["shutOff"]: self.set("Eint3", 0., -1)
#         # if self.pars_ode_rhs["shutOff"]: self.set("Ead3", 0., -1)
#         # if self.pars_ode_rhs["shutOff"]: self.set("Erad3", 0., -1)
#         # if self.pars_ode_rhs["shutOff"]: self.set("Esh3", 0., -1)
#         # if self.pars_ode_rhs["shutOff"]: self.set("M3", 0., -1)
#         # if self.pars_ode_rhs["shutOff"]: self.set("deltaR4", 0., -1)
#
#     def evolove(self, R, rho, dlnrho1dR):
#         super(Driver_Nava_FSRS, self).evolove(R, rho, dlnrho1dR)
#         if self.get("Gamma")[-1] < self.get("Gamma")[0] and self.get("rho4")[-1] < self.rs_shutOff_criterion_rho:
#             self.pars_ode_rhs["shutOff"] = True
#         self.odeinstance.set_f_params(self.pars_ode_rhs)
#
#     def get_U_e_rs(self):
#         rho3 = 4. * self.get("Gamma")[-1] * self.get("rho4")[-1] # IS it correct iwth Gamma and not Gamma43?
#         V3 = self.get("M3")[-1] / rho3      # comoving volume
#         U_e = self.get("Eint3")[-1] / V3    # comoving energy density (electrons)
#         # U_b = eps_b * U_e  # comoving energy density (MF)
#         return U_e
#
#     def apply_units(self, i_res):
#         # self.units_dic = {
#         #     "M2": M0,
#         #     "Eint2": M0 * cgs.c**2,
#         #     "Erad2": M0 * cgs.c**2,
#         #     "Esh2": M0 * cgs.c**2,
#         #     "Ead2": M0 * cgs.c**2,
#         #     "M3": M0,
#         #     "Eint3": M0 * cgs.c ** 2,
#         #     "Erad3": M0 * cgs.c ** 2,
#         #     "Esh3": M0 * cgs.c ** 2,
#         #     "Ead3": M0 * cgs.c ** 2,
#         # }
#         i_res[self.i_nv("M2")] *= self.pars_ode_rhs["M0"]
#         i_res[self.i_nv("Eint2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         i_res[self.i_nv("Erad2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         i_res[self.i_nv("Esh2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         i_res[self.i_nv("Ead2")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#
#         i_res[self.i_nv("M3")] *= self.pars_ode_rhs["M0"]
#         i_res[self.i_nv("Eint3")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         i_res[self.i_nv("Erad3")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         i_res[self.i_nv("Esh3")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#         i_res[self.i_nv("Ead3")] *= self.pars_ode_rhs["M0"] * cgs.c**2
#
#         return i_res


def evolove_driver(
        driver,
        E0=1e53,
        Gamma0=100.,
        thetaE=0.,
        theta0=np.pi / 2,
        M0=None,
        r_grid_pars=(18., 22., 100),
        r_grid=None,
        dens_pars=(1e-3, None, None, None, None),
        **kwargs
):
    if (M0 is None): M0 = E0 * cgs.c ** -2 / Gamma0
    if (r_grid is None): r_grid = np.logspace(r_grid_pars[0], r_grid_pars[1], r_grid_pars[2])
    rho0, dlnrho1dR0 = rho_dlnrho1dR(r_grid[0], *dens_pars)

    dynamics = driver(
        E0=E0,
        Gamma0=Gamma0,
        thetaE=thetaE,
        theta0=theta0,
        M0=M0,
        r_grid=r_grid,
        rho0=rho0,
        dlnrho1dR_0=dlnrho1dR0,
        **kwargs
    )

    for i in range(1, len(r_grid)):
        rho_i, dlnrho1dR_i = rho_dlnrho1dR(r_grid[i], *dens_pars)
        dynamics.evolove(r_grid[i], rho_i, dlnrho1dR_i)
        # print(dynamics.get_last("Gamma"))

    # import matplotlib.pyplot as plt
    # plt.loglog(dynamics.get("R"), dynamics.get("Gamma"))
    # plt.show()
    return dynamics

if __name__ == "__main__":
    pass
    # rho = 1e-2 * cgs.mppme
    # RR = np.logspace(14., 23., 1000)
    # dyn = Driver_Peer_FS(E0=1e53, M0=1e53 / (cgs.c ** 2 * 1000), Rstart=RR[0], rho0=rho, useSpread=False, aa=-1., ncells=1,
    #                      ode_rtol=1e-4, ode_nsteps=1000, eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi/2.)
    # for i in range(1,len(RR)):
    #     # dyn.update_ode_pars(rho=rho)
    #     rho, dlnrho1dR = rho_dlnrho1dR(RR[i], 1e-2, None, None, None, None)
    #     dyn.evolove(RR[i],rho, dlnrho1dR)
    #
    # print(dyn.get("Gamma"))
    # dyn2 = Driver_Nava_FS(E0=1e53, M0=1e53 / (cgs.c ** 2 * 1000), Rstart=RR[0], rho0=rho, useSpread=False, aa=-1., ncells=1,
    #                       adiabLoss=True, epsilon_e_rad=0,
    #                       ode_rtol=1e-4, ode_nsteps=1000,  eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi/2., )
    # for i in range(1,len(RR)):
    #     # dyn2.update_ode_pars(rho=rho)
    #     rho, dlnrho1dR = rho_dlnrho1dR(RR[i], 1e-2, None, None, None, None)
    #     dyn2.evolove(RR[i],rho, dlnrho1dR)
    #
    # dyn3 = Driver_Nava_FSRS(E0=1e53, M0=1e53 / (cgs.c ** 2 * 1000), Rstart=RR[0], rho0=rho, useSpread=False, aa=-1., ncells=1,
    #                         adiabLoss=True, epsilon_e_rad=0, tprompt=1e3,
    #                         adiabLoss_RS=True, epsilon_e_rad_RS=0,
    #                         ode_rtol=1e-8, ode_nsteps=1000,  eq_dthetadr=EqOpts.dthetadr_None, thetaMax=np.pi/2.)
    # for i in range(1,len(RR)):
    #     # dyn3.update_ode_pars(rho=rho)
    #     rho, dlnrho1dR = rho_dlnrho1dR(RR[i], 1e-2, None, None, None, None)
    #     dyn3.evolove(RR[i],rho, dlnrho1dR)
    #
    #
    # import matplotlib.pyplot as plt
    # plt.semilogx(dyn.get("R"), dyn.get("Gamma"), label="P")
    # # plt.loglog(dyn2.get("R"), dyn2.get("Gamma"), label="N1")
    # # plt.loglog(dyn3.get("R"), dyn3.get("Gamma"), label="N2")
    # plt.legend()
    # plt.show()
