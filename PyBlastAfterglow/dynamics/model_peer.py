import numpy as np
import numba as nb
from PyBlastAfterglow.uutils import cgs
from .dynamics import Driver
from .eqs import (get_beta, get_Rdec2, rho_dlnrho1dR)
from .eqs_opts import (opts_dthetadr, opts_dmdr, opts_gammaAdi, opts_rho2, opts_shock_thickness)


def get_normT(gamma, beta):
    mom = gamma * beta
    return mom / 3. * (mom + 1.07 * mom ** 2.) / (1 + mom + 1.07 * mom ** 2.)

def get_adabatic_index(theta):
    zz = theta / (0.24 + theta)
    return (5 - 1.21937 * zz + 0.18203 * zz ** 2. - 0.96583 * zz ** 3. + 2.32513 * zz ** 4. -
            2.39332 * zz ** 5. + 1.07136 * zz ** 6.) / 3.

# @staticmethod
# def gamma_adi(Gamma, beta):
#     """ https://arxiv.org/pdf/1203.5797.pdf """
#     TT = Peer_rhs.normT(Gamma, beta)
#     ada = Peer_rhs.adabatic_index(TT)
#     return ada

# @staticmethod
# def gamma_adi(Gamma, _):
#     """ From Nava paper """
#     return (4 + 1 / Gamma) / 3.

def get_U_e(Gamma, rho):

    nn = rho / cgs.mppme
    beta = np.sqrt(1. - np.power(float(Gamma), -2))
    TT = get_normT(Gamma, beta)
    ada = get_adabatic_index(TT)
    eT = (ada * Gamma + 1.) / (ada - 1) * (Gamma - 1.) * nn * cgs.mp * cgs.c ** 2.

    U_e = eT  # assumption

    return U_e

def get_dgdm(M0, Gamma, beta, mm, gamma_adi):
    """
    https://arxiv.org/pdf/1203.5797.pdf
    :param M0:
    :param Gamma:
    :param mm:
    :return:
    """
    # assert np.isfinite(gam)

    # ada = (4 + 1 / gam) / 3. # TODO REMOVE

    # numerator = -4.*pi*jet.nn*mp*rr**2. * ( ada*(gam**2.-1)-(ada-1)*gam*beta**2  )
    # denominator = jet.EE/(jet.Gam0*cc**2.) + 4./3.*pi*jet.nn*mp*rr**3.*(2.*ada*gam-(ada-1)*(1.+gam**(-2)))
    # [original]
    # numerator = -10. ** mm * np.log(10) * (ada * (gam ** 2. - 1) - (ada - 1) * gam * beta ** 2)
    # denominator = M0 + 10. ** mm * (2. * ada * gam - (ada - 1) * (1. + gam ** (-2)))

    numerator = -(gamma_adi * (Gamma ** 2. - 1) - (gamma_adi - 1) * Gamma * beta ** 2)
    denominator = 1 + mm * (2. * gamma_adi * Gamma - (gamma_adi - 1) * (1. + Gamma ** (-2)))  # first 1 is M0

    # print denominator

    return numerator / denominator

def rhs(R, solution_array, pars_dict,
        eq_gammaAdi, eq_dmdr, eq_dthetadr):

    tburst = solution_array[0]
    tcomov = solution_array[1]
    Gamma = solution_array[2]
    theta = solution_array[3]
    m = solution_array[4]

    M0 = pars_dict["M0"]
    aa = pars_dict["aa"]
    Rd = pars_dict["Rd"]
    thetaE = pars_dict["thetaE"]
    rhoi = pars_dict["rho"] / M0

    # ODEs
    beta = np.sqrt(1. - np.power(Gamma, -2))
    gamma_adi = eq_gammaAdi(Gamma, beta)
    dmdr = eq_dmdr(Gamma, R, thetaE=thetaE, theta=theta, rho=rhoi, aa=aa) / pars_dict["ncells"]
    dthetadr = eq_dthetadr(gamma_adi, Gamma, beta, R, theta, aa, pars_dict["useSpread"], Rd, pars_dict["thetaMax"])
    dgdm = get_dgdm(M0, Gamma, beta, m, gamma_adi)
    dgdr = dgdm * dmdr

    return np.array([
        1 / beta / cgs.c,
        1 / beta / Gamma / cgs.c,
        dgdr,
        dthetadr,
        dmdr
    ])


class Driver_Peer_FS(Driver):

    def __init__(
            self,
            E0=1.e53,
            Gamma0=1000.,
            thetaE=0.,
            theta0=np.pi / 2.,
            M0=1.e53 * cgs.c ** -2 / 1000.,
            r_grid = np.logspace(18., 22., 500),
            rho0=1.e-2*cgs.mppme,
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
        Rd = get_Rdec2(E0, rho0/cgs.mppme, Gamma0)
        #(3. / (4. * cgs.pi) * 1. / (cgs.c ** 2. * cgs.mp) * E0 / (rho0 / cgs.mppme * Gamma0 ** 2.)) ** (1. / 3.)
        m20 = (2. / 3.) * cgs.pi * (1. - np.cos(theta0)) * rho0 * Rstart ** 3. / kwargs["ncells"]

        # filling constant parameters for ODEs
        pars_ode_rhs = {
            "M0": M0,
            "aa": kwargs["aa"],
            "Rd": Rd,
            "ncells": kwargs["ncells"],
            "thetaE": thetaE,
            "rho": rho0,
            "dlnrho1dR": dlnrho1dR_0,
            "useSpread": kwargs["useSpread"],
            "thetaMax": kwargs["thetaMax"]
        }

        init_v_ns = ["tburst", "tcomoving", "Gamma", "theta", "M2"]
        all_v_ns = init_v_ns + ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2", "gammaAdi"]

        # storage for solutions
        dtypes = []
        for v_n in all_v_ns:
            dtypes.append((v_n, 'f8'))
        vals = np.zeros(len(r_grid), dtype=dtypes)

        # filling initial data
        vals['tburst'][0] = Rstart / (beta0 * cgs.c)
        vals['tcomoving'][0] = Rstart / (beta0 * Gamma0 * cgs.c)
        vals['Gamma'][0] = Gamma0
        vals['theta'][0] = theta0
        vals['M2'][0] = m20 / M0

        # filling non-ODE initial values
        vals['rho'][0] = rho0
        vals['tt'][0] = self.init_elapsed_time(Rstart, beta0, Gamma0, kwargs['useSpread']) # tt0
        vals['R'][0] = Rstart
        vals['thickness'][0] = 0.
        vals['beta'][0] = beta0
        vals['gammaAdi'][0] = gammaAdi0
        vals['rho2'][0] = eq_rho2(Gamma0, beta0, rho0, gammaAdi0)
        vals['U_e'][0] = get_U_e(Gamma0, rho0)#self.get_U_e(idx=0)

        # Rescale the values to 'cgs'
        # self.vals[0, :] = self.apply_units(self.vals[0, :])

        super(Driver_Peer_FS, self).__init__(r_grid, rhs, vals, pars_ode_rhs, init_v_ns, all_v_ns, **kwargs)

        #
        #
        # self.v_ns_init_vals = ["tburst", "tcomoving", "Gamma", "theta", "M2"]
        # init_vals_dic = {
        #     "tburst": Rstart / (beta0 * cgs.c),
        #     "tcomoving": Rstart / (beta0 * Gamma0 * cgs.c),
        #     "Gamma": Gamma0,  # dg/dr
        #     "theta": theta0,  # dthet/dr
        #     "M2": m20 / M0  # dm/dr
        # }
        # self.initial_data = self._set_ode_inits(init_vals_dic)
        #
        # # set additional parameters for ODE
        # self.pars_ode_rhs = {
        #     "M0": M0,
        #     "aa": kwargs["aa"],
        #     # "to_dens": (self.dens),
        #     "Rd": Rd,
        #     "ncells": kwargs["ncells"],
        #     "thetaE": thetaE,
        #     "rho": rho0,
        #     "dlnrho1dR": dlnrho1dR_0,
        #     "useSpread": kwargs["useSpread"],
        #     "thetaMax": kwargs["thetaMax"]
        # }
        #
        # # initialize time elapsed in the comoving frame
        # tt0 = self.init_elapsed_time(Rstart, beta0, Gamma0, **kwargs)
        # gammaAdi0 = kwargs["eq_gammaAdi"](Gamma0, beta0)
        # rho20 = kwargs["eq_rhoprime"](Gamma0, beta0, rho0, gammaAdi0)  # self.get_rhoprime(rho0, Gamma0)
        # self.all_v_ns = self.v_ns_init_vals + ["rho", "tt", "R", "thickness", "U_e", "beta", "rho2", "gammaAdi"]
        # self.vals = np.zeros((1, len(self.all_v_ns)))
        # self.vals[0, :len(self.initial_data)] = self.initial_data
        # self.vals[0, len(self.initial_data):] = np.array([rho0, tt0, Rstart, 0., 0., beta0, rho20, gammaAdi0])
        # self.vals[0, self.i_nv("U_e")] = self.get_U_e(idx=0)
        # self.vals[0] = self.apply_units(self.vals[0])
        # # self.vals = np.hstack((self.apply_units(np.copy(self.initial_data)),
        # #                            np.array([rho0, tt0, Rstart, 0., 0., beta0])))
        #
        # # set the RHS
        # self.rhs = Peer_rhs()
        #
        # self.kwargs = kwargs
        #
        # super(Driver_Peer_FS, self).__init__(Rstart, kwargs["ode_rtol"], kwargs["ode_nsteps"])

    def get_U_e(self, idx=-1):
        return get_U_e(self.vals["Gamma"][idx],
                       self.vals["rho"][idx])

        # Gamma = self.get("Gamma")[idx]
        # rho = self.get("rho")[idx]

        # Gamma = self.dynamics["Gamma"][idx]
        # rho = self.dynamics["rho"][idx]
        #
        # nn = rho / cgs.mppme
        # beta = np.sqrt(1. - np.power(float(Gamma), -2))
        # TT = get_normT(Gamma, beta)
        # ada = get_adabatic_index(TT)
        # eT = (ada * Gamma + 1.) / (ada - 1) * (Gamma - 1.) * nn * cgs.mp * cgs.c ** 2.
        #
        # U_e = eT  # assumption
        #
        # return U_e

    def apply_units(self, idx):
        self.vals["M2"][idx] *= self.pars_ode_rhs["M0"]


if __name__ == '__main__':
    rho = 1e-2 * cgs.mppme
    RR = np.logspace(14., 23., 1000)
    o = Driver_Peer_FS(
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
        thetaMax=np.pi / 2.
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