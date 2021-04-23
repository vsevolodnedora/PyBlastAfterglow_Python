import numpy as np
from scipy import optimize
from PyBlastAfterglow.uutils import cgs

from .electrons import Electron_Base
from .distributions import BrokenPowerLaw


def _gamma_min(Gamma, p, gamma_c, gamma_max, eps_e):

    def gmin_fzero(gamma_min, gamma_c, gamma_max, mue, Gamma, p, eps_e):
        gamma_max_m1 = gamma_max - 1
        gamma_c_m1 = gamma_c - 1
        gamma_min_m1 = gamma_min - 1
        if gamma_c < gamma_min:
            # dN/dgamma = gamma**(-p-1) For more accuracy use: (gamma-1)**(-p-1)
            # numerator = gamma_min ** (1 - p) * (np.log(gamma_min / gamma_c) + gamma_min ** -1 + gamma_c ** -1) + (
            #             gamma_max ** (1 - p) - gamma_min ** (1 - p)) / (1 - p) + (
            #                         gamma_max ** (-p) - gamma_min ** (-p)) / p
            numerator = (gamma_max_m1 ** (1 - p) - gamma_min_m1 ** (1 - p)) * p
            # denominator = gamma_min ** (1 - p) * gamma_c ** -1 - gamma_min ** -p - (
            #             gamma_max ** (-p) - gamma_min ** (-p)) / p
            denominator = (gamma_min_m1 ** -p - gamma_max_m1 ** -p) * (1 - p)
        else:
            # dN/dgamma = (gamma-1)**(-p) For more accuracy use gamma**(-p)
            if gamma_c > gamma_max:  gamma_c = gamma_max
            # numerator = (gamma_c ** (2 - p) - gamma_min ** (2 - p)) / (2 - p) - (
            #             gamma_c ** (1 - p) - gamma_min ** (1 - p)) / (1 - p) + (
            #                         gamma_c * gamma_max ** (1 - p) - gamma_c ** (2 - p)) / (1 - p) + (
            #                         gamma_c * gamma_max ** -p - gamma_c ** (1 - p)) / p
            numerator = (gamma_c_m1 ** (2 - p) - gamma_min_m1 ** (2 - p)) / (2 - p) + (
                        gamma_max_m1 ** (1 - p) - gamma_c_m1 ** (1 - p)) / (1 - p)
            # denominator = (gamma_c ** (1 - p) - gamma_min ** (1 - p)) / (1 - p) - (
            #             gamma_c * gamma_max ** -p - gamma_c ** (1 - p)) / p
            denominator = (gamma_c_m1 ** (1 - p) - gamma_min_m1 ** (1 - p)) / (1 - p) - (
                        gamma_max_m1 ** (-p) - gamma_c_m1 ** (-p)) / p

        lhs = eps_e * (Gamma - 1.) / mue
        res = lhs - (numerator / denominator)
        return res

    mup, mue = 1., cgs.me / cgs.mp
    try:
        res = optimize.bisect(gmin_fzero, a=1e-3, b=gamma_max * 1.00001,
                              args=(gamma_c, gamma_max, mue, Gamma, p, eps_e),
                              xtol=1e-15, rtol=1e-5, maxiter=1000)
    except ValueError:
        print("Failed [ValueError]")
        res = 0.
    except RuntimeError:
        print("Failed [RuntimeError]")
        res = 0.

    # res, _, ler, msg = optimize.fsolve(gmin_fzero, x0=np.array([2.]), args=(gamma_c, gamma_max, mue))
    assert np.isfinite(res)
    return np.float64(res)
    # return cgs.mp / cgs.me * (p - 2.) / (p - 1.) * self.eps_e * (Gamma - 1.)

def _get_gamma_c(M2, tcomoving, B):
    """ input arrays """
    gc = cgs.gamma_c_w_fac * np.trapz(1. / (np.power(B, 2) * tcomoving), M2) / (M2[-1] - M2[0])
    return gc

    # return 6. * cgs.pi * cgs.me * cgs.c / (cgs.sigmaT * Gamma * tt * np.power(B, 2.))

def _gamma_max(B):
    return (6. * np.pi * cgs.qe / cgs.sigmaT / B) ** .5

def _B(U_e, eps_b):
    U_b = eps_b * U_e
    return np.sqrt(8. * np.pi * U_b)


class Electron_BPL_Accurate(Electron_Base):
    """
        Class that accumulates methods to work with broken power law electron distribution
        Uses the more accurate methods wherever possible
    """

    def __init__(
            self,
            r_grid,
            Gammash0,
            U_e_0,
            p,
            eps_e,
            eps_b,
    ):

        # set constants
        self.r_grid = r_grid
        self.eps_e = eps_e
        self.eps_b = eps_b
        self.p1 = p
        self.p2 = p + 1.

        # allocate space
        all_v_ns = ["gamma_min", "gamma_c", "gamma_max", "B"]
        dtypes = []
        for v_n in all_v_ns:
            dtypes.append((v_n, 'f8'))
        vals = np.zeros(len(r_grid), dtype=dtypes)

        # initial data
        vals["B"][0] = _B(U_e_0, eps_b) if U_e_0 > 0 else 1.
        vals["gamma_c"][0] = np.inf
        vals["gamma_max"][0] = _gamma_max(vals["B"][0]) if vals["B"][0] > 0 else 1e8
        vals["gamma_min"][0] = _gamma_min(Gammash0, p, vals["gamma_c"][0],
                                          vals["gamma_max"][0], eps_e) if Gammash0 > 0 else 1.

        super(Electron_BPL_Accurate, self).__init__(all_v_ns, vals)

        # # set initial values
        # B0 = self._B(U_e_0) if U_e_0 > 0 else 1.
        # gc0 = np.inf
        # gM0 = self._gamma_max(B0) if B0 > 0 else 1e8
        # gm0 = self._gamma_min(Gammash0, p, gc0, gM0) if Gammash0 > 0 else 1.
        #
        # v_ns = ["gamma_min", "gamma_c", "gamma_max", "B"]
        # vals = [gm0, gc0, gM0, B0]
        #
        # super(Electron_BPL_Accurate, self).__init__(v_ns, vals)

    @classmethod
    def from_obj_fs(cls, dynamics, **kwargs):
        return cls(
            r_grid=dynamics.vals["R"],
            Gammash0=dynamics.vals["Gamma"][0],
            U_e_0=dynamics.vals["U_e"][0],
            **kwargs
        )

    @classmethod
    def from_obj_rs(cls, dynamics, **kwargs):
        return cls(
            r_grid=dynamics.vals["R"],
            Gammash0=dynamics.vals["Gamma43"][0],
            U_e_0=dynamics.vals["U_e_RS"][0],
            **kwargs
        )

    def _compute_char_lfs(
            self,
            idx,
            GammaSh,
            U_e,
            M2,
            tcomoving
    ):

        # self.data = np.vstack((self.data, np.zeros(len(self.all_v_ns))))
        # self.data = np.vstack((self.data, np.zeros(len(self.all_v_ns))))
        #
        # p = self.p1
        # B = self._B(U_e[-1]) if U_e[-1] > 0 else 0.
        #
        # self.set_last("B", B)
        #
        # gM = self._gamma_max(B) if U_e[-1] > 0 else 0.
        # gc = self._get_gamma_c(M2, tcomoving, self.get("B")) if U_e[-1] > 0 else 0.
        # gm = self._gamma_min(GammaSh[-1], p, gc, gM) if U_e[-1] > 0 else 0.
        #
        # self.set_last("gamma_min", gm)
        # self.set_last("gamma_max", gM)
        # self.set_last("gamma_c", gc)
        p = self.p1
        B = _B(U_e[idx], self.eps_b) if U_e[idx] > 0 else 0.

        self.vals["B"][idx] = B

        gM = _gamma_max(B) if U_e[idx] > 0 else 0.
        gc = _get_gamma_c(M2[:idx], tcomoving[:idx], self.vals["B"][:idx]) if U_e[idx] > 0 else 0.
        gm = _gamma_min(GammaSh[idx], p, gc, gM, self.eps_e) if U_e[idx] > 0 else 0.

        self.vals["gamma_min"][idx] = gm
        self.vals["gamma_max"][idx] = gM
        self.vals["gamma_c"][idx] = gc



    def compute_char_lfs_fs(self, R, dynamics):

        # GammaSh = dynamics.vals["Gamma"]
        # U_e = dynamics.vals["U_e"]
        # M2 = dynamics.vals["M2"]
        # tcomoving = dynamics.vals["tcomoving"]
        idx = int(np.where(self.r_grid == R)[0][0])
        assert idx > 0
        self._compute_char_lfs(idx,
                               dynamics.vals["Gamma"],
                               dynamics.vals["U_e"],
                               dynamics.vals["M2"],
                               dynamics.vals["tcomoving"])

    def compute_char_lfs_rs(self, R, dynamics):

        # GammaSh43 = dynamics.get("Gamma43")
        # U_e_rs = dynamics.get("U_e_RS")
        # M3 = dynamics.get("M3")
        # tcomoving = dynamics.get("tcomoving")
        idx = int(np.where(self.r_grid == R)[0][0])
        assert idx > 0
        self._compute_char_lfs(idx,
                               dynamics.vals["Gamma43"],
                               dynamics.vals["U_e_RS"],
                               dynamics.vals["M3"],
                               dynamics.vals["tcomoving"])

    def compute_electron_distribution(self, R, rho):
        idx = int(np.where(self.r_grid == R)[0][0])
        assert idx > 0

        p = self.p1
        Ne = rho / cgs.mppme
        gm = self.vals["gamma_min"][idx]
        gM = self.vals["gamma_max"][idx]
        gc = self.vals["gamma_c"][idx]

        self.electrons = BrokenPowerLaw.from_normalised_density(
            n_e_tot=Ne,
            p1=p if gm < gc else 2.,
            p2=p + 1,
            gamma_b=gc if gm < gc else gm,
            gamma_min=gm if gm < gc else gc,
            gamma_max=gM
        )

    # def _gamma_min(self, Gamma, p, gamma_c, gamma_max):
    #
    #     def gmin_fzero(gamma_min, gamma_c, gamma_max, mue, Gamma, p):
    #         gamma_max_m1 = gamma_max - 1
    #         gamma_c_m1 = gamma_c - 1
    #         gamma_min_m1 = gamma_min - 1
    #         if gamma_c < gamma_min:
    #             # dN/dgamma = gamma**(-p-1) For more accuracy use: (gamma-1)**(-p-1)
    #             # numerator = gamma_min ** (1 - p) * (np.log(gamma_min / gamma_c) + gamma_min ** -1 + gamma_c ** -1) + (
    #             #             gamma_max ** (1 - p) - gamma_min ** (1 - p)) / (1 - p) + (
    #             #                         gamma_max ** (-p) - gamma_min ** (-p)) / p
    #             numerator = (gamma_max_m1**(1-p) - gamma_min_m1**(1-p)) * p
    #             # denominator = gamma_min ** (1 - p) * gamma_c ** -1 - gamma_min ** -p - (
    #             #             gamma_max ** (-p) - gamma_min ** (-p)) / p
    #             denominator = (gamma_min_m1**-p-gamma_max_m1**-p) * (1-p)
    #         else:
    #             # dN/dgamma = (gamma-1)**(-p) For more accuracy use gamma**(-p)
    #             if gamma_c > gamma_max:  gamma_c = gamma_max
    #             # numerator = (gamma_c ** (2 - p) - gamma_min ** (2 - p)) / (2 - p) - (
    #             #             gamma_c ** (1 - p) - gamma_min ** (1 - p)) / (1 - p) + (
    #             #                         gamma_c * gamma_max ** (1 - p) - gamma_c ** (2 - p)) / (1 - p) + (
    #             #                         gamma_c * gamma_max ** -p - gamma_c ** (1 - p)) / p
    #             numerator = (gamma_c_m1**(2-p)-gamma_min_m1**(2-p))/(2-p) + (gamma_max_m1**(1-p)-gamma_c_m1**(1-p))/(1-p)
    #             # denominator = (gamma_c ** (1 - p) - gamma_min ** (1 - p)) / (1 - p) - (
    #             #             gamma_c * gamma_max ** -p - gamma_c ** (1 - p)) / p
    #             denominator = (gamma_c_m1**(1-p)-gamma_min_m1**(1-p))/(1-p) - (gamma_max_m1**(-p)-gamma_c_m1**(-p))/p
    #
    #         lhs = self.eps_e * (Gamma-1.) / mue
    #         res = lhs - (numerator / denominator)
    #         return res
    #
    #     mup, mue = 1., cgs.me / cgs.mp
    #     try:
    #         res = optimize.bisect(gmin_fzero, a=1e-3, b=gamma_max * 1.00001, args=(gamma_c, gamma_max, mue, Gamma, p),
    #                               xtol=1e-15, rtol=1e-5, maxiter=1000)
    #     except ValueError:
    #         print("Failed [ValueError]")
    #         res = 0.
    #     except RuntimeError:
    #         print("Failed [RuntimeError]")
    #         res = 0.
    #
    #     # res, _, ler, msg = optimize.fsolve(gmin_fzero, x0=np.array([2.]), args=(gamma_c, gamma_max, mue))
    #     assert np.isfinite(res)
    #     return np.float64(res)
    #     # return cgs.mp / cgs.me * (p - 2.) / (p - 1.) * self.eps_e * (Gamma - 1.)
    #
    # def _get_gamma_c(self, M2, tcomoving, B):
    #     """ input arrays """
    #     gc = cgs.gamma_c_w_fac * np.trapz(1. / (np.power(B, 2) * tcomoving), M2) / (M2[-1] - M2[0])
    #     return gc
    #
    #     # return 6. * cgs.pi * cgs.me * cgs.c / (cgs.sigmaT * Gamma * tt * np.power(B, 2.))
    #
    # def _gamma_max(self, B):
    #     return (6. * np.pi * cgs.qe / cgs.sigmaT / B) ** .5
    #
    # def _B(self, U_e):
    #     U_b = self.eps_b * U_e
    #     return np.sqrt(8. * np.pi * U_b)