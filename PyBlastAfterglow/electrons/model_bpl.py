import numpy as np
from scipy import optimize
from PyBlastAfterglow.uutils import cgs

from .electrons import Electron_Base
from .distributions import BrokenPowerLaw


def _B(U_e, eps_b):
    U_b = eps_b * U_e
    return np.sqrt(8. * np.pi * U_b)

def _gamma_max(B):
    return (6. * np.pi * cgs.qe / cgs.sigmaT / B) ** .5

def _get_gamma_c(Gamma, tt, B):
    return 6. * cgs.pi * cgs.me * cgs.c / (cgs.sigmaT * Gamma * tt * np.power(B, 2.))

def _gamma_min(Gamma, p, eps_e):
    return cgs.mp / cgs.me * (p - 2.) / (p - 1.) * eps_e * (Gamma - 1.)


# def _compute_char_lfs(
#         Gamma,
#         GammaSh,
#         U_e,
#         tt,
#         eps_e,
#         eps_b
# ):
#     p = self.p1
#
#     B = _B(U_e, eps_b) if U_e > 0 else 0.
#     gm = _gamma_min(GammaSh, p, eps_e) if U_e > 0 else 0.
#     gM = _gamma_max(B) if U_e > 0 else 0.
#     gc = _get_gamma_c(Gamma, tt, B) if U_e > 0 else 0.
#
#     return
#
#     self.data = np.vstack((self.data, np.zeros(len(self.all_v_ns))))
#     self.set_last("gamma_min", gm)
#     self.set_last("gamma_max", gM)
#     self.set_last("gamma_c", gc)
#     self.set_last("B", B)


class Electron_BPL(Electron_Base):
    """
        Class that accumulates methods to work with broken power law electron distribution

    """
    def __init__(
            self,
            r_grid,
            Gamma0,
            Gammash0,
            tt0,
            U_e_0,
            p,
            eps_e,
            eps_b,
    ):

        # set constants
        self.r_grid = r_grid
        self.eps_e = eps_e
        self.eps_b = eps_b
        self.p1 = p # slow cooling
        self.p2 = p + 1. # fast cooling (assume for BPL model)

        # allocate space
        all_v_ns = ["gamma_min", "gamma_c", "gamma_max", "B"]
        dtypes = []
        for v_n in all_v_ns:
            dtypes.append((v_n, 'f8'))
        vals = np.zeros(len(r_grid), dtype=dtypes)

        # initial data
        vals["B"][0] = _B(U_e_0, eps_b) if U_e_0 > 0 else 1.
        vals["gamma_min"][0] = _gamma_min(Gammash0, p, eps_e) if Gammash0 > 0 else 1.
        vals["gamma_max"][0] = _gamma_max(vals["B"][0]) if U_e_0 > 0 else 1e8
        vals["gamma_c"][0] = _get_gamma_c(Gamma0, tt0, vals["B"][0]) if tt0 > 0 else np.inf

        super(Electron_BPL, self).__init__(all_v_ns, vals)

    @classmethod
    def from_obj_fs(cls, dynamics, **kwargs):
        return cls(
            r_grid=dynamics.vals["R"],
            Gamma0=dynamics.vals["Gamma"][0],
            Gammash0=dynamics.vals["Gamma"][0],
            tt0=dynamics.vals["tt"][0],
            U_e_0=dynamics.vals["U_e"][0],
            **kwargs
        )

    @classmethod
    def from_obj_rs(cls, dynamics, **kwargs):
        return cls(
            r_grid=dynamics.vals["R"],
            Gamma0=dynamics.vals["Gamma"][0],
            Gammash0=dynamics.vals["Gamma43"][0],
            tt0=dynamics.vals["tt"][0],
            U_e_0=dynamics.vals["U_e_RS"][0],
            **kwargs
        )

    def _compute_char_lfs(
            self,
            idx,
            Gamma,
            GammaSh,
            U_e,
            tt
    ):
        p = self.p1 # here p1 = p and p2 = p - 1 are implicitly assumed

        B = _B(U_e, self.eps_b) if U_e > 0 else 0.
        gm = _gamma_min(GammaSh, p, self.eps_e) if U_e > 0 else 0.
        gM = _gamma_max(B) if U_e > 0 else 0.
        gc = _get_gamma_c(Gamma, tt, B) if U_e > 0 else 0.

        # initial data
        self.vals["B"][idx] = B
        self.vals["gamma_min"][idx] = gm
        self.vals["gamma_max"][idx] = gM
        self.vals["gamma_c"][idx] = gc



    def compute_char_lfs_fs(self, R, dynamics):
        """ electron distribution if forward shock """
        idx = int(np.where(self.r_grid == R)[0][0])
        assert idx > 0

        # extract dynamics values
        # Gamma = dynamics.vals["Gamma"][idx]
        # GammaSh = dynamics.vals["Gamma"][idx]
        # U_e = dynamics.vals["U_e"][idx]
        # tt = dynamics.vals["tt"][idx]

        self._compute_char_lfs(idx,
                               dynamics.vals["Gamma"][idx],
                               dynamics.vals["Gamma"][idx],
                               dynamics.vals["U_e"][idx],
                               dynamics.vals["tt"][idx])

        # p = self.p1
        #
        # # compute values
        # B = _B(U_e, self.eps_b) if U_e > 0 else 0.
        # gm = _gamma_min(GammaSh, p, self.eps_e) if U_e > 0 else 0.
        # gM = _gamma_max(B) if U_e > 0 else 0.
        # gc = _get_gamma_c(Gamma, tt, B) if U_e > 0 else 0.
        #
        # # set new plasma values
        # self.vals["B"][idx] = B
        # self.vals["gamma_min"][idx] = gm
        # self.vals["_gamma_max"][idx] = gM
        # self.vals["gamma_c"][idx] = gc

    def compute_char_lfs_rs(self, R, dynamics):
        """ electron distribution if reverse shock """
        idx = int(np.where(self.r_grid == R)[0][0])
        assert idx > 0
        #
        # Gamma = dynamics.vals["Gamma"][idx]
        # GammaSh43 = dynamics.vals["Gamma43"][idx]
        # U_e_rs = dynamics.vals["U_e_RS"][idx]
        # tt = dynamics.vals["tt"][idx]

        self._compute_char_lfs(idx,
                               dynamics.vals["Gamma"][idx],
                               dynamics.vals["Gamma43"][idx],
                               dynamics.vals["U_e_RS"][idx],
                               dynamics.vals["tt"][idx])

        # p = self.p1
        #
        # # compute values
        # B = _B(U_e_rs, self.eps_b) if U_e_rs > 0 else 0.
        # gm = _gamma_min(GammaSh43, p, self.eps_e) if U_e_rs > 0 else 0.
        # gM = _gamma_max(B) if U_e_rs > 0 else 0.
        # gc = _get_gamma_c(Gamma, tt, B) if U_e_rs > 0 else 0.
        #
        # # set new plasma values
        # self.vals["B"][idx] = B
        # self.vals["gamma_min"][idx] = gm
        # self.vals["_gamma_max"][idx] = gM
        # self.vals["gamma_c"][idx] = gc

        # self._compute_char_lfs(Gamma, GammaSh43, U_e_rs, tt)

    def compute_electron_distribution(self, R, rho):
        """
        Using the broken power law to evaluate the electron distribution manually
        :param rho:
        :return:
        """

        idx = int(np.where(self.r_grid == R)[0][0])
        assert idx > 0

        p = self.p1
        Ne = rho / cgs.mppme
        gm = self.vals["gamma_min"][idx]
        gM = self.vals["gamma_max"][idx]
        gc = self.vals["gamma_c"][idx]

        self.electrons = BrokenPowerLaw.from_normalised_density(
            n_e_tot=Ne,
            p1 = p if gm < gc else 2.,
            p2 = p + 1,
            gamma_b = gc if gm < gc else gm,
            gamma_min = gm if gm < gc else gc,
            gamma_max = gM
        )


