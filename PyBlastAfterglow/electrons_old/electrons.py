import numpy as np
from scipy import optimize
from PyBlastAfterglow.uutils import cgs

class Distribution:
    """
        Base class grouping common functionalities to be used by all electron
        distributions. Choose the function to be used for integration. The
        default is :class:`~numpy.trapz`
    """
    def __init__(self, integrator=np.trapz):
        self.integrator = integrator

    def __call__(self, *args, **kwargs):
        """ template """
        pass

    @staticmethod
    def evaluate(**kwargs):
        """ template """
        return None

    def general_integral(self, gamma_low, gamma_up, gamma_power=0, integrator=np.trapz, **kwargs):
        """integral of the electron distribution over the range gamma_low,
                gamma_up for a general set of parameters

                Parameters
                ----------
                gamma_low : float
                    lower integration limit
                gamma_up : float
                    higher integration limit
                gamma_power : int
                    power of gamma to raise the electron distribution before integration
                integrator: func
                    function to be used for integration, default is :class:`~numpy.trapz`
                kwargs : dict
                    parameters of the electron distribution
            """
        gamma = np.logspace(np.log10(gamma_low), np.log10(gamma_up), 200)
        values = self.evaluate(gamma, **kwargs)
        values *= np.power(gamma, gamma_power)
        return integrator(values, gamma, axis=0)

    def integral(self, gamma_low, gamma_up, gamma_power=0):
        """
            integral of **this particular** electron distribution over the range
        gamma_low, gamma_up

        Parameters
        ----------
        gamma_low : float
            lower integration limit
        gamma_up : float
            higher integration limit
        """
        gamma = np.logspace(np.log10(gamma_low), np.log10(gamma_up), 200)
        values = self.__call__(gamma)
        values *= np.power(gamma, gamma_power)
        return self.integrator(values, gamma, axis=0)

    @classmethod
    def from_normalised_density(cls, n_e_tot, **kwargs):
        r"""
            sets the normalisation :math:`k_e` from the total particle density
            :math:`n_{e,\,tot}`
            :n_e_tot: in 'cm-3'
        """
        # use gamma_min and gamma_max of the electron distribution as
        # integration limits
        if "gamma_min" in kwargs:
            gamma_min = kwargs.get("gamma_min")
        if "gamma_max" in kwargs:
            gamma_max = kwargs.get("gamma_max")
        k_e = n_e_tot / cls.general_integral(
            cls, gamma_low=gamma_min, gamma_up=gamma_max, gamma_power=0, k_e=1, **kwargs
        )
        return cls(k_e, **kwargs) # .to("cm-3")

    @classmethod
    def from_normalised_energy_density(cls, u_e, **kwargs):
        r"""
            sets the normalisation :math:`k_e` from the total energy density
            :math:`u_e`, Eq. 6.64 in [DermerMenon2009]_
            :u_e: is in 'ergs * cm-3'
        """
        # use gamma_min and gamma_max of the electron distribution as
        # integration limits
        if "gamma_min" in kwargs:
            gamma_min = kwargs.get("gamma_min")
        if "gamma_max" in kwargs:
            gamma_max = kwargs.get("gamma_max")
        integral = cls.general_integral(
            cls, gamma_low=gamma_min, gamma_up=gamma_max, gamma_power=1, k_e=1, **kwargs
        )
        k_e = u_e / (cgs.mec2 * integral)
        return cls(k_e, **kwargs)

    @classmethod
    def from_norm_at_gamma_1(cls, norm, **kwargs):
        r"""
            sets :math:`k_e` such that `norm` = :math:`n_e(\gamma=1)`.
            :norm: is in cm-3
        """
        k_e = norm / cls.evaluate(1, 1, **kwargs)
        return cls(k_e, **kwargs)

class PowerLaw(Distribution):
    r"""
        Class for power-law particle spectrum.
        When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

        .. math::
            n_e(\gamma') = k_e \, \gamma'^{-p} \, H(\gamma'; \gamma'_{\rm min}, \gamma'_{\rm max})

        Parameters
        ----------
        k_e : float
            spectral normalisation
        p : float
            spectral index, note it is positive by definition, will change sign in the function
        gamma_min : float
            minimum Lorentz factor of the electron distribution
        gamma_max : float
            maximum Lorentz factor of the electron distribution
        integrator: func
            function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k_e=1e-13, # * u.Unit("cm-3"),
        p=2.1,
        gamma_min=10,
        gamma_max=1e5,
        integrator=np.trapz,
    ):
        super().__init__(integrator)
        self.k_e = k_e
        self.p = p
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [self.k_e, self.p, self.gamma_min, self.gamma_max]

    @staticmethod
    def evaluate(gamma, k_e, p, gamma_min, gamma_max):
        return np.where((gamma_min <= gamma) * (gamma <= gamma_max), k_e * gamma ** (-p), 0)

    def __call__(self, gamma):
        return self.evaluate(gamma, self.k_e, self.p, self.gamma_min, self.gamma_max)

    @staticmethod
    def evaluate_SSA_integrand(gamma, k_e, p, gamma_min, gamma_max):
        r"""
        (analytical)
        integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`
        """
        return k_e * np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            -(p + 2) * np.power(gamma, -p - 1),
            0,
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma, self.k_e, self.p, self.gamma_min, self.gamma_max
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - power law\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )

class BrokenPowerLaw(Distribution):
    r"""
    Class for broken power-law particle spectrum.
    When called, the particle density
    :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \left[
        \left(\frac{\gamma'}{\gamma'_b}\right)^{-p_1} \, H(\gamma'; \gamma'_{\rm min}, \gamma'_b) +
        \left(\frac{\gamma'}{\gamma'_b}\right)^{-p_2} \, H(\gamma'; \gamma'_{b}, \gamma'_{\rm max})
        \right]

    Parameters
    ----------
    k_e : float
        spectral normalisation (cm-3)
    p1 : float
        spectral index before the break (positive by definition)
    p2 : float
        spectral index after the break (positive by definition)
    gamma_b : float
        Lorentz factor at which the change in spectral index is occurring
    gamma_min : float
        minimum Lorentz factor of the electron distribution
    gamma_max : float
        maximum Lorentz factor of the electron distribution
    integrator: func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
            self,
            k_e=1e-13, #* u.Unit("cm-3"),
            p1=2.0,
            p2=3.0,
            gamma_b=1e3,
            gamma_min=10,
            gamma_max=1e7,
            integrator=np.trapz,
        ):
        super().__init__(integrator)
        self.k_e = k_e
        self.p1 = p1
        self.p2 = p2
        self.gamma_b = gamma_b
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        ]

    @staticmethod
    def evaluate(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
        index = np.where(gamma <= gamma_b, p1, p2)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            k_e * (gamma / gamma_b) ** (-index),
            0,
        )

    def __call__(self, gamma):
        return self.evaluate(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )

    @staticmethod
    def evaluate_SSA_integrand(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
        r"""
            (analytical) integrand for the synchrotron self-absorption:
            :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`
        """
        index = np.where(gamma <= gamma_b, p1, p2)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            k_e * -(index + 2) / gamma * (gamma / gamma_b) ** (-index),
            0,
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - broken power law\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p1: {self.p1:.2f}\n"
            + f" - p2: {self.p2:.2f}\n"
            + f" - gamma_b: {self.gamma_b:.2e}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )

class LogParabola(Distribution):
    r"""
    Class for log-parabolic particle spectrum. Built on :class:`~astropy.modeling.Fittable1DModel`.
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \, \left(\frac{\gamma'}{\gamma'_0}\right)^{-(p + q \log_{10}(\gamma' / \gamma'_0))}

    Parameters
    ----------
    k_e : :class:`~astropy.units.Quantity`
        spectral normalisation
    p : float
        spectral index, note it is positive by definition, will change sign in the function
    q : float
        spectral curvature, note it is positive by definition, will change sign in the function
    gamma_0 : float
        reference Lorentz factor
    gamma_min : float
        minimum Lorentz factor of the electron distribution
    gamma_max : float
        maximum Lorentz factor of the electron distribution
    integrator: func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k_e=1e-13, #* u.Unit("cm-3"),
        p=2.0,
        q=0.1,
        gamma_0=1e3,
        gamma_min=10,
        gamma_max=1e7,
        integrator=np.trapz,
    ):
        super().__init__(integrator)
        self.k_e = k_e
        self.p = p
        self.q = q
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [self.k_e, self.p, self.q, self.gamma_0, self.gamma_min, self.gamma_max]

    @staticmethod
    def evaluate(gamma, k_e, p, q, gamma_0, gamma_min, gamma_max):
        gamma_ratio = gamma / gamma_0
        index = -p - q * np.log10(gamma_ratio)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max), k_e * gamma_ratio ** index, 0,
        )

    def __call__(self, gamma):
        return self.evaluate(
            gamma,
            self.k_e,
            self.p,
            self.q,
            self.gamma_0,
            self.gamma_min,
            self.gamma_max,
        )

    @staticmethod
    def evaluate_SSA_integrand(gamma, k_e, p, q, gamma_0, gamma_min, gamma_max):
        r"""
            (analytical) integrand for the synchrotron self-absorption:
            :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`
        """
        prefactor = -(p + 2 * q * np.log10(gamma / gamma_0) + 2) / gamma
        return prefactor * LogParabola.evaluate(
            gamma, k_e, p, q, gamma_0, gamma_min, gamma_max
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma,
            self.k_e,
            self.p,
            self.q,
            self.gamma_0,
            self.gamma_min,
            self.gamma_max,
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - log parabola\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - q: {self.q:.2f}\n"
            + f" - gamma_0: {self.gamma_0:.2e}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )

# 

class Electron_Base():

    def __init__(
            self,
            field_names,
            initial_values,
    ):

        self.all_v_ns = field_names
        self.data = np.array(initial_values)

        pass

    def i_v_n(self, v_n):
        return self.all_v_ns.index(v_n)

    def set_init_val(self, v_n, value):
        if self.data.ndim == 2:
            self.data[0, self.i_v_n(v_n)] = value
        else:
            self.data[self.i_v_n(v_n)] = value

    def set(self, v_n, value):
        self.data[:, self.i_v_n(v_n)] = value

    def set_last(self, v_n, value):
        self.data[-1, self.i_v_n(v_n)] = value

    def get(self, v_n):
        return self.data[:, self.i_v_n(v_n)]

    def get_last(self, v_n):
        return self.data[-1, self.i_v_n(v_n)]

class Electron_BPL(Electron_Base):
    """
        Class that accumulates methods to work with broken power law electron distribution

    """
    def __init__(
            self,
            Gamma0,
            Gammash0,
            tt0,
            U_e_0,
            p,
            eps_e,
            eps_b,
    ):

        self.eps_e = eps_e
        self.eps_b = eps_b
        self.p1 = p
        self.p2 = p + 1.

        # initial data
        B0 = self._B(U_e_0) if U_e_0 > 0 else 1.
        gm0 = self._gamma_min(Gammash0, p) if Gammash0 > 0 else 1.
        gM0 = self._gamma_max(B0) if U_e_0 > 0 else 1e8
        gc0 = self._get_gamma_c(Gamma0, tt0, B0) if tt0 > 0 else np.inf

        v_ns = ["gamma_min", "gamma_c", "gamma_max", "B"]
        vals = [gm0, gc0, gM0, B0]

        super(Electron_BPL, self).__init__(v_ns, vals)

    @classmethod
    def from_obj_fs(cls, dynamics, **kwargs):
        return cls(
            Gamma0=dynamics.get_init_val("Gamma"),
            Gammash0=dynamics.get_init_val("Gamma"),
            tt0=dynamics.get_init_val("tt"),
            U_e_0=dynamics.get_init_val("U_e"),
            **kwargs
        )

    @classmethod
    def from_obj_rs(cls, dynamics, **kwargs):
        return cls(
            Gamma0=dynamics.get_init_val("Gamma"),
            Gammash0=dynamics.get_init_val("Gamma43"),
            tt0=dynamics.get_init_val("tt"),
            U_e_0=dynamics.get_init_val("U_e_RS"),
            **kwargs
        )

    def _compute_char_lfs(
            self,
            Gamma,
            GammaSh,
            U_e,
            tt
    ):
        p = self.p1

        B = self._B(U_e) if U_e > 0 else 0.
        gm = self._gamma_min(GammaSh, p) if U_e > 0 else 0.
        gM = self._gamma_max(B) if U_e > 0 else 0.
        gc = self._get_gamma_c(Gamma, tt, B) if U_e > 0 else 0.

        self.data = np.vstack((self.data, np.zeros(len(self.all_v_ns))))
        self.set_last("gamma_min", gm)
        self.set_last("gamma_max", gM)
        self.set_last("gamma_c", gc)
        self.set_last("B", B)

    def compute_char_lfs_fs(self, dynamics):

        Gamma = dynamics.get_last("Gamma")
        GammaSh = dynamics.get_last("Gamma")
        U_e = dynamics.get_last("U_e")
        tt = dynamics.get_last("tt")

        self._compute_char_lfs(Gamma, GammaSh, U_e, tt)

    def compute_char_lfs_rs(self, dynamics):

        Gamma = dynamics.get_last("Gamma")
        GammaSh43 = dynamics.get_last("Gamma43")
        U_e_rs = dynamics.get_last("U_e_RS")
        tt = dynamics.get_last("tt")

        self._compute_char_lfs(Gamma, GammaSh43, U_e_rs, tt)

    def compute_electron_distribution(self, rho):

        p = self.p1
        Ne = rho / cgs.mppme
        gm = self.get_last("gamma_min")
        gM = self.get_last("gamma_max")
        gc = self.get_last("gamma_c")

        self.electrons = BrokenPowerLaw.from_normalised_density(
            n_e_tot=Ne,
            p1 = p if gm < gc else 2.,
            p2 = p + 1,
            gamma_b = gc if gm < gc else gm,
            gamma_min = gm if gm < gc else gc,
            gamma_max = gM
        )

    def _gamma_min(self, Gamma, p):
        return cgs.mp / cgs.me * (p - 2.) / (p - 1.) * self.eps_e * (Gamma - 1.)

    def _get_gamma_c(self, Gamma, tt, B):
        return 6. * cgs.pi * cgs.me * cgs.c / (cgs.sigmaT * Gamma * tt * np.power(B,2.))

    def _gamma_max(self, B):
        return (6. * np.pi * cgs.qe / cgs.sigmaT / B) ** .5

    def _B(self, U_e):
        U_b = self.eps_b * U_e
        return np.sqrt(8. * np.pi * U_b)

class Electron_BPL_Accurate(Electron_Base):
    """
        Class that accumulates methods to work with broken power law electron distribution
        Uses the more accurate methods wherever possible
    """

    def __init__(
            self,
            Gammash0,
            U_e_0,
            p,
            eps_e,
            eps_b,
    ):

        self.eps_e = eps_e
        self.eps_b = eps_b
        self.p1 = p
        self.p2 = p + 1.

        # set initial values
        B0 = self._B(U_e_0) if U_e_0 > 0 else 1.
        gc0 = np.inf
        gM0 = self._gamma_max(B0) if B0 > 0 else 1e8
        gm0 = self._gamma_min(Gammash0, p, gc0, gM0) if Gammash0 > 0 else 1.

        v_ns = ["gamma_min", "gamma_c", "gamma_max", "B"]
        vals = [gm0, gc0, gM0, B0]

        super(Electron_BPL_Accurate, self).__init__(v_ns, vals)

    @classmethod
    def from_obj_fs(cls, dynamics, **kwargs):
        return cls(
            Gammash0=dynamics.get_init_val("Gamma"),
            U_e_0=dynamics.get_init_val("U_e"),
            **kwargs
        )

    @classmethod
    def from_obj_rs(cls, dynamics, **kwargs):
        return cls(
            Gammash0=dynamics.get_init_val("Gamma43"),
            U_e_0=dynamics.get_init_val("U_e_RS"),
            **kwargs
        )

    def _compute_char_lfs(
            self,
            GammaSh,
            U_e,
            M2,
            tcomoving
    ):

        self.data = np.vstack((self.data, np.zeros(len(self.all_v_ns))))

        p = self.p1
        B = self._B(U_e[-1]) if U_e[-1] > 0 else 0.

        self.set_last("B", B)

        gM = self._gamma_max(B) if U_e[-1] > 0 else 0.
        gc = self._get_gamma_c(M2, tcomoving, self.get("B")) if U_e[-1] > 0 else 0.
        gm = self._gamma_min(GammaSh[-1], p, gc, gM) if U_e[-1] > 0 else 0.

        self.set_last("gamma_min", gm)
        self.set_last("gamma_max", gM)
        self.set_last("gamma_c", gc)

    def compute_char_lfs_fs(self, dynamics):

        GammaSh = dynamics.get("Gamma")
        U_e = dynamics.get("U_e")
        M2 = dynamics.get("M2")
        tcomoving = dynamics.get("tcomoving")

        self._compute_char_lfs(GammaSh, U_e, M2, tcomoving)

    def compute_char_lfs_rs(self, dynamics):

        GammaSh43 = dynamics.get("Gamma43")
        U_e_rs = dynamics.get("U_e_RS")
        M3 = dynamics.get("M3")
        tcomoving = dynamics.get("tcomoving")

        self._compute_char_lfs(GammaSh43, U_e_rs, M3, tcomoving)

    def compute_electron_distribution(self, rho):
        p = self.p1
        Ne = rho / cgs.mppme
        gm = self.get_last("gamma_min")
        gM = self.get_last("gamma_max")
        gc = self.get_last("gamma_c")

        self.electrons = BrokenPowerLaw.from_normalised_density(
            n_e_tot=Ne,
            p1=p if gm < gc else 2.,
            p2=p + 1,
            gamma_b=gc if gm < gc else gm,
            gamma_min=gm if gm < gc else gc,
            gamma_max=gM
        )

    def _gamma_min(self, Gamma, p, gamma_c, gamma_max):

        def gmin_fzero(gamma_min, gamma_c, gamma_max, mue, Gamma, p):
            gamma_max_m1 = gamma_max - 1
            gamma_c_m1 = gamma_c - 1
            gamma_min_m1 = gamma_min - 1
            if gamma_c < gamma_min:
                # dN/dgamma = gamma**(-p-1) For more accuracy use: (gamma-1)**(-p-1)
                # numerator = gamma_min ** (1 - p) * (np.log(gamma_min / gamma_c) + gamma_min ** -1 + gamma_c ** -1) + (
                #             gamma_max ** (1 - p) - gamma_min ** (1 - p)) / (1 - p) + (
                #                         gamma_max ** (-p) - gamma_min ** (-p)) / p
                numerator = (gamma_max_m1**(1-p) - gamma_min_m1**(1-p)) * p
                # denominator = gamma_min ** (1 - p) * gamma_c ** -1 - gamma_min ** -p - (
                #             gamma_max ** (-p) - gamma_min ** (-p)) / p
                denominator = (gamma_min_m1**-p-gamma_max_m1**-p) * (1-p)
            else:
                # dN/dgamma = (gamma-1)**(-p) For more accuracy use gamma**(-p)
                if gamma_c > gamma_max:  gamma_c = gamma_max
                # numerator = (gamma_c ** (2 - p) - gamma_min ** (2 - p)) / (2 - p) - (
                #             gamma_c ** (1 - p) - gamma_min ** (1 - p)) / (1 - p) + (
                #                         gamma_c * gamma_max ** (1 - p) - gamma_c ** (2 - p)) / (1 - p) + (
                #                         gamma_c * gamma_max ** -p - gamma_c ** (1 - p)) / p
                numerator = (gamma_c_m1**(2-p)-gamma_min_m1**(2-p))/(2-p) + (gamma_max_m1**(1-p)-gamma_c_m1**(1-p))/(1-p)
                # denominator = (gamma_c ** (1 - p) - gamma_min ** (1 - p)) / (1 - p) - (
                #             gamma_c * gamma_max ** -p - gamma_c ** (1 - p)) / p
                denominator = (gamma_c_m1**(1-p)-gamma_min_m1**(1-p))/(1-p) - (gamma_max_m1**(-p)-gamma_c_m1**(-p))/p

            lhs = self.eps_e * (Gamma-1.) / mue
            res = lhs - (numerator / denominator)
            return res

        mup, mue = 1., cgs.me / cgs.mp
        try:
            res = optimize.bisect(gmin_fzero, a=1e-3, b=gamma_max * 1.00001, args=(gamma_c, gamma_max, mue, Gamma, p),
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

    def _get_gamma_c(self, M2, tcomoving, B):
        """ input arrays """
        gc = cgs.gamma_c_w_fac * np.trapz(1. / (np.power(B, 2) * tcomoving), M2) / (M2[-1] - M2[0])
        return gc

        # return 6. * cgs.pi * cgs.me * cgs.c / (cgs.sigmaT * Gamma * tt * np.power(B, 2.))

    def _gamma_max(self, B):
        return (6. * np.pi * cgs.qe / cgs.sigmaT / B) ** .5

    def _B(self, U_e):
        U_b = self.eps_b * U_e
        return np.sqrt(8. * np.pi * U_b)

