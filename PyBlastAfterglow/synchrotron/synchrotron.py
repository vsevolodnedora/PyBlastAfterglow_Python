"""
    pass
"""

import numpy as np
from PyBlastAfterglow.uutils import cgs, axes_reshaper
from scipy import interpolate


freq_to_integrate = np.logspace(5., 24., 300)

class Synchrotron:
    def __init__(self):
        pass

    def sed(self, freqs):
        """ to be orverritten in sublcass """
        pass

    def get(self, v_n):
        if v_n == "nu_m": return self.nu_m
        elif v_n == "nu_c": return self.nu_c
        elif v_n == "pprimemax": return self.pprimemax
        else:
            raise NameError("unrecognized name: {}".format(v_n))

    def get_spectrum(self):
        sed = self.sed(freq_to_integrate)
        spectrum = sed / freq_to_integrate #* boost
        return spectrum

    def get_interpolator(self, intepolator=interpolate.interp1d, **kwargs):
        sed = self.sed(freq_to_integrate)
        spectrum = sed / freq_to_integrate#  * boost
        return intepolator(freq_to_integrate, spectrum, **kwargs)
