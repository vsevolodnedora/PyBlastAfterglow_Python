import numpy as np
from scipy import optimize
from PyBlastAfterglow.uutils import cgs

class Electron_Base():

    def __init__(
            self,
            field_names,
            vals,
    ):

        self.all_v_ns = field_names
        # self.data = np.array(initial_values)
        self.vals = vals


    # def i_v_n(self, v_n):
    #     return self.all_v_ns.index(v_n)
    #
    # def set_init_val(self, v_n, value):
    #     if self.data.ndim == 2:
    #         self.data[0, self.i_v_n(v_n)] = value
    #     else:
    #         self.data[self.i_v_n(v_n)] = value
    #
    # def set(self, v_n, value):
    #     self.data[:, self.i_v_n(v_n)] = value
    #
    # def set_last(self, v_n, value):
    #     self.data[-1, self.i_v_n(v_n)] = value
    #
    # def get(self, v_n):
    #     return self.data[:, self.i_v_n(v_n)]
    #
    # def get_last(self, v_n):
    #     return self.data[-1, self.i_v_n(v_n)]
    #
