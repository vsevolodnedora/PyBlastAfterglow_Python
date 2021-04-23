"""
    testing stuff
"""

import numpy as np
import numba as nb
import time

nb.njit()
def opt_func_1(x):
    y = 0
    for i in range(1000):
        y += np.cos(i) * x
    return x ** 2 + .5 + np.cos(x) * y

nb.njit()
def opt_func_2(x):
    y = 0
    for i in range(1000):
        y += np.cos(i) * x
    return x ** 2 - .5 - np.cos(x) * y

nb.njit()
def jit_func(x, func, sct):
    if sct["eq"] == 'Eq.1':
        pass
    return func(x)

if __name__ == '__main__':
    t1 = time.time()
    for x in range(10000):
        t1 = time.time()
        jit_func(x, opt_func_1, sct={"eq":"Eq.1"})
    t2 = time.time()
    print("delta t = {:.2f} (10^-5)".format((t2-t1)*1e5))