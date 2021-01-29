"""
    additional modulus
"""
import numpy as np

class cgs:

    pi = 3.141592653589793

    tmp = 1221461.4847847277

    c = 2.9979e10
    mp = 1.6726e-24
    me = 9.1094e-28
    # e = 1.602176634e-19 # Si
    # h = 6.62607015e-34 # ???? Si
    h = 6.6260755e-27 # erg s
    mpe = mp + me
    mue = me / mp
    hcgs = 6.6260755e-27  # Planck constant in cgs
    # si_h = 6.62607015e-34
    kB = 1.380658e-16
    sigmaT = 6.6524e-25
    qe = 4.803204e-10
    # si_qe = 1.602176634e-19
    sigma_B = 5.6704e-5  ### Stephan Boltzann constant in cgs
    lambda_c = (h / (me * c)) # Compton wavelength
    mecc2MeV = 0.511
    mec2 = 8.187105649650028e-07  # erg # electron mass in erg, mass_energy equivalence
    gamma_c_w_fac = 6 * np.pi * me * c / sigmaT
    rad_const = 4 * sigma_B / c   #### Radiation constant
    mppme = mp + me

    pc = 3.0857e18 # cm
    year= 3.154e+7 # sec
    day = 86400

    solar_m = 1.989e+33

    ns_rho = 1.6191004634e-5
    time_constant = 0.004925794970773136  # to to to ms
    energy_constant = 1787.5521500932314
    volume_constant = 2048

    sTy = 365. * 24. * 60. * 60.    # seconds to years conversion factor
    sTd = 24. * 60. * 60.           # seconds to days conversion factor
    rad2mas = 206264806.247

def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx


# type of float to be used for math operation
numpy_type = np.float64
# smallest positive float
ftiny = np.finfo(numpy_type).tiny
# largest positive float
fmax = np.finfo(numpy_type).max


def log(x):
    """clipped log to avoid RuntimeWarning: divide by zero encountered in log"""
    values = np.clip(x, ftiny, fmax)
    return np.log(values)

def trapz_loglog(y, x, axis=0):
    """
    from https://github.com/cosimoNigro/agnpy.git

    Integrate a function approximating its discrete intervals as power-laws
    (straight lines in log-log space), largely copied from naima

    Parameters
    ----------
    y : array_like
        array to integrate
    x : array_like, optional
        independent variable to integrate over
    axis : int, optional
        along which axis the integration has to be performed

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    slice_low = [slice(None)] * y.ndim
    slice_up = [slice(None)] * y.ndim
    # multi-dimensional equivalent of x_low = x[:-1]
    slice_low[axis] = slice(None, -1)
    # multi-dimensional equivalent of x_up = x[1:]
    slice_up[axis] = slice(1, None)

    # reshape x to be broadcastable with y
    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    x_low = x[tuple(slice_low)]
    x_up = x[tuple(slice_up)]
    y_low = y[tuple(slice_low)]
    y_up = y[tuple(slice_up)]

    # slope in the given logarithmic bin
    m = (log(y_low) - log(y_up)) / (log(x_low) - log(x_up))

    vals = np.where(
        np.abs(m + 1) > 1e-10,
        y_low / (m + 1) * (x_up * np.power(x_up / x_low, m) - x_low),
        x_low * y_low * log(x_up / x_low),
    )

    tozero = (
        (y[tuple(slice_low)] == 0.0)
        + (y[tuple(slice_up)] == 0.0)
        + (x[tuple(slice_low)] == x[tuple(slice_up)])
    )
    vals[tozero] = 0.0

    return np.add.reduce(vals, axis) * x_unit * y_unit


def axes_reshaper(*args):
    """reshape 1-dimensional arrays of different lengths in order for them to be
    broadcastable in multi-dimensional operations

    the rearrangement scheme for a list of n arrays is the following:
    `args[0]` is reshaped as `(args[0].size, 1, 1, ..., 1)` -> axis 0
    `args[1]` is reshaped as `(1, args[1].size, 1, ..., 1)` -> axis 1
    `args[2]` is reshaped as `(1, 1, args[2].size ..., 1)` -> axis 2
        .
        .
        .
    `args[n-1]` is reshaped as `(1, 1, 1, ..., args[n-1].size)` -> axis n-1

    Parameters
    ----------
    args: 1-dimensional `~numpy.ndarray`s to be reshaped
    """
    n = len(args)
    dim = (1,) * n
    reshaped_arrays = []
    for i, arg in enumerate(args):
        reshaped_dim = list(dim)  # the tuple is copied in the list
        reshaped_dim[i] = arg.size
        reshaped_array = np.reshape(arg, reshaped_dim)
        reshaped_arrays.append(reshaped_array)
    return reshaped_arrays