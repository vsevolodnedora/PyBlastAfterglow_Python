"""

"""
import numpy as np
import numba
from numba import types
from numba import int32, float32, double    # import the types
from numba.experimental import jitclass
from scipy.interpolate import (interp1d, RegularGridInterpolator)
import gc

from PyBlastAfterglow.uutils import cgs
from PyBlastAfterglow.synchrotron import freq_to_integrate

@numba.njit(["double(double, double, double)"])
def obsangle(thetas, phis, alpha_obs):
    """
    Return the cosine of the observer angle for the different shockwave
    segments in the counter jet and observer at an angle alpha_obs with respect to the jet axis
    (contained in yz plane)

    """
    # u_obs_x, u_obs_y, u_obs_z = 0., sin(alpha_obs), cos(alpha_obs)
    u_obs_y, u_obs_z = np.sin(alpha_obs), np.cos(alpha_obs)

    # seg_x =
    seg_y = np.sin(thetas) * np.sin(phis)
    seg_z = np.cos(thetas)

    # return  arccos(u_obs_x*seg_x + u_obs_y*seg_y + u_obs_z*seg_z)
    return u_obs_y * seg_y + u_obs_z * seg_z

@numba.njit(["double(double, double, double)"])
def obsangle_cj(thetas, phis, alpha_obs):
    """
    Return the cosine of the observer angle for the different shockwave
    segments in the counter jet and observer at an angle alpha_obs with respect to the jet axis
    (contained in yz plane)
    """
    # u_obs_x, u_obs_y, u_obs_z = 0., sin(alpha_obs), cos(alpha_obs)
    u_obs_y, u_obs_z = np.sin(alpha_obs), np.cos(alpha_obs)

    seg_y = np.sin(np.pi - thetas) * np.sin(phis)
    seg_z = np.cos(np.pi - thetas)

    # return  arccos(u_obs_x*seg_x + u_obs_y*seg_y + u_obs_z*seg_z)
    return u_obs_y * seg_y + u_obs_z * seg_z

@numba.njit(["double[:](double[:], double[:], double[:,:], double[:], double[:])"])
def interp2d(x, y, values, targetxs, targetys):

    nx=len(x)
    ny=len(y)

    ntarget=targetxs.shape[0]

    result=np.zeros((ntarget,))

    for targ in range(ntarget):
        westix=len(x)-2
        eastix=len(x)-1
        for ix in range(1,nx):
            if targetxs[targ] <= x[ix]:
                westix=ix-1
                eastix=ix
                break

        southiy=len(y)-2
        northiy=len(y)-1
        for iy in range(1,ny):
            if targetys[targ] <= y[iy]:
                southiy=iy-1
                northiy=iy
                break

        xratio=(targetxs[targ]-x[westix])/(x[eastix]-x[westix])
        yratio=(targetys[targ]-y[southiy])/(y[northiy]-y[southiy])

        lowerresult=values[southiy,westix]+(values[southiy,eastix]-values[southiy,westix])*xratio+(values[northiy,westix]-values[southiy,westix])*yratio+(values[northiy,eastix]-values[northiy,westix]-values[southiy,eastix]+values[southiy,westix])*xratio*yratio
        upperresult=values[southiy,westix]+(values[southiy,eastix]-values[southiy,westix])*xratio+(values[northiy,westix]-values[southiy,westix])*yratio+(values[northiy,eastix]-values[northiy,westix]-values[southiy,eastix]+values[southiy,westix])*xratio*yratio

        result[targ]=lowerresult+(upperresult-lowerresult)

    return result

@numba.njit(["double[:](double[:], double[:], double[:])"])
def interp1d(x, values, targetxs):

    nx=len(x)

    ntarget=targetxs.shape[0]

    result=np.zeros((ntarget,))

    for targ in range(ntarget):
        westix=len(x)-2
        eastix=len(x)-1
        for ix in range(1,nx):
            if targetxs[targ] <= x[ix]:
                westix=ix-1
                eastix=ix
                break

        xratio=(targetxs[targ]-x[westix])/(x[eastix]-x[westix])

        lowerresult=values[westix]+(values[eastix]-values[westix])*xratio+(values[westix]-values[westix])+(values[eastix]-values[westix]-values[eastix]+values[westix])*xratio
        upperresult=values[westix]+(values[eastix]-values[westix])*xratio+(values[westix]-values[westix])+(values[eastix]-values[westix]-values[eastix]+values[westix])*xratio

        result[targ]=lowerresult+(upperresult-lowerresult)

    return result

# @numba.njit("double[:](double[:])")
# def get_beta(Gamma):
#     return np.sqrt(1. - np.power(Gamma, -2))

# @numba.njit("double[:](double[:], double)")
# def get_delta_D(Gamma_obs, calpha):
#     beta_obs = np.sqrt(1. - np.power(Gamma_obs, -2))
#     delta_D = Gamma_obs * (1. - beta_obs * calpha)
#     return delta_D

# @numba.njit()
def interpolate_flux(interpolator, freqprime, radii):
    freqsprime = np.full_like(radii, freqprime)
    power = interpolator((radii, freqsprime))
    return power  # np.power(10., power)

# @numba.njit()
def compute_fluxes(
        light_curve,
        ncells,
        layer,
        cphis,
        cthetas0,
        tts,
        Rs,
        Gammas,
        spectra,
        timegrid,
        alpha_obs,
        freq_z,
        jet = 'principle',
):

    if jet == 'principle':
        obsangle_func = obsangle
    elif jet == 'counter':
        obsangle_func = obsangle_cj
    else:
        raise NameError("Jet type is not recognized")

    for ii in range(ncells):  # tqdm(range(self.ncells)):
        il = layer[ii] - 1
        phi_cell = cphis[ii]
        theta_cellR = cthetas0[il]  # Value! -> ARRAY [Gavin suggested use 0]

        calphaR = obsangle_func(theta_cellR, phi_cell, alpha_obs)  # 0. for forward jet

        ttobs = tts[il, :] + Rs[il, :] / 2.9979e10 * (1. - calphaR)  # arr
        aval_times = (np.min(ttobs) < timegrid) & (timegrid <= np.max(ttobs))
        # r_obs = interp1d(ttobs, Rs[il, :], timegrid[aval_times])  # , copy=False)
        # Gamma_obs = interp1d(Rs[il, :], Gammas[il, :], r_obs)
        r_obs = np.interp(timegrid[aval_times], ttobs, Rs[il, :])  # , copy=False)
        Gamma_obs = np.interp(r_obs, Rs[il, :], Gammas[il, :])
        calpha = calphaR  # obsangle_func(theta_obs, phi_cell, alpha_obs)  # 0. for forward

        beta_obs = np.sqrt(1. - np.power(Gamma_obs, -2))
        delta_D = Gamma_obs * (1. - beta_obs * calpha)
        # delta_D = get_delta_D(Gamma_obs, calpha)
        freqprime = freq_z * delta_D

        fluxprime = interp2d(freq_to_integrate, Rs[il, :], spectra[il], freqprime, r_obs)
        # fluxprime = self.int_data["pprime"][il].compute(freqprime, r_obs)
        layer_eats_flux = fluxprime * np.power(delta_D, -3)
        light_curve[aval_times, il] = light_curve[aval_times, il] + layer_eats_flux

        # del layer_eats_flux
        # gc.collect()

    # return light_curve
# @jitclass(spec)
class VectorizedBilinearInterpolator:
    def __init__(self, x0, y0, z0):
        """Create a bilinear interpolator for gridded input data

        Inputs:
            x0: shape (ngridx,)
            y0: shape (ngridy,)
            z0: shape batch_shape + (ngridy, ngridx) (viewed as batches)
        """

        if z0.shape[-2:] != y0.shape + x0.shape:
            raise ValueError("The last two dimensions of z0 must match that of y0 and x0, respectively!")

        ind_x = np.argsort(x0)
        self.x0 = x0[ind_x]

        ind_y = np.argsort(y0)
        self.y0 = y0[ind_y]

        self.batch_shape = z0.shape[:-2]
        indexer = np.ix_(ind_y, ind_x)
        self.z0 = z0[..., indexer[0], indexer[1]].reshape(-1, y0.size, x0.size)  # shape (nbatch, ngridy, ngridx)

        # compute auxiliary coefficients for interpolation
        # we have ngridx-1 boxes along x and ngridy-1 boxes along y
        # for each box we need 4 coefficients for bilinear interpolation
        # see e.g. https://en.wikipedia.org/wiki/Bilinear_interpolation#Alternative_algorithm
        # construct a batch of matrices with size (ngridy-1, ngridx-1, 4, 4) to invert
        x1 = self.x0[:-1]
        x2 = self.x0[1:]
        y1 = self.y0[:-1, None]
        y2 = self.y0[1:, None]
        x1,x2,y1,y2,one = np.broadcast_arrays(x1, x2, y1, y2, 1)  # all shaped (ngridy-1, ngridx-1)

        M = np.array([[one, x1, y1, x1*y1], [one, x1, y2, x1*y2],
                      [one, x2, y1, x2*y1], [one, x2, y2, x2*y2]]).transpose((2, 3, 0, 1))  # shape (ngridy-1, ngridx-1, 4, 4)
        zvec = np.array([self.z0[:, :-1, :-1], self.z0[:, 1:, :-1], self.z0[:, :-1, 1:], self.z0[:, 1:, 1:]])  # shape (4, nbatch, ngridy-1, ngridx-1)

        self.coeffs = np.einsum('yxab,bnyx -> nyxa', np.linalg.inv(M), zvec)  # shape (nbatch, ngridy-1, ngridx-1, 4) for "a0,a1,a2,a3" coefficients
        # for a given box (i,j) the interpolated value is given by self.coeffs[:,i,j,:] @ [1, x, y, x*y]



    def __call__(self, x, y):
        """Evaluate the interpolator at the given coordinates

        Inputs:
            x: shape (noutx,)
            y: shape (nouty,)

        Output:
            z: shape batch_shape + (nouty, noutx) (see __init__)
        """

        # identify outliers (and mask at the end)
        out_x = (x < self.x0[0]) | (self.x0[-1] < x)
        out_y = (y < self.y0[0]) | (self.y0[-1] < y)

        # clip outliers, mask later
        xbox = (self.x0.searchsorted(x) - 1).clip(0, self.x0.size - 2)  # shape (noutx,) indices
        ybox = (self.y0.searchsorted(y) - 1).clip(0, self.y0.size - 2)  # shape (nouty,) indices
        indexer = np.ix_(ybox, xbox)

        xgrid,ygrid = np.meshgrid(x, y)  # both shape (nouty, noutx)

        coeffs_now = self.coeffs[:, indexer[0], indexer[1], :]  # shape (nbatch, nouty, noutx, 4)
        poly = np.array([np.ones_like(xgrid), xgrid, ygrid, xgrid*ygrid])  # shape (4, nouty, noutx)
        values =  np.einsum('nyxa,ayx -> nyx', coeffs_now, poly)  # shape (nbatch, nouty, noutx)

        # reshape final result and mask outliers
        z = values.reshape(self.batch_shape + xgrid.shape)
        z[..., out_y, :] = np.nan
        z[..., :, out_x] = np.nan

        return z


class EATS_StructuredLayersSource_Jit:

    def __init__(
            self,
            nlayers,
            ncells,
            layers,
            cphis,
            cthetas,
            Rs,
            Gammas,
            betas,
            tts,
            thickness,
            spectra
    ):

        for ii in range(nlayers):
            if Gammas[ii][0] <= 1. + 1e-5:
                print("Error! layer:{} starts with Gamma:{} too low! Removing from EATS".format(ii, Gammas[ii][0]))

        self.ncells = ncells
        self.layer = layers
        self.cphis = cphis
        self.nlayers = nlayers

        self.tts = tts
        self.Rs = Rs
        self.cthetas = cthetas
        self.cthetas0 = np.array([ctheta[0] for ctheta in cthetas])
        self.Gammas = Gammas
        self.spectra = []

        self.int_data = {
            "Gamma": [], "beta": [], "cthetas": [], "TTs": [], "pprime": [], "thickness": []
        }

        for ii in range(self.nlayers):
            # print(ii)
            self.spectra.append(np.vstack((spectra[ii])))
        #     self.int_data["Gamma"].append(interp1d(Rs[ii], Gammas[ii], kind="linear", copy=False))
        #     self.int_data["beta"].append(interp1d(Rs[ii], betas[ii], kind="linear", copy=False))
        #     self.int_data["cthetas"].append(interp1d(Rs[ii], cthetas[ii], kind="linear", copy=False))
        #     self.int_data["thickness"].append(interp1d(Rs[ii], thickness[ii], kind="linear", copy=False))
        #     # self.int_data["pprime"].append(RegularGridInterpolator(
        #     #     (Rs[ii], freq_to_integrate), np.vstack((spectra[ii])))
        #     # )
        #     self.int_data["pprime"].append(Interp2D(
        #         freq_to_integrate, Rs[ii], np.vstack((spectra[ii])))
        #     )
        #     # self.int_data["pprime"].append(VectorizedBilinearInterpolator(
        #     #     Rs[ii],
        #     #     freq_to_integrate,
        #     #     np.vstack((spectra[ii])).T
        #     # ))

    @classmethod
    def from_single_model(
            cls,
            nlayers,
            thetas,
            R,
            Gamma,
            tt,
            thickness,
            spectra
    ):

        def phi_cells_in_theta_layer(nlayers):

            ii_cells = lambda ii: 2 * ii + 1  # Origin of this EXACT equations is unknown
            cil = []
            for ii in range(nlayers):
                cil.append(ii_cells(ii))
            return (cil, np.sum(cil))

        cil, ncells = phi_cells_in_theta_layer(nlayers)

        def cthetas(nlayers, theta0):
            fac = np.arange(0, nlayers + 1) / float(nlayers)
            thetas = 2. * np.arcsin(fac * np.sin(theta0 / 2.))
            cthetas = 0.5 * (thetas[1:] + thetas[:-1])
            return cthetas

        cthetas = cthetas(nlayers, thetas[0])

        layer = np.array([])
        cphis = np.array([])
        for ii in range(nlayers):  # Loop over layers and populate the arrays
            num = cil[ii]
            layer = np.append(layer, np.ones(num) * (ii + 1))  # Layer on which the cells are
            cphis = np.append(cphis, (np.arange(0, num)) * 2. * np.pi / num)
            layer = layer.astype('int')


        cthetas = [cthetas[i] + 0.5 * (2 * thetas - 2 * thetas[0]) for i in range(nlayers)]
        beta = np.sqrt(1. - np.power(np.array(Gamma), -2))
        Rs = [R for i in range(nlayers)]
        beta = [beta for i in range(nlayers)]
        Gamma = [Gamma for i in range(nlayers)]
        tt = [tt for i in range(nlayers)]
        thickness = [thickness for i in range(nlayers)]
        spectra = [spectra for i in range(nlayers)]

        print(len(cthetas[0]))
        print(len(Rs[0]))
        assert len(cthetas[0]) == len(Rs[0])

        return cls(nlayers, ncells, layer, cphis, cthetas,
                    Rs, Gamma, beta, tt, thickness,
                    spectra)

    @classmethod
    def from_model_objs(cls, o_struct, list_o_shells, cthetas):

        return cls(
            nlayers = o_struct.nlayers,
            ncells = o_struct.ncells,
            layers = o_struct.layer,
            cphis = o_struct.cphis,
            cthetas = cthetas,
            Rs = [list_o_shells[i].dyn.get("R")[1:] for i in range(o_struct.nlayers)],
            Gammas = [list_o_shells[i].dyn.get("Gamma")[1:] for i in range(o_struct.nlayers)],
            betas = [list_o_shells[i].dyn.get("beta")[1:] for i in range(o_struct.nlayers)],
            tts = [list_o_shells[i].dyn.get("tt")[1:] for i in range(o_struct.nlayers)],
            thickness = [list_o_shells[i].dyn.get("thickness")[1:] for i in range(o_struct.nlayers)],
            spectra = [list_o_shells[i].spectrum for i in range(o_struct.nlayers)],
        )

    def _get_eats_flux(self, freqprime, radii, ilayer):
        rs = radii
        freqsprime = np.full_like(radii, freqprime)
        power = self.int_data["pprime"][ilayer]((rs, freqsprime))
        return power#np.power(10., power)

    def _compute_eats_vals(self, alpha_obs, time, freq, z, jet='principle'):

        if self.nlayers == 1:
            raise ValueError("This EATS module requires nlayers >> 1 for correct calculations. Use nlayers ~ 100")

        if jet == 'principle':
            obsangle_func = obsangle
        elif jet == 'counter':
            obsangle_func = obsangle_cj
        else:
            raise NameError("Jet type is not recognized")

        # allocate momery
        fluxes = np.zeros(self.ncells)
        obs_Rs = np.zeros(self.ncells)
        obs_gams = np.zeros(self.ncells)
        obs_thetas = np.zeros(self.ncells)
        obs_calphas = np.zeros(self.ncells)
        obs_betas = np.zeros(self.ncells)

        for ii in tqdm(range(self.ncells)):
            layer = self.layer[ii] - 1
            phi_cell = self.cphis[ii]
            ctheta_cell = self.cthetas[layer][0]  # initial value

            obs_calphas[ii] = obsangle_func(ctheta_cell, phi_cell, alpha_obs)
            ttobs = self.tts[layer] + self.Rs[layer] / cgs.c * (1. - obs_calphas[ii])
            Rint = interpolate.interp1d(ttobs, self.Rs[layer])

            obs_Rs[ii] = Rint(time)
            obs_gams[ii] = self.int_data["Gamma"][layer](obs_Rs[ii])
            obs_thetas[ii] = self.int_data["cthetas"][layer](obs_Rs[ii])

            obs_betas[ii] = np.sqrt(1. - np.power(obs_gams[ii], -2))
            delta_D = obs_gams[ii] * (1. - obs_betas[ii] * obs_calphas[ii])

            freqprime = (1. + z) * freq * delta_D
            fluxprime = self.int_data["pprime"][layer]((obs_Rs[ii], freqprime))
            fluxes[ii] = fluxprime * np.power(delta_D, -3)

        return (fluxes, obs_Rs, obs_thetas, obs_gams, obs_betas, obs_calphas)

    def _loop_for_lightcurve(self, jet, alpha_obs, timegrid, freq_z):

        light_curve = np.zeros([len(timegrid), self.nlayers])

        if jet == 'principle': obsangle_func = obsangle
        elif jet == 'counter': obsangle_func = obsangle_cj
        else: raise NameError("Jet type is not recognized")

        for ii in range(self.ncells):#tqdm(range(self.ncells)):
            i_layer = self.layer[ii] - 1
            # phi coordinate point of the cell
            phi_cell = self.cphis[ii]

            # theta coordiantes of the cell
            theta_cellR = self.cthetas0[i_layer] # Value! -> ARRAY [Gavin suggested use 0]
            # theta_cellR = self.cthetas[i_layer][:]
            # calphaR = np.zeros_like(theta_cellR)

            calphaR = obsangle_func(theta_cellR, phi_cell, alpha_obs)  # 0. for forward jet

            # observer times during which radiation from 'phi, theta[]' elements arrive
            ttobs = self.tts[i_layer] + self.Rs[i_layer] / cgs.c * (1. - calphaR)  # arr
            # part of the 'emssion' observer time that falls within observation window
            aval_times = (np.min(ttobs) < timegrid) & (timegrid <= np.max(ttobs))
            # Radii values of elements emission from which is visible at 'ttobs'
            r_obs = np.interp(timegrid[aval_times], ttobs, self.Rs[i_layer])#, copy=False)
            #r_obs = Rint(timegrid[aval_times])  #
            # Quantities visible at ttobs when rad. emitted at r_obs of the jet reaches observer
            # Gamma_obs = self.int_data["Gamma"][i_layer](r_obs)
            # beta_obs = self.int_data["beta"][i_layer](r_obs)

            Gamma_obs = np.interp(r_obs, self.Rs[i_layer], self.Gammas[i_layer])
            # beta_obs = np.sqrt(1. - np.power(Gamma_obs, -2))

            # theta_obs = self.cthetas0[i_layer] # self.int_data["cthetas"][i_layer](r_obs) [Gavin suggested to use 0]
            # theta_obs = self.int_data["cthetas"][i_layer](r_obs)
            calpha = calphaR#obsangle_func(theta_obs, phi_cell, alpha_obs)  # 0. for forward

            # doppler factor
            # delta_D =  Gamma_obs * (1. - beta_obs * calpha) # (1. - beta_obs) / (1. - beta_obs * calpha)
            delta_D = get_delta_D(Gamma_obs, calpha)
            # frequency in the comoving frame
            freqprime = freq_z * delta_D
            # flux in the comoving frame
            # fluxprime = interpolate_flux(self.int_data["pprime"][i_layer], freqprime, r_obs)

            # fluxprime = numbaInterp(freq_to_integrate, self.Rs[i_layer], self.spectra[i_layer], freqprime, r_obs)
            fluxprime = self.int_data["pprime"][i_layer].compute(freqprime, r_obs)

            # part of the flux visible at times 'timegrid' from layer 'i_layer'
            layer_eats_flux = fluxprime * np.power(delta_D, -3)
            #
            light_curve[aval_times, i_layer] = light_curve[aval_times, i_layer] + layer_eats_flux

        return light_curve

    def lightcurve(self, alpha_obs, timegrid, freq, z, d_l, jet='principle'):

        # light_curve = np.zeros([len(timegrid), self.nlayers])
        if self.nlayers == 1:
            raise ValueError("This EATS module requires nlayers >> 1 for correct calculations. Use nlayers ~ 100")

        freq_z = (1. + z) * freq

        # tmp = njit(self._loop_for_lightcurve)
        # light_curve = self._loop_for_lightcurve(jet, alpha_obs, timegrid, freq_z)
        # nlayers,
        # ncells,
        # layer,
        # cphis,
        # cthetas0,
        # tts,
        # Rs,
        # Gammas,
        # spectra,
        # timegrid,
        # alpha_obs,
        # freq_z,
        # jet = 'principle'
        light_curve = np.zeros([len(timegrid), self.nlayers])
        compute_fluxes(
            light_curve,
            self.ncells,
            self.layer,
            self.cphis,
            self.cthetas0,
            np.vstack(self.tts),
            np.vstack(self.Rs),
            np.vstack(self.Gammas),
            self.spectra,
            timegrid,
            alpha_obs,
            freq_z,
            jet
        )

        light_curve *= (1. + z) / (d_l ** 2)

        # for ii in range(self.ncells):#tqdm(range(self.ncells)):
        #     i_layer = self.layer[ii] - 1
        #     # phi coordinate point of the cell
        #     phi_cell = self.cphis[ii]
        #
        #     # theta coordiantes of the cell
        #     theta_cellR = self.cthetas0[i_layer] # Value! -> ARRAY [Gavin suggested use 0]
        #     # theta_cellR = self.cthetas[i_layer][:]
        #     calphaR = obsangle_func(theta_cellR, phi_cell, alpha_obs)  # 0. for forward jet
        #
        #     # observer times during which radiation from 'phi, theta[]' elements arrive
        #     ttobs = self.tts[i_layer] + self.Rs[i_layer] / cgs.c * (1. - calphaR)  # arr
        #     # part of the 'emssion' observer time that falls within observation window
        #     aval_times = (np.min(ttobs) < timegrid) & (timegrid <= np.max(ttobs))
        #     # Radii values of elements emission from which is visible at 'ttobs'
        #     Rint = interpolate.interp1d(ttobs, self.Rs[i_layer], copy=False)
        #     r_obs = Rint(timegrid[aval_times])  #
        #     # Quantities visible at ttobs when rad. emitted at r_obs of the jet reaches observer
        #     Gamma_obs = self.int_data["Gamma"][i_layer](r_obs)
        #     beta_obs = self.int_data["beta"][i_layer](r_obs)
        #
        #     # theta_obs = self.cthetas0[i_layer] # self.int_data["cthetas"][i_layer](r_obs) [Gavin suggested to use 0]
        #     # theta_obs = self.int_data["cthetas"][i_layer](r_obs)
        #     calpha = calphaR#obsangle_func(theta_obs, phi_cell, alpha_obs)  # 0. for forward
        #
        #     # doppler factor
        #     delta_D =  Gamma_obs * (1. - beta_obs * calpha) # (1. - beta_obs) / (1. - beta_obs * calpha)
        #     # frequency in the comoving frame
        #     freqprime = freq_z * delta_D
        #     # flux in the comoving frame
        #     fluxprime = self._get_eats_flux(freqprime, r_obs, i_layer)
        #     # part of the flux visible at times 'timegrid' from layer 'i_layer'
        #     layer_eats_flux = fluxprime * np.power(delta_D, -3)
        #     #
        #     light_curve[aval_times, i_layer] = light_curve[aval_times, i_layer] + layer_eats_flux
        #
        # light_curve *= (1. + z) / (d_l ** 2) #  / 2 ??? or 4 ???
        #
        return light_curve#np.sum(light_curve, axis=1)

    def flux_at_time(self, alpha_obs, time, freq, z, d_l, jet='principle'):
        fluxes, _, _, _, _, _ = self._compute_eats_vals(alpha_obs, time, freq, z, jet=jet)
        return np.sum(fluxes) * (1. + z) / (d_l ** 2)

    def skymap(self, alpha_obs, time, freq, z, d_l):

        if self.nlayers == 1:
            raise ValueError("This EATS module requires nlayers >> 1 for correct calculations. Use nlayers ~ 100")

        # allocate memory
        obs_Rs = np.zeros(2 * self.ncells)
        obs_gams = np.zeros(2 * self.ncells)
        obs_thetas = np.zeros(2 * self.ncells)
        obs_calphas = np.zeros(2 * self.ncells)
        obs_betas = np.zeros(2 * self.ncells)
        fluxes = np.zeros(2 * self.ncells)

        # compute eats values
        fluxes[:self.ncells], obs_Rs[:self.ncells], obs_thetas[:self.ncells], \
        obs_gams[:self.ncells], obs_betas[:self.ncells], obs_calphas[:self.ncells] = \
            self._compute_eats_vals(alpha_obs, time, freq, z, jet='principle')

        fluxes[self.ncells:], obs_Rs[self.ncells:], obs_thetas[self.ncells:], \
        obs_gams[self.ncells:], obs_betas[self.ncells:], obs_calphas[self.ncells:] = \
            self._compute_eats_vals(alpha_obs, time, freq, z, jet='counter')

        # for ii in tqdm(range(self.ncells)):
        #     layer = self.layer[ii] - 1
        #     phi_cell = self.cphis[ii]
        #     ctheta_cell = self.cthetas[layer][0] # initial value
        #
        #     # --- principle jet
        #     obs_calphas[ii] = obsangle(ctheta_cell, phi_cell, alpha_obs)
        #     ttobs = self.tts[layer] + self.Rs[layer] / cgs.c * (1. - obs_calphas[ii])
        #     Rint = interpolate.interp1d(ttobs, self.Rs[layer])
        #     Robs = Rint(time)
        #
        #     obs_Rs[ii] = Robs
        #     obs_gams[ii] = self.int_data["Gamma"][layer](Robs)
        #     obs_thetas[ii] = self.int_data["cthetas"][layer](Robs)
        #
        #     obs_betas[ii] = np.sqrt(1. - np.power(obs_gams[ii], -2))
        #     delta_D = obs_gams[ii] * (1. - obs_betas[ii] * obs_calphas[ii])
        #     freqprime = (1. + z) * freq * delta_D
        #     fluxes[ii] = self.int_data["pprime"][layer]((obs_Rs[ii], freqprime)) * delta_D ** -3
        #
        #     # --- counter jet
        #     obs_calphas[ii + self.ncells] = obsangle_cj(ctheta_cell, phi_cell, alpha_obs)
        #     ttobs_cj = self.tts[layer] + self.Rs[layer] / cgs.c * (1. - obs_calphas[ii + self.ncells])
        #     Rint_cj = interpolate.interp1d(ttobs_cj, self.Rs[layer])
        #     Robs_cj = Rint_cj(time)
        #
        #     obs_Rs[ii + self.ncells] = Robs_cj
        #     obs_gams[ii + self.ncells] = self.int_data["Gamma"][layer](Robs_cj)
        #     obs_thetas[ii + self.ncells] = self.int_data["cthetas"][layer](Robs_cj)
        #
        #     obs_betas[ii + self.ncells] = np.sqrt(1. - np.power(obs_gams[ii + self.ncells], -2))
        #     delta_D = obs_gams[ii + self.ncells] * (1. - obs_betas[ii + self.ncells] * obs_calphas[ii + self.ncells])
        #     freqprime = (1. + z) * freq * delta_D
        #     fluxes[ii + self.ncells] = self.int_data["pprime"][layer]((obs_Rs[ii + self.ncells], freqprime)) * delta_D ** -3

        # generate image coordiantes
        im_xxs, im_yys = np.zeros(2 * self.ncells), np.zeros(2 * self.ncells)

        # Principal jet
        im_xxs[:self.ncells] = -1. * np.cos(alpha_obs) * \
                               np.sin(obs_thetas[:self.ncells]) * \
                               np.sin(self.cphis) + \
                               np.sin(alpha_obs) * \
                               np.cos(obs_thetas[:self.ncells])
        im_yys[:self.ncells] = np.sin(obs_thetas[:self.ncells]) * \
                               np.cos(self.cphis)

        # Counter jet
        im_xxs[self.ncells:] = -1. * np.cos(alpha_obs) * \
                               np.sin(np.pi - obs_thetas[self.ncells:]) * \
                               np.sin(self.cphis) + \
                               np.sin(alpha_obs) * \
                               np.cos(np.pi - obs_thetas[self.ncells:])
        im_yys[self.ncells:] = np.sin(np.pi - obs_thetas[self.ncells:]) * \
                               np.cos(self.cphis)

        return (fluxes, obs_Rs * im_xxs, obs_Rs * im_yys, obs_Rs)











spec = [
    ('x', double[:]),               # a simple scalar field
    ('y', double[:]),          # an array field
    ('values', double[:,:]),          # an array field
]
@jitclass(spec)
class Interp2D:
    def __init__(self, x, y, values):
        self.x = x
        self.y = y
        self.values = values


    def compute(self, targetxs, targetys):

        nx = len(self.x)
        ny = len(self.y)

        ntarget = targetxs.shape[0]

        result = np.zeros((ntarget,))

        for targ in range(ntarget):
            westix = len(self.x) - 2
            eastix = len(self.x) - 1
            for ix in range(1, nx):
                if targetxs[targ] <= self.x[ix]:
                    westix = ix - 1
                    eastix = ix
                    break

            southiy = len(self.y) - 2
            northiy = len(self.y) - 1
            for iy in range(1, ny):
                if targetys[targ] <= self.y[iy]:
                    southiy = iy - 1
                    northiy = iy
                    break

            xratio = (targetxs[targ] - self.x[westix]) / (self.x[eastix] - self.x[westix])
            yratio = (targetys[targ] - self.y[southiy]) / (self.y[northiy] - self.y[southiy])

            lowerresult = self.values[southiy, westix] + (self.values[southiy, eastix] - self.values[southiy, westix]) * xratio + (
                        self.values[northiy, westix] - self.values[southiy, westix]) * yratio + (
                                      self.values[northiy, eastix] - self.values[northiy, westix] - self.values[southiy, eastix] +
                                      self.values[southiy, westix]) * xratio * yratio
            upperresult = self.values[southiy, westix] + (self.values[southiy, eastix] - self.values[southiy, westix]) * xratio + (
                        self.values[northiy, westix] - self.values[southiy, westix]) * yratio + (
                                      self.values[northiy, eastix] - self.values[northiy, westix] - self.values[southiy, eastix] +
                                      self.values[southiy, westix]) * xratio * yratio

            result[targ] = lowerresult + (upperresult - lowerresult)

        return result


@numba.njit()#(parallel=True)
def ring_fluxes(ii, ctheta0, timegrid, cphis, alpha_obs, freq_z,
                cthetas, tts, Rs, Gamma, spectra2d, obs_flux, obs_angle_func):
    """
    RegularGridInterpolator((
                Rs[ii],
                freq_to_integrate),
                np.vstack((spectra[ii])))
            )
    spectra = [R:,freq:]

    :param ii:
    :param ctheta0:
    :param timegrid:
    :param cphis:
    :param alpha_obs:
    :param freq_z:
    :param cthetas:
    :param tts:
    :param Rs:
    :param Gamma:
    :param spectra2d:
    :return:
    """

    cgs_c = 2.9979e10
    # x = np.interp(1., np.array([1., 2., 4., 5.], np.array([2., 4., 6., 8.])), Rs)
    # return 0

    for iphi in range(len(cphis)):  # loop over phis
        phi_cell = cphis[iphi]
        for it in range(len(timegrid)):  # loop over times
            timestep = timegrid[it]
            calphaR = obs_angle_func(ctheta0, phi_cell, alpha_obs) # float
            ttobs = tts + Rs / cgs_c * (1. - calphaR)  # -> 1D array
            if (timestep > min(ttobs)) and (timestep < max(ttobs)):
                r_obs = np.interp(it, ttobs, Rs)
                Gamma_obs = np.interp(r_obs, Gamma, Rs)
                beta_obs = np.sqrt(1. - (Gamma_obs ** -2))
                calpha = calphaR
                delta_D = Gamma_obs * (1. - beta_obs * calpha)
                freqprime = freq_z * delta_D
                # --- very crude 2D interpolation
                if r_obs < min(Rs) or r_obs > max(Rs):
                    raise ValueError()
                if r_obs == Rs[0]:
                    fluxprime = np.interp(freqprime, freq_to_integrate, spectra2d[0, :])
                elif r_obs == Rs[-1]:
                    fluxprime = np.interp(freqprime, freq_to_integrate, spectra2d[-1, :])
                else:
                    ridx1 = np.argmax(Rs > r_obs)
                    ridx2 = np.argmin(Rs < r_obs)
                    fluxprime1 = np.interp(freqprime, freq_to_integrate, spectra2d[ridx1, :])
                    fluxprime2 = np.interp(freqprime, freq_to_integrate, spectra2d[ridx2, :])
                    fluxprime = (fluxprime1 + fluxprime2) / 2.
                # ------------------------------
                obs_flux[it, ii, iphi] = fluxprime * delta_D ** -3


# def compute_fluxes(timegrid, cphis, cthetas0, alpha_obs, freq_z,
#            cthetas, tts, Rs, Gamma, spectra, jet='principle'
#            ):
#     """
#     :param timegrid:
#     :param cphis:
#     :param cthetas0:
#     :param alpha_obs:
#     :param freq_z:
#     :param cthetas:
#     :param tts:
#     :param Rs:
#     :param Gamma:
#     :param spectra:
#     :return:
#     """
#     # allocate memory
#     obs_flux = np.zeros((len(timegrid), len(cthetas0), len(cphis)))
#     # run main
#     if jet == 'principle': obs_angle_func = obsangle
#     else: obs_angle_func = obsangle_cj
#     for ii in range(len(cthetas0)): # loop over layers
#         # print(ii)
#         theta_cellR = cthetas0[ii]
#         ring_fluxes(ii, theta_cellR, timegrid, cphis, alpha_obs, freq_z,
#                     cthetas[ii], tts[ii], Rs[ii], Gamma[ii], np.vstack((spectra[ii])),
#                     obs_flux, obs_angle_func)
#     # ---------
#     return obs_flux

class broken_EATS_StructuredLayersSource_Jit():
    def __init__(
            self,
            nlayers,
            ncells,
            layers,
            cphis,
            cthetas,
            Rs,
            Gammas,
            betas,
            tts,
            thickness,
            spectra
    ):

        for ii in range(nlayers):
            if Gammas[ii][0] <= 1. + 1e-5:
                print("Error! layer:{} starts with Gamma:{} too low! Removing from EATS".format(ii, Gammas[ii][0]))

        self.cphis = cphis
        self.nlayers = nlayers

        self.tts = tts
        self.Rs = Rs
        self.cthetas = cthetas
        self.cthetas0 = np.array([ctheta[0] for ctheta in cthetas])
        self.Gammas = Gammas
        self.spectra = spectra

    @classmethod
    def from_single_model(
            cls,
            nlayers,
            thetas,
            R,
            Gamma,
            tt,
            thickness,
            spectra
    ):

        def phi_cells_in_theta_layer(nlayers):

            ii_cells = lambda ii: 2 * ii + 1  # Origin of this EXACT equations is unknown
            cil = []
            for ii in range(nlayers):
                cil.append(ii_cells(ii))
            return (cil, np.sum(cil))

        cil, ncells = phi_cells_in_theta_layer(nlayers)

        def cthetas(nlayers, theta0):
            fac = np.arange(0, nlayers + 1) / float(nlayers)
            thetas = 2. * np.arcsin(fac * np.sin(theta0 / 2.))
            cthetas = 0.5 * (thetas[1:] + thetas[:-1])
            return cthetas

        cthetas = cthetas(nlayers, thetas[0])

        layer = np.array([])
        cphis = np.array([])
        for ii in range(nlayers):  # Loop over layers and populate the arrays
            num = cil[ii]
            layer = np.append(layer, np.ones(num) * (ii + 1))  # Layer on which the cells are
            cphis = np.append(cphis, (np.arange(0, num)) * 2. * np.pi / num)
            layer = layer.astype('int')

        cthetas = [cthetas[i] + 0.5 * (2 * thetas - 2 * thetas[0]) for i in range(nlayers)]
        beta = np.sqrt(1. - np.power(np.array(Gamma), -2))
        Rs = [R for i in range(nlayers)]
        beta = [beta for i in range(nlayers)]
        Gamma = [Gamma for i in range(nlayers)]
        tt = [tt for i in range(nlayers)]
        thickness = [thickness for i in range(nlayers)]
        spectra = [spectra for i in range(nlayers)]

        print(len(cthetas[0]))
        print(len(Rs[0]))
        assert len(cthetas[0]) == len(Rs[0])

        return cls(nlayers, ncells, layer, cphis, cthetas,
                   Rs, Gamma, beta, tt, thickness,
                   spectra)

    @classmethod
    def from_model_objs(cls, o_struct, list_o_shells, cthetas):

        return cls(
            nlayers=o_struct.nlayers,
            ncells=o_struct.ncells,
            layers=o_struct.layer,
            cphis=o_struct.cphis,
            cthetas=cthetas,
            Rs=[list_o_shells[i].dyn.get("R")[1:] for i in range(o_struct.nlayers)],
            Gammas=[list_o_shells[i].dyn.get("Gamma")[1:] for i in range(o_struct.nlayers)],
            betas=[list_o_shells[i].dyn.get("beta")[1:] for i in range(o_struct.nlayers)],
            tts=[list_o_shells[i].dyn.get("tt")[1:] for i in range(o_struct.nlayers)],
            thickness=[list_o_shells[i].dyn.get("thickness")[1:] for i in range(o_struct.nlayers)],
            spectra=[list_o_shells[i].spectrum for i in range(o_struct.nlayers)],
        )

    def lightcurve(self, alpha_obs, timegrid, freq, z, d_l, jet='principle'):
        freq_z = (1. + z) * freq
        obs_flux = compute_fluxes(
            timegrid,
            self.cphis,
            self.cthetas0,
            alpha_obs,
            freq_z,
            self.cthetas,
            self.tts,
            self.Rs,
            self.Gammas,
            self.spectra,
            jet
           )
        lightcurve = np.sum(obs_flux, axis=(1, 2))
        lightcurve *= (1. + z) / (d_l ** 2)  # freq_z = (1. + z) * freq
        return lightcurve















    # for it in range(timegrid):
    #     timestep = timegrid[it]
    #     for iphi, in range(cphis):
    #         phi_cell = cphis[iphi]
    #         for itheta0 in range(cthetas0):
    #             theta_cellR = cthetas0[itheta0]
    #             calphaR = obsangle(theta_cellR, phi_cell, alpha_obs)
    #             # observer times during which radiation from 'phi, theta[]' elements arrive
    #             tts1d = tts2d[itheta0, :]
    #             Rs1d = Rs2d[itheta0, :]
    #             Gamma1d = Gamma2d[itheta0, :]
    #
    #             ttobs = tts1d + Rs1d / cgs.c * (1. - calphaR) # -> 1D array
    #             if timestep > min(ttobs) and timestep < max(ttobs):
    #                 r_obs = np.interp(it, ttobs, Rs1d)
    #                 Gamma_obs = np.interp(r_obs, Gamma1d, Rs1d)
    #                 beta_obs = np.sqrt(1. - (Gamma_obs ** -2))
    #                 calpha = calphaR
    #                 delta_D = Gamma_obs * (1. - beta_obs * calpha)
    #                 freqprime = freq_z * delta_D
    #                 fluxprime = self._get_eats_flux(freqprime, r_obs, i_layer)


# def fluxes(timegrid, cphis, cthetas0, alpha_obs, freq_z,
#            cthetas2d, tts2d, Rs2d, Gamma2d,
#            ):
#
#     for it in range(timegrid):
#         timestep = timegrid[it]
#         for iphi, in range(cphis):
#             phi_cell = cphis[iphi]
#             for itheta0 in range(cthetas0):
#                 theta_cellR = cthetas0[itheta0]
#                 calphaR = obsangle(theta_cellR, phi_cell, alpha_obs)
#                 # observer times during which radiation from 'phi, theta[]' elements arrive
#                 tts1d = tts2d[itheta0, :]
#                 Rs1d = Rs2d[itheta0, :]
#                 Gamma1d = Gamma2d[itheta0, :]
#
#                 ttobs = tts1d + Rs1d / cgs.c * (1. - calphaR) # -> 1D array
#                 if timestep > min(ttobs) and timestep < max(ttobs):
#                     r_obs = np.interp(it, ttobs, Rs1d)
#                     Gamma_obs = np.interp(r_obs, Gamma1d, Rs1d)
#                     beta_obs = np.sqrt(1. - (Gamma_obs ** -2))
#                     calpha = calphaR
#                     delta_D = Gamma_obs * (1. - beta_obs * calpha)
#                     freqprime = freq_z * delta_D
#                     fluxprime = self._get_eats_flux(freqprime, r_obs, i_layer)

                    # rs = radii
                    # freqsprime = np.full_like(radii, freqprime)
                    # power = self.int_data["pprime"][ilayer]((rs, freqsprime))
                    # return power  # np.power(10., power)

# @njit
# def _loop_for_lightcurve_fs(alpha_obs, timegrid, freq_z,
#                             layer, ncells, cthetas0, cphis, tts, Rs,
#                             int_Gamma, int_beta,
#                             mode=0):
#     light_curve = np.zeros([len(timegrid), self.nlayers])
#
#     # if jet == 'principle':
#     #     obsangle_func = obsangle
#     # elif jet == 'counter':
#     #     obsangle_func = obsangle_cj
#     # else:
#     #     raise NameError("Jet type is not recognized")
#
#     for ii in range(ncells):  # tqdm(range(self.ncells)):
#         i_layer = layer[ii] - 1
#         # phi coordinate point of the cell
#         phi_cell = cphis[ii]
#
#         # theta coordiantes of the cell
#         theta_cellR = cthetas0[i_layer]  # Value! -> ARRAY [Gavin suggested use 0]
#         # theta_cellR = self.cthetas[i_layer][:]
#         calphaR = obsangle(theta_cellR, phi_cell, alpha_obs)  # 0. for forward jet
#
#         # observer times during which radiation from 'phi, theta[]' elements arrive
#         ttobs = tts[i_layer] + Rs[i_layer] / cgs.c * (1. - calphaR)  # arr
#         # part of the 'emssion' observer time that falls within observation window
#         aval_times = (np.min(ttobs) < timegrid) & (timegrid <= np.max(ttobs))
#         # Radii values of elements emission from which is visible at 'ttobs'
#         Rint = interpolate.interp1d(ttobs, Rs[i_layer], copy=False)
#         r_obs = Rint(timegrid[aval_times])  #
#         # Quantities visible at ttobs when rad. emitted at r_obs of the jet reaches observer
#         Gamma_obs = self.int_data["Gamma"][i_layer](r_obs)
#         beta_obs = self.int_data["beta"][i_layer](r_obs)
#
#         # theta_obs = self.cthetas0[i_layer] # self.int_data["cthetas"][i_layer](r_obs) [Gavin suggested to use 0]
#         # theta_obs = self.int_data["cthetas"][i_layer](r_obs)
#         calpha = calphaR  # obsangle_func(theta_obs, phi_cell, alpha_obs)  # 0. for forward
#
#         # doppler factor
#         delta_D = Gamma_obs * (1. - beta_obs * calpha)  # (1. - beta_obs) / (1. - beta_obs * calpha)
#         # frequency in the comoving frame
#         freqprime = freq_z * delta_D
#         # flux in the comoving frame
#         fluxprime = self._get_eats_flux(freqprime, r_obs, i_layer)
#         # part of the flux visible at times 'timegrid' from layer 'i_layer'
#         layer_eats_flux = fluxprime * np.power(delta_D, -3)
#         #
#         light_curve[aval_times, i_layer] = light_curve[aval_times, i_layer] + layer_eats_flux
#
#     return light_curve
#
#



#
# class EATS_StructuredLayersSource_Jit:
#
#     def __init__(
#             self,
#             nlayers,
#             ncells,
#             layers,
#             cphis,
#             cthetas,
#             Rs,
#             Gammas,
#             betas,
#             tts,
#             thickness,
#             spectra
#     ):
#
#         for ii in range(nlayers):
#             if Gammas[ii][0] <= 1. + 1e-5:
#                 print("Error! layer:{} starts with Gamma:{} too low! Removing from EATS".format(ii, Gammas[ii][0]))
#
#
#
#
#         self.ncells = ncells
#         self.layer = layers
#         self.cphis = cphis
#         self.nlayers = nlayers
#
#         self.tts = tts
#         self.Rs = Rs
#         self.cthetas = cthetas
#         self.cthetas0 = np.array([ctheta[0] for ctheta in cthetas])
#         self.Gammas = Gammas
#         self.spectra = spectra
#
#         self.int_data = {
#             "Gamma": [], "beta": [], "cthetas": [], "TTs": [], "pprime": [], "thickness": []
#         }
#
#         for ii in range(self.nlayers):
#             # print(ii)
#             self.int_data["Gamma"].append(interpolate.interp1d(Rs[ii], Gammas[ii], kind="linear", copy=False))
#             self.int_data["beta"].append(interpolate.interp1d(Rs[ii], betas[ii], kind="linear", copy=False))
#             self.int_data["cthetas"].append(interpolate.interp1d(Rs[ii], cthetas[ii], kind="linear", copy=False))
#             self.int_data["thickness"].append(interpolate.interp1d(Rs[ii], thickness[ii], kind="linear", copy=False))
#             self.int_data["pprime"].append(interpolate.RegularGridInterpolator((
#                 Rs[ii],
#                 freq_to_integrate),
#                 np.vstack((spectra[ii])))
#             )
#
#     @classmethod
#     def from_single_model(
#             cls,
#             nlayers,
#             thetas,
#             R,
#             Gamma,
#             tt,
#             thickness,
#             spectra
#     ):
#
#         def phi_cells_in_theta_layer(nlayers):
#
#             ii_cells = lambda ii: 2 * ii + 1  # Origin of this EXACT equations is unknown
#             cil = []
#             for ii in range(nlayers):
#                 cil.append(ii_cells(ii))
#             return (cil, np.sum(cil))
#
#         cil, ncells = phi_cells_in_theta_layer(nlayers)
#
#         def cthetas(nlayers, theta0):
#             fac = np.arange(0, nlayers + 1) / float(nlayers)
#             thetas = 2. * np.arcsin(fac * np.sin(theta0 / 2.))
#             cthetas = 0.5 * (thetas[1:] + thetas[:-1])
#             return cthetas
#
#         cthetas = cthetas(nlayers, thetas[0])
#
#         layer = np.array([])
#         cphis = np.array([])
#         for ii in range(nlayers):  # Loop over layers and populate the arrays
#             num = cil[ii]
#             layer = np.append(layer, np.ones(num) * (ii + 1))  # Layer on which the cells are
#             cphis = np.append(cphis, (np.arange(0, num)) * 2. * np.pi / num)
#             layer = layer.astype('int')
#
#
#         cthetas = [cthetas[i] + 0.5 * (2 * thetas - 2 * thetas[0]) for i in range(nlayers)]
#         beta = np.sqrt(1. - np.power(np.array(Gamma), -2))
#         Rs = [R for i in range(nlayers)]
#         beta = [beta for i in range(nlayers)]
#         Gamma = [Gamma for i in range(nlayers)]
#         tt = [tt for i in range(nlayers)]
#         thickness = [thickness for i in range(nlayers)]
#         spectra = [spectra for i in range(nlayers)]
#
#         print(len(cthetas[0]))
#         print(len(Rs[0]))
#         assert len(cthetas[0]) == len(Rs[0])
#
#         return cls(nlayers, ncells, layer, cphis, cthetas,
#                     Rs, Gamma, beta, tt, thickness,
#                     spectra)
#
#     @classmethod
#     def from_model_objs(cls, o_struct, list_o_shells, cthetas):
#
#         return cls(
#             nlayers = o_struct.nlayers,
#             ncells = o_struct.ncells,
#             layers = o_struct.layer,
#             cphis = o_struct.cphis,
#             cthetas = cthetas,
#             Rs = [list_o_shells[i].dyn.get("R")[1:] for i in range(o_struct.nlayers)],
#             Gammas = [list_o_shells[i].dyn.get("Gamma")[1:] for i in range(o_struct.nlayers)],
#             betas = [list_o_shells[i].dyn.get("beta")[1:] for i in range(o_struct.nlayers)],
#             tts = [list_o_shells[i].dyn.get("tt")[1:] for i in range(o_struct.nlayers)],
#             thickness = [list_o_shells[i].dyn.get("thickness")[1:] for i in range(o_struct.nlayers)],
#             spectra = [list_o_shells[i].spectrum for i in range(o_struct.nlayers)],
#         )
#
#     def _get_eats_flux(self, freqprime, radii, ilayer):
#         rs = radii
#         freqsprime = np.full_like(radii, freqprime)
#         power = self.int_data["pprime"][ilayer]((rs, freqsprime))
#         return power#np.power(10., power)
#
#     def _compute_eats_vals(self, alpha_obs, time, freq, z, jet='principle'):
#
#         if self.nlayers == 1:
#             raise ValueError("This EATS module requires nlayers >> 1 for correct calculations. Use nlayers ~ 100")
#
#         if jet == 'principle':
#             obsangle_func = obsangle
#         elif jet == 'counter':
#             obsangle_func = obsangle_cj
#         else:
#             raise NameError("Jet type is not recognized")
#
#         # allocate momery
#         fluxes = np.zeros(self.ncells)
#         obs_Rs = np.zeros(self.ncells)
#         obs_gams = np.zeros(self.ncells)
#         obs_thetas = np.zeros(self.ncells)
#         obs_calphas = np.zeros(self.ncells)
#         obs_betas = np.zeros(self.ncells)
#
#         for ii in tqdm(range(self.ncells)):
#             layer = self.layer[ii] - 1
#             phi_cell = self.cphis[ii]
#             ctheta_cell = self.cthetas[layer][0]  # initial value
#
#             obs_calphas[ii] = obsangle_func(ctheta_cell, phi_cell, alpha_obs)
#             ttobs = self.tts[layer] + self.Rs[layer] / cgs.c * (1. - obs_calphas[ii])
#             Rint = interpolate.interp1d(ttobs, self.Rs[layer])
#
#             obs_Rs[ii] = Rint(time)
#             obs_gams[ii] = self.int_data["Gamma"][layer](obs_Rs[ii])
#             obs_thetas[ii] = self.int_data["cthetas"][layer](obs_Rs[ii])
#
#             obs_betas[ii] = np.sqrt(1. - np.power(obs_gams[ii], -2))
#             delta_D = obs_gams[ii] * (1. - obs_betas[ii] * obs_calphas[ii])
#
#             freqprime = (1. + z) * freq * delta_D
#             fluxprime = self.int_data["pprime"][layer]((obs_Rs[ii], freqprime))
#             fluxes[ii] = fluxprime * np.power(delta_D, -3)
#
#         return (fluxes, obs_Rs, obs_thetas, obs_gams, obs_betas, obs_calphas)
#
#     def _loop_for_lightcurve(self, jet, alpha_obs, timegrid, freq_z):
#
#         light_curve = np.zeros([len(timegrid), self.nlayers])
#
#         if jet == 'principle': obsangle_func = obsangle
#         elif jet == 'counter': obsangle_func = obsangle_cj
#         else: raise NameError("Jet type is not recognized")
#
#         for ii in range(self.ncells):#tqdm(range(self.ncells)):
#             i_layer = self.layer[ii] - 1
#             # phi coordinate point of the cell
#             phi_cell = self.cphis[ii]
#
#             # theta coordiantes of the cell
#             theta_cellR = self.cthetas0[i_layer] # Value! -> ARRAY [Gavin suggested use 0]
#             # theta_cellR = self.cthetas[i_layer][:]
#             calphaR = obsangle_func(theta_cellR, phi_cell, alpha_obs)  # 0. for forward jet
#
#             # observer times during which radiation from 'phi, theta[]' elements arrive
#             ttobs = self.tts[i_layer] + self.Rs[i_layer] / cgs.c * (1. - calphaR)  # arr
#             # part of the 'emssion' observer time that falls within observation window
#             aval_times = (np.min(ttobs) < timegrid) & (timegrid <= np.max(ttobs))
#             # Radii values of elements emission from which is visible at 'ttobs'
#             Rint = interpolate.interp1d(ttobs, self.Rs[i_layer], copy=False)
#             r_obs = Rint(timegrid[aval_times])  #
#             # Quantities visible at ttobs when rad. emitted at r_obs of the jet reaches observer
#             Gamma_obs = self.int_data["Gamma"][i_layer](r_obs)
#             beta_obs = self.int_data["beta"][i_layer](r_obs)
#
#             # theta_obs = self.cthetas0[i_layer] # self.int_data["cthetas"][i_layer](r_obs) [Gavin suggested to use 0]
#             # theta_obs = self.int_data["cthetas"][i_layer](r_obs)
#             calpha = calphaR#obsangle_func(theta_obs, phi_cell, alpha_obs)  # 0. for forward
#
#             # doppler factor
#             delta_D =  Gamma_obs * (1. - beta_obs * calpha) # (1. - beta_obs) / (1. - beta_obs * calpha)
#             # frequency in the comoving frame
#             freqprime = freq_z * delta_D
#             # flux in the comoving frame
#             fluxprime = self._get_eats_flux(freqprime, r_obs, i_layer)
#             # part of the flux visible at times 'timegrid' from layer 'i_layer'
#             layer_eats_flux = fluxprime * np.power(delta_D, -3)
#             #
#             light_curve[aval_times, i_layer] = light_curve[aval_times, i_layer] + layer_eats_flux
#
#         return light_curve
#
#     def lightcurve(self, alpha_obs, timegrid, freq, z, d_l, jet='principle'):
#
#         # light_curve = np.zeros([len(timegrid), self.nlayers])
#         if self.nlayers == 1:
#             raise ValueError("This EATS module requires nlayers >> 1 for correct calculations. Use nlayers ~ 100")
#
#         freq_z = (1. + z) * freq
#
#         # tmp = njit(self._loop_for_lightcurve)
#         light_curve = self._loop_for_lightcurve(jet, alpha_obs, timegrid, freq_z)
#         light_curve *= (1. + z) / (d_l ** 2)
#
#         # for ii in range(self.ncells):#tqdm(range(self.ncells)):
#         #     i_layer = self.layer[ii] - 1
#         #     # phi coordinate point of the cell
#         #     phi_cell = self.cphis[ii]
#         #
#         #     # theta coordiantes of the cell
#         #     theta_cellR = self.cthetas0[i_layer] # Value! -> ARRAY [Gavin suggested use 0]
#         #     # theta_cellR = self.cthetas[i_layer][:]
#         #     calphaR = obsangle_func(theta_cellR, phi_cell, alpha_obs)  # 0. for forward jet
#         #
#         #     # observer times during which radiation from 'phi, theta[]' elements arrive
#         #     ttobs = self.tts[i_layer] + self.Rs[i_layer] / cgs.c * (1. - calphaR)  # arr
#         #     # part of the 'emssion' observer time that falls within observation window
#         #     aval_times = (np.min(ttobs) < timegrid) & (timegrid <= np.max(ttobs))
#         #     # Radii values of elements emission from which is visible at 'ttobs'
#         #     Rint = interpolate.interp1d(ttobs, self.Rs[i_layer], copy=False)
#         #     r_obs = Rint(timegrid[aval_times])  #
#         #     # Quantities visible at ttobs when rad. emitted at r_obs of the jet reaches observer
#         #     Gamma_obs = self.int_data["Gamma"][i_layer](r_obs)
#         #     beta_obs = self.int_data["beta"][i_layer](r_obs)
#         #
#         #     # theta_obs = self.cthetas0[i_layer] # self.int_data["cthetas"][i_layer](r_obs) [Gavin suggested to use 0]
#         #     # theta_obs = self.int_data["cthetas"][i_layer](r_obs)
#         #     calpha = calphaR#obsangle_func(theta_obs, phi_cell, alpha_obs)  # 0. for forward
#         #
#         #     # doppler factor
#         #     delta_D =  Gamma_obs * (1. - beta_obs * calpha) # (1. - beta_obs) / (1. - beta_obs * calpha)
#         #     # frequency in the comoving frame
#         #     freqprime = freq_z * delta_D
#         #     # flux in the comoving frame
#         #     fluxprime = self._get_eats_flux(freqprime, r_obs, i_layer)
#         #     # part of the flux visible at times 'timegrid' from layer 'i_layer'
#         #     layer_eats_flux = fluxprime * np.power(delta_D, -3)
#         #     #
#         #     light_curve[aval_times, i_layer] = light_curve[aval_times, i_layer] + layer_eats_flux
#         #
#         # light_curve *= (1. + z) / (d_l ** 2) #  / 2 ??? or 4 ???
#         #
#         return light_curve#np.sum(light_curve, axis=1)
#
#     def flux_at_time(self, alpha_obs, time, freq, z, d_l, jet='principle'):
#         fluxes, _, _, _, _, _ = self._compute_eats_vals(alpha_obs, time, freq, z, jet=jet)
#         return np.sum(fluxes) * (1. + z) / (d_l ** 2)
#
#     def skymap(self, alpha_obs, time, freq, z, d_l):
#
#         if self.nlayers == 1:
#             raise ValueError("This EATS module requires nlayers >> 1 for correct calculations. Use nlayers ~ 100")
#
#         # allocate memory
#         obs_Rs = np.zeros(2 * self.ncells)
#         obs_gams = np.zeros(2 * self.ncells)
#         obs_thetas = np.zeros(2 * self.ncells)
#         obs_calphas = np.zeros(2 * self.ncells)
#         obs_betas = np.zeros(2 * self.ncells)
#         fluxes = np.zeros(2 * self.ncells)
#
#         # compute eats values
#         fluxes[:self.ncells], obs_Rs[:self.ncells], obs_thetas[:self.ncells], \
#         obs_gams[:self.ncells], obs_betas[:self.ncells], obs_calphas[:self.ncells] = \
#             self._compute_eats_vals(alpha_obs, time, freq, z, jet='principle')
#
#         fluxes[self.ncells:], obs_Rs[self.ncells:], obs_thetas[self.ncells:], \
#         obs_gams[self.ncells:], obs_betas[self.ncells:], obs_calphas[self.ncells:] = \
#             self._compute_eats_vals(alpha_obs, time, freq, z, jet='counter')
#
#         # for ii in tqdm(range(self.ncells)):
#         #     layer = self.layer[ii] - 1
#         #     phi_cell = self.cphis[ii]
#         #     ctheta_cell = self.cthetas[layer][0] # initial value
#         #
#         #     # --- principle jet
#         #     obs_calphas[ii] = obsangle(ctheta_cell, phi_cell, alpha_obs)
#         #     ttobs = self.tts[layer] + self.Rs[layer] / cgs.c * (1. - obs_calphas[ii])
#         #     Rint = interpolate.interp1d(ttobs, self.Rs[layer])
#         #     Robs = Rint(time)
#         #
#         #     obs_Rs[ii] = Robs
#         #     obs_gams[ii] = self.int_data["Gamma"][layer](Robs)
#         #     obs_thetas[ii] = self.int_data["cthetas"][layer](Robs)
#         #
#         #     obs_betas[ii] = np.sqrt(1. - np.power(obs_gams[ii], -2))
#         #     delta_D = obs_gams[ii] * (1. - obs_betas[ii] * obs_calphas[ii])
#         #     freqprime = (1. + z) * freq * delta_D
#         #     fluxes[ii] = self.int_data["pprime"][layer]((obs_Rs[ii], freqprime)) * delta_D ** -3
#         #
#         #     # --- counter jet
#         #     obs_calphas[ii + self.ncells] = obsangle_cj(ctheta_cell, phi_cell, alpha_obs)
#         #     ttobs_cj = self.tts[layer] + self.Rs[layer] / cgs.c * (1. - obs_calphas[ii + self.ncells])
#         #     Rint_cj = interpolate.interp1d(ttobs_cj, self.Rs[layer])
#         #     Robs_cj = Rint_cj(time)
#         #
#         #     obs_Rs[ii + self.ncells] = Robs_cj
#         #     obs_gams[ii + self.ncells] = self.int_data["Gamma"][layer](Robs_cj)
#         #     obs_thetas[ii + self.ncells] = self.int_data["cthetas"][layer](Robs_cj)
#         #
#         #     obs_betas[ii + self.ncells] = np.sqrt(1. - np.power(obs_gams[ii + self.ncells], -2))
#         #     delta_D = obs_gams[ii + self.ncells] * (1. - obs_betas[ii + self.ncells] * obs_calphas[ii + self.ncells])
#         #     freqprime = (1. + z) * freq * delta_D
#         #     fluxes[ii + self.ncells] = self.int_data["pprime"][layer]((obs_Rs[ii + self.ncells], freqprime)) * delta_D ** -3
#
#         # generate image coordiantes
#         im_xxs, im_yys = np.zeros(2 * self.ncells), np.zeros(2 * self.ncells)
#
#         # Principal jet
#         im_xxs[:self.ncells] = -1. * np.cos(alpha_obs) * \
#                                np.sin(obs_thetas[:self.ncells]) * \
#                                np.sin(self.cphis) + \
#                                np.sin(alpha_obs) * \
#                                np.cos(obs_thetas[:self.ncells])
#         im_yys[:self.ncells] = np.sin(obs_thetas[:self.ncells]) * \
#                                np.cos(self.cphis)
#
#         # Counter jet
#         im_xxs[self.ncells:] = -1. * np.cos(alpha_obs) * \
#                                np.sin(np.pi - obs_thetas[self.ncells:]) * \
#                                np.sin(self.cphis) + \
#                                np.sin(alpha_obs) * \
#                                np.cos(np.pi - obs_thetas[self.ncells:])
#         im_yys[self.ncells:] = np.sin(np.pi - obs_thetas[self.ncells:]) * \
#                                np.cos(self.cphis)
#
#         return (fluxes, obs_Rs * im_xxs, obs_Rs * im_yys, obs_Rs)
#
