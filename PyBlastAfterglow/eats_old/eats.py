"""

"""
import numpy as np
from scipy import interpolate
from tqdm import tqdm

# from numba import njit

from PyBlastAfterglow.uutils import cgs
from PyBlastAfterglow.synchrotron import freq_to_integrate

# TODO move the uniform sphere integrator from StructJet to Uniform

# from multiprocessing import Pool
# import os

# def dopplerFactor(cosa, beta):
#     """
#     Calculate the doppler factors of the different jethead segments
#     cosa -> cosine of observeration angle, obtained using obsangle
#     """
#     return (1. - beta) / (1. - beta * cosa)

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

def obsangle_cj(thetas, phis, alpha_obs):
    """
    Return the cosine of the observer angle for the different shockwave
    segments in the counter jet and observer at an angle alpha_obs with respect to the jet axis
    (contained in yz plane)
    """
    # u_obs_x, u_obs_y, u_obs_z = 0., sin(alpha_obs), cos(alpha_obs)
    u_obs_y, u_obs_z = np.sin(alpha_obs), np.cos(alpha_obs)

    # seg_x =
    seg_y = np.sin(np.pi - thetas) * np.sin(phis)
    seg_z = np.cos(np.pi - thetas)

    # return  arccos(u_obs_x*seg_x + u_obs_y*seg_y + u_obs_z*seg_z)
    return u_obs_y * seg_y + u_obs_z * seg_z

def generate_skyimage(fluxes, xxs, yys, nx, ny=1, fac=1, scale=False, method='linear'):

    if (ny == 1 and not scale):
        ny = nx
    elif (scale):
        dX = xxs.max() - xxs.min()
        dY = yys.max() - yys.min()
        fac = max(dX, dY) / min(dX, dY)
        print(fac)
        if (dY >= dX):
            ny = round(fac * nx)
        else:
            ny = nx
            nx = round(fac * nx)
    else:
        pass

    nx = np.complex(0, nx)
    ny = np.complex(0, ny)

    grid_x, grid_y = np.mgrid[xxs.min():xxs.max():nx, yys.min():yys.max():ny]

    image = interpolate.griddata(
        np.vstack((xxs, yys)).T,
        fluxes,
        np.array((grid_x * fac, grid_y * fac)).T,
        method=method,
        fill_value=np.nan
    )

    return (grid_x, grid_y, image)

class EATS_StructuredLayersSource:

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
        self.spectra = spectra

        self.int_data = {
            "Gamma": [], "beta": [], "cthetas": [], "TTs": [], "pprime": [], "thickness": []
        }

        for ii in range(self.nlayers):
            # print(ii)
            self.int_data["Gamma"].append(interpolate.interp1d(Rs[ii], Gammas[ii], kind="linear", copy=False))
            self.int_data["beta"].append(interpolate.interp1d(Rs[ii], betas[ii], kind="linear", copy=False))
            self.int_data["cthetas"].append(interpolate.interp1d(Rs[ii], cthetas[ii], kind="linear", copy=False))
            self.int_data["thickness"].append(interpolate.interp1d(Rs[ii], thickness[ii], kind="linear", copy=False))
            self.int_data["pprime"].append(interpolate.RegularGridInterpolator((
                Rs[ii],
                freq_to_integrate),
                np.vstack((spectra[ii])))
            )

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
            calphaR = obsangle_func(theta_cellR, phi_cell, alpha_obs)  # 0. for forward jet

            # observer times during which radiation from 'phi, theta[]' elements arrive
            ttobs = self.tts[i_layer] + self.Rs[i_layer] / cgs.c * (1. - calphaR)  # arr
            # part of the 'emssion' observer time that falls within observation window
            aval_times = (np.min(ttobs) < timegrid) & (timegrid <= np.max(ttobs))
            # Radii values of elements emission from which is visible at 'ttobs'
            Rint = interpolate.interp1d(ttobs, self.Rs[i_layer], copy=False)
            r_obs = Rint(timegrid[aval_times])  #
            # Quantities visible at ttobs when rad. emitted at r_obs of the jet reaches observer
            Gamma_obs = self.int_data["Gamma"][i_layer](r_obs)
            beta_obs = self.int_data["beta"][i_layer](r_obs)

            # theta_obs = self.cthetas0[i_layer] # self.int_data["cthetas"][i_layer](r_obs) [Gavin suggested to use 0]
            # theta_obs = self.int_data["cthetas"][i_layer](r_obs)
            calpha = calphaR#obsangle_func(theta_obs, phi_cell, alpha_obs)  # 0. for forward

            # doppler factor
            delta_D =  Gamma_obs * (1. - beta_obs * calpha) # (1. - beta_obs) / (1. - beta_obs * calpha)
            # frequency in the comoving frame
            freqprime = freq_z * delta_D
            # flux in the comoving frame
            fluxprime = self._get_eats_flux(freqprime, r_obs, i_layer)
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
        light_curve = self._loop_for_lightcurve(jet, alpha_obs, timegrid, freq_z)
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


















# class SimpleSphere:
#
#     def __init__(
#                 self,
#                 R,
#                 Gamma,
#                 tt,
#                 thickness,
#                 spectra
#         ):
#         self.R = R
#         self.Gamma = Gamma
#         self.tobs = tt
#         self.beta = np.sqrt(1 - 1 / np.power(Gamma, 2))
#         self.thickness = thickness
#
#         # self.volume = 4. / 3. * np.pi * np.power(R, 3)
#
#         self.spectra = []
#         for i in range(len(self.R)):
#             self.spectra.append( interpolate.interp1d(freq_to_integrate, spectra[i], kind="linear") )
#
#         #
#         # flux = (1+z) * 4/3 * np.pi * np.power(R, 3)
#
#     def lightcurve(self, alpha_obs, timegrid, freq, z, d_l):
#         # phi = 2*np.pi
#         # mu_s = np.cos(phi)
#         # delta_D = 1 / (self.Gamma * (1 - self.beta * mu_s))
#         # prefactor = np.power(delta_D, 4) / (4 * np.pi * np.power(self.R, 2))
#         #
#         #
#         # Phi = np.full_like(self.Gamma, 0)
#         # self.fluxes = np.zeros_like(self.tobs)
#         # for i in range(len(self.tobs)):
#         #     nuprime = (1 + z) * freq * self.Gamma[i] * (1 - self.beta[i] * np.cos(Phi[i]))
#         #     eats_fac =  np.power(self.Gamma[i] * (1 - self.beta[i] * np.cos(Phi[i])), -3) #* self.R[i] ** 2 * self.thickness[i] *
#         #     power = self.spectra[i](nuprime)
#         #     self.fluxes[i] = 4 * self.R[i] ** 2 * self.thickness[i] * self.Gamma[i] * power * (1 + z) / (2 * d_l ** 2)
#         # lightcruve = interpolate.interp1d(self.tobs, self.fluxes, kind="linear")(timegrid)
#         # return lightcruve
#
#         # Phi = np.full_like(self.Gamma, 0)
#         # self.fluxes = np.zeros_like(self.tobs)
#         # for i in range(len(self.tobs)):
#         #     nuprime = (1 + z) * freq * self.Gamma[i] * (1 - self.beta[i] * np.cos(Phi[i]))
#         #     power = self.spectra[i](nuprime)
#         #     self.fluxes[i] = 4 * np.pi * self.R[i] ** 2 * self.thickness[i] * self.Gamma[i] * power * (1 + z) / (4 * d_l ** 2)
#         # lightcruve = interpolate.interp1d(self.tobs, self.fluxes, kind="linear")(timegrid)
#         # return lightcruve
#
#         fluxes = np.zeros(self.tobs)
#         for i in range(len(self.tobs)):
#             calpha = 0.
#             delta_D = self.Gamma[i] * (1. - self.beta[i] * calpha)
#             nuprime = (1 + z) * freq * delta_D
#             fluxprime = self.spectra[i](nuprime)
#             fluxes[i] = 4 * np.pi * (1 + z) / (4 * d_l ** 2) * np.power(delta_D, -3) * fluxprime
#
#         return fluxes

''' -- [below] to be removed [below] --- '''

# class SphereSourceNoEATS:
#     def __init__(
#             self,
#             R,
#             Gamma,
#             tt,
#             thickness,
#             spectra
#     ):
#
#         self.R = R
#         self.Gamma = Gamma
#         self.tobs = tt
#         self.beta = np.sqrt(1 - 1 / np.power(Gamma, 2))
#         self.thickness = thickness
#
#         # self.volume = 4. / 3. * np.pi * np.power(R, 3)
#
#         self.spectra = []
#         for i in range(len(self.R)):
#             self.spectra.append( interpolate.interp1d(freq_to_integrate, spectra[i], kind="linear") )
#
#         #
#         # flux = (1+z) * 4/3 * np.pi * np.power(R, 3)
#
#     def lightcurve(self, alpha_obs, timegrid, freq, z, d_l):
#         # phi = 2*np.pi
#         # mu_s = np.cos(phi)
#         # delta_D = 1 / (self.Gamma * (1 - self.beta * mu_s))
#         # prefactor = np.power(delta_D, 4) / (4 * np.pi * np.power(self.R, 2))
#         #
#         #
#         # Phi = np.full_like(self.Gamma, 0)
#         # self.fluxes = np.zeros_like(self.tobs)
#         # for i in range(len(self.tobs)):
#         #     nuprime = (1 + z) * freq * self.Gamma[i] * (1 - self.beta[i] * np.cos(Phi[i]))
#         #     eats_fac =  np.power(self.Gamma[i] * (1 - self.beta[i] * np.cos(Phi[i])), -3) #* self.R[i] ** 2 * self.thickness[i] *
#         #     power = self.spectra[i](nuprime)
#         #     self.fluxes[i] = 4 * self.R[i] ** 2 * self.thickness[i] * self.Gamma[i] * power * (1 + z) / (2 * d_l ** 2)
#         # lightcruve = interpolate.interp1d(self.tobs, self.fluxes, kind="linear")(timegrid)
#         # return lightcruve
#
#         Phi = np.full_like(self.Gamma, 0)
#         self.fluxes = np.zeros_like(self.tobs)
#         for i in range(len(self.tobs)):
#             nuprime = (1 + z) * freq * self.Gamma[i] * (1 - self.beta[i] * np.cos(Phi[i]))
#             power = self.spectra[i](nuprime)
#             self.fluxes[i] = 4 * np.pi * self.R[i] ** 2 * self.thickness[i] * self.Gamma[i] * power * (1 + z) / (4 * d_l ** 2)
#         lightcruve = interpolate.interp1d(self.tobs, self.fluxes, kind="linear")(timegrid)
#         return lightcruve
#
#

def doppler_D_b(Gamma, beta, cosalpha):
    return 1 / ( (1. - beta) / (1. - beta * cosalpha) )

def doppler_D(Gamma, beta, cosalpha):
    return Gamma * (1 - beta * cosalpha)

class RingEATS:
    """
        For every EATS rim,
        precompute values of all needed arrays at the forefront of the rim
        at the 'idx1'
    """

    def __init__(
            self, tobsdyn, arrs, v_ns, z=None, freq=None
    ):
        """
        Set arrays that are needed to be intepolated for every ring
        :param tobsdyn:
        :param arrs:
        :param v_ns:
        """
        assert len(v_ns) == len(arrs)
        self.tobsdyn = tobsdyn
        self.arrs = arrs
        self.v_ns = v_ns
        self.z = z
        self.freq = freq
        # output
        self.front = {}
        self.edge = {}
        self.mid = {}

    @classmethod
    def from_obj(cls, dyn_obj):
        v_ns = ["R", "Gamma", "beta", "theta", "tburst", "thickness"]
        arrs = []
        for v_n in v_ns:
            arrs.append(dyn_obj.get(v_n))
        return cls(dyn_obj.get("tt"), arrs, v_ns, None, None)

    def evaulate_front(self, idx1, tobs):

        """
        For a given tobs and from idx 'idx1', intepolate all arrays 'front' value
        :param idx1:
        :param tobs:
        :return:
        """

        assert not self.z is None

        tobsdyn = self.tobsdyn
        connect = (np.log(tobs) - np.log(tobsdyn[idx1])) / (np.log(tobsdyn[idx1 + 1]) - np.log(tobsdyn[idx1]))
        # linear interpolation (of log values) of the arr[idx]
        front_vals = {}
        for v_n, arr in zip(self.v_ns, self.arrs):
            front_vals[v_n] = np.exp(np.log(arr[idx1]) + (np.log(arr[idx1 + 1]) - np.log(arr[idx1])) * connect)
            assert np.isfinite(front_vals[v_n])
            # front_vals[v_n] = arr[idx1]
        # additional values
        front_vals["Phi"] = 0  # Phi[-1] = 0.  ### Per definition
        #
        onePzFreq = (1 + self.z) * self.freq
        nuPrim_front = onePzFreq * doppler_D(front_vals["Gamma"], front_vals["beta"], np.cos(front_vals["Phi"]))
        front_vals["nuPrim"] = nuPrim_front
        #
        eats_fac_front = np.power(doppler_D(front_vals["Gamma"], front_vals["beta"], np.cos(front_vals["Phi"])), -3)
        front_vals["dop3"] = eats_fac_front
        # output
        self.front = front_vals

    def evaluate_edge(self, idx2, tobs, tobs_behind, tobs_before):

        assert not self.z is None

        # tobsdyn = self.tobsdyn
        edge_vals = {}
        # tobs_behind = last_index
        # tobs_before = last_index + 1
        #         self.edge_connector_log = (np.log(tobs) - np.log(tobs_behind)) / (np.log(tobs_before) - np.log(tobs_behind))
        connect = (np.log(tobs) - np.log(tobs_behind)) / (np.log(tobs_before) - np.log(tobs_behind))
        # linear interpolation (of log values) of the arr[idx]
        for v_n, arr in zip(self.v_ns, self.arrs):
            edge_vals[v_n] = np.exp(np.log(arr[idx2]) + (np.log(arr[idx2 + 1]) - np.log(arr[idx2])) * connect)
            if not np.isfinite(edge_vals[v_n]):
                raise ValueError("Value: {} at the ednge is not finite".format(v_n))
            # edge_vals[v_n] = arr[idx2]
        # additional values
        cosPhi_edge = cgs.c / edge_vals["R"] * (edge_vals["tburst"] - tobs / (1 + self.z))
        if (cosPhi_edge >= 1) or (cosPhi_edge <= -1):
            Phi_edge = np.pi / 2
        else:
            Phi_edge = np.arccos(cosPhi_edge)
        edge_vals["Phi"] = Phi_edge
        #
        onePzFreq = (1 + self.z) * self.freq
        nuPrim_edge = onePzFreq * doppler_D(edge_vals["Gamma"], edge_vals["beta"], np.cos(edge_vals["Phi"]))
        edge_vals["nuPrim"] = nuPrim_edge
        #
        eats_fac_edge = np.power(doppler_D(edge_vals["Gamma"], edge_vals["beta"], cosPhi_edge), -3)
        edge_vals["dop3"] = eats_fac_edge
        # ---
        self.edge = edge_vals

    def evaluate_middle(self, idxes, tobs):

        middle_arrs = {}
        for v_n, arr in zip(self.v_ns, self.arrs):
            middle_arrs[v_n] = arr[idxes]
        #
        onePzFreq = (1 + self.z) * self.freq
        cosPhi_mid = cgs.c / middle_arrs["R"] * (middle_arrs["tburst"] - tobs / (1 + self.z))
        middle_arrs["Phi"] = np.arccos(cosPhi_mid)
        #
        nuPrim_edge = onePzFreq * doppler_D(middle_arrs["Gamma"], middle_arrs["beta"], cosPhi_mid)
        middle_arrs["nuPrim"] = nuPrim_edge
        #
        eats_fac_midd = np.power(doppler_D(middle_arrs["Gamma"], middle_arrs["beta"], cosPhi_mid), -3)
        middle_arrs["dop3"] = eats_fac_midd
        #
        self.mid = middle_arrs

    def get_eats_fac(self):
        pass

    def get_nuprime(self):
        pass
#
class EATS_UniformSingleSource:

    def __init__(
            self,
            R,
            Gamma,
            beta,
            theta,
            thickness,
            tobs,
            tburst,
            spectra
    ):

        # import matplotlib.pyplot as plt
        # plt.loglog(tobs, Gamma)
        # plt.grid(1)
        # plt.show()

        # settings
        self.max_jet_theta = np.pi/2.

        self.thetadyn = theta
        self.tburstdyn = tburst
        self.Rdyn = R
        # self.tobsdyn = tobs# tt
        self.tobsdyn = (self.tburstdyn - self.Rdyn / cgs.c) *  (1 + 0.0099)
        self.Gamma = Gamma

        # self.spectra = comov_spectra
        # assert len(self.spectra) == len(self.Rdyn)
        self.spectra = interpolate.RegularGridInterpolator((
            R,
            freq_to_integrate),
            np.vstack((spectra))
        )

        self.eats = RingEATS(
            self.tobsdyn,
            [R, Gamma, beta, theta, thickness, self.tobsdyn, tburst],
            ["R", "Gamma", "beta", "theta", "thickness", "tobs", "tburst"]
        )

    @classmethod
    def from_model_objs(cls, o_struct, o_shell, cthetas):
        if hasattr(o_shell, '__len__'):
            if len(o_shell) > 1:
                raise ValueError("This EATS method is only valid for non-structured, non-layers jets")
            o_shell = o_shell[0]
        return cls(
            R=o_shell.dyn.get("R")[1:],
            Gamma=o_shell.dyn.get("Gamma")[1:],
            beta=o_shell.dyn.get("beta")[1:],
            theta=o_shell.dyn.get("theta")[1:],
            thickness=o_shell.dyn.get("thickness")[1:],
            tobs=o_shell.dyn.get("tt")[1:],
            tburst=o_shell.dyn.get("tburst")[1:],
            spectra=o_shell.spectrum
        )


    @classmethod
    def from_obj(cls, dyn_obj, spectra):
        v_ns = ["R", "Gamma", "beta", "theta", "thickness", "tt", "tburst"]
        arrs = []
        for v_n in v_ns:
            arrs.append(dyn_obj.get(v_n)[1:])
        assert len(arrs[0]) == len(spectra)
        return cls(*arrs, spectra=spectra)

    def compute_eats_flux(self, tobs, freqobs, alphaobs, z, d_l):

        self.eats.freq = freqobs
        self.eats.z = z

        total_angle = self.thetadyn + alphaobs
        total_angle[np.where(total_angle > self.max_jet_theta)] = self.max_jet_theta
        offaxistobs = (1 + z) * (self.tburstdyn - self.Rdyn * np.cos(total_angle) / cgs.c)

        # Finding the index just behind the point at the very rim of the jet
        idx2 = np.argmin(np.abs(offaxistobs - tobs))  # array
        if offaxistobs[idx2] > tobs:
            idx2 -= 1
        if idx2 < 1: raise ValueError("tobs goes below the range of dynamic tobs. Set lower Rstart")

        # Index to the point right behind the foremost point on the EATS
        idx1 = np.argmin(np.abs(self.tobsdyn - tobs))
        if self.tobsdyn[idx1] > tobs:
            if idx1 != 0: idx1 -= 1

        # An array with obs times of all points on the EATS. Then we integrate using trapzoidal rule
        idx21 = np.arange(idx2 + 1, idx1 + 1)

        tobs2 = offaxistobs[idx2]
        tobs1 = offaxistobs[idx1 + 1]

        Phi_f = cgs.c / self.Rdyn[idx21] * (self.tburstdyn[idx21] - tobs / (1 + z))
        idx21 = idx21[np.where(Phi_f < 1)]  # Phi_f > 1 :: Jet has expanded to 90 degrees
       #  print("\t\tt:{} idx [{},{}] ({})".format(tobs, idx2 + 1, idx1 + 1, len(idx21)))

        n_rings = len(idx21) + 2

        self.eats.evaluate_edge(idx2, tobs, offaxistobs[idx2], offaxistobs[idx2 + 1])
        self.eats.evaulate_front(idx1, tobs)
        self.eats.evaluate_middle(idx21, tobs)

        # Setting array containing angle from LoS to EATS rings
        Phi = np.full(n_rings, 0.)
        Phi[1:-1] = self.eats.mid["Phi"]# np.arccos(cgs.c / self.eats.mid["R"] * (self.eats.mid["tburst"] - tobs / (1 + z)))
        Phi[0] = self.eats.edge["Phi"]  # InterWeights.Phi_edge
        Phi[-1] = 0.  ### Per definition # front

        # phi ring -- full or partial
        phiI = np.ones(n_rings) * 2 * np.pi
        if alphaobs != 0:  # Off-axis
            # partial ring indexes
            idxpr = np.where(
                (Phi[1:-1] > self.eats.mid["theta"] - alphaobs) & (Phi[1:-1] < self.eats.mid["theta"] + alphaobs))
            # partial Rings Ind_edge Rings crossing the rim.
            isedge = (Phi[0] > self.eats.edge["theta"] - alphaobs) & (Phi[0] < self.eats.edge["theta"] + alphaobs)
            offAxisFoV = (self.eats.mid["theta"][idxpr] ** 2 - alphaobs ** 2 - Phi[1:-1][idxpr] ** 2) / \
                         (2 * alphaobs * Phi[1:-1][idxpr])
            if isedge:
                offAxisFoV_edge = (self.eats.edge["theta"] ** 2 - alphaobs ** 2 - Phi[0] ** 2) / (2 * alphaobs * Phi[0])
                phiI[0] = 2 * np.pi - 2 * np.arccos(offAxisFoV_edge)
            ### phiI at the centre is always 2*pi if theta is larger than alpha, and vice verse if alpha is larger than theta (orphan burst)
            if self.eats.edge["theta"] < alphaobs:
                print("\t\tt:{} Orphan".format(tobs))
                phiI[-1] = 0.
            offAxisFoV[np.where(offAxisFoV < -1)] = -1.
            phiI[idxpr] = 2 * np.pi - 2 * np.arccos(offAxisFoV)
        if np.isnan(Phi[-1]):
            raise ValueError("Phi[-1] = nan ")

        # evaluate flux from comoving SED

        # nuPrim = np.zeros(n_rings)
        # nuPrim[1:-1] = (1 + z) * freqobs * doppler_D_b(self.eats.mid["Gamma"], self.eats.mid["beta"], np.cos(Phi[1:-1]))
        # nuPrim[0] = self.eats.edge["nuPrim"]  # nuPrim_edge
        # nuPrim[-1] = self.eats.front["nuPrim"]  # nuPrim_front
        # or
        # nuPrim[1:-1] = freqobs / ((1. - o_eats.mid["beta"]) / (1. - o_eats.mid["beta"] * np.cos(Phi[1:-1])))
        # nuPrim[0] = freqobs / ((1. - o_eats.edge["beta"]) / (1. - o_eats.edge["beta"] * np.cos(Phi[0])))  # nuPrim_edge
        # nuPrim[-1] = freqobs / ((1. - o_eats.front["beta"]) / (1. - o_eats.front["beta"] * np.cos(Phi[-1])))

        # eats_fac = np.zeros(n_rings)
        # eats_fac[1:-1] = self.eats.mid["R"] ** 2 * self.eats.mid["thickness"] *  \
        #                  np.power(self.eats.mid["Gamma"] * (1 - self.eats.mid["beta"] * np.cos(Phi[1:-1])), -3)
        # eats_fac[0] = self.eats.edge["R"] ** 2 * self.eats.edge["thickness"] *  \
        #               np.power(self.eats.edge["Gamma"] * (1 - self.eats.edge["beta"] * np.cos(Phi[0])), -3)
        # eats_fac[-1] = self.eats.front["R"] ** 2 * self.eats.front["thickness"] *  \
        #                np.power(self.eats.front["Gamma"] * (1 - self.eats.front["beta"] * np.cos(Phi[-1])), -3)

        # eats_fac[1:-1] = np.power(self.eats.mid["Gamma"] * (1 - self.eats.mid["beta"] * np.cos(Phi[1:-1])), -3)
        # eats_fac[0] = np.power(self.eats.edge["Gamma"] * (1 - self.eats.edge["beta"] * np.cos(Phi[0])), -3)
        # eats_fac[-1] = np.power(self.eats.front["Gamma"] * (1 - self.eats.front["beta"] * np.cos(Phi[-1])), -3)

        # eats_fac[1:-1] = np.power(doppler_D(self.eats.mid["Gamma"], self.eats.mid["beta"], np.cos(Phi[1:-1])), -3)
        # eats_fac[0] = np.power(doppler_D(self.eats.edge["Gamma"], self.eats.edge["beta"], np.cos(Phi[0])), -3)
        # eats_fac[-1] = np.power(doppler_D(self.eats.front["Gamma"], self.eats.front["beta"], np.cos(Phi[-1])), -3)

        # PprimTemp = np.zeros(n_rings)
        # PprimTemp[1:-1] = eats_fac[1:-1] * self.spectra((self.eats.mid["R"], nuPrim[1:-1]))
        # PprimTemp[0] = eats_fac[0] * self.spectra((self.eats.edge["R"], nuPrim[0]))
        # PprimTemp[-1] = eats_fac[-1] * self.spectra((self.eats.front["R"], nuPrim[-1]))


        PprimTemp = np.zeros(n_rings)
        PprimTemp[1:-1] = self.eats.mid["dop3"] * self.spectra((self.eats.mid["R"], self.eats.mid["nuPrim"]))
        PprimTemp[0] = self.eats.edge["dop3"] * self.spectra((self.eats.edge["R"], self.eats.edge["nuPrim"]))
        PprimTemp[-1] = self.eats.front["dop3"] * self.spectra((self.eats.front["R"], self.eats.front["nuPrim"]))

        # PprimTemp[0] = eats_fac[0] * self.spectra[idx2](nuPrim[0])
        # PprimTemp[-1] = eats_fac[-1] * self.spectra[idx1](nuPrim[-1])
        # _prime = self.spectra((self.eats.mid["R"], nuPrim[1:-1]))


        # for i, ii in enumerate(idx21-1):
        #     PprimTemp[i+1] = eats_fac[i+1] * self.spectra[ii](nuPrim[i+1])


        # PprimTemp[1:-1] = eats_fac_mid * o_rad.SED_joh06(nuPrim[1:-1], o_eats.mid["nu_m"], o_eats.mid["nu_c"], p,
        #                                                  o_eats.mid["PmaxF"], o_eats.mid["PmaxS"])
        # PprimTemp[0] = eats_fac_edge * o_rad.SED_joh06(nuPrim[0], o_eats.edge["nu_m"], o_eats.edge["nu_c"], p,
        #                                                o_eats.edge["PmaxF"], o_eats.edge["PmaxS"])
        # PprimTemp[-1] = eats_fac_front * o_rad.SED_joh06(nuPrim[-1], o_eats.front["nu_m"], o_eats.front["nu_c"], p,
        #                                                  o_eats.front["PmaxF"], o_eats.front["PmaxS"])

        distance_factor = (1 + z) / (4 * np.pi * d_l ** 2)
        flux_fs = np.trapz(PprimTemp * phiI, np.cos(Phi)) * distance_factor

        return flux_fs

    def lightcurve(self, alpha_obs, timegrid, freq, z, d_l, jet="principle"):

        fluxes = np.zeros_like(timegrid)
        for i in range(len(fluxes)):
            fluxes[i] = self.compute_eats_flux(timegrid[i], freq, alpha_obs, z, d_l)
        return fluxes

    def uniformspherelightcurve(self, timegrid, freq, z, d_l):

        tobs = self.tobsdyn
        Gamma = self.Gamma
        beta = np.sqrt(1. - np.power(Gamma, -2))
        spectrum = self.spectra
        fluxes = np.zeros_like(tobs)
        R = self.Rdyn
        for i in range(len(tobs)):
            calpha = 0.
            delta_D = Gamma[i] * (1. - beta[i] * calpha)
            nuprime = (1 + z) * freq * delta_D
            fluxprime = spectrum((R[i], nuprime))
            fluxes[i] = (2) * np.pi * fluxprime * np.power(delta_D, 1)
        lightcruve = interpolate.interp1d(tobs, fluxes, kind="linear")(timegrid)
        return lightcruve * (1 + z) / (4 * d_l ** 2)

# class StructuredSourceEATS_old:
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
#
#         self.ncells = ncells
#         self.layer = layers
#         self.cphis = cphis
#         self.nlayers = nlayers
#
#         self.tts = tts
#         self.Rs = Rs
#         self.cthetas = cthetas
#         x = cthetas[0]
#
#         self.int_data = {
#             "Gamma": [], "beta": [], "cthetas": [], "TTs": [], "pprime": [], "thickness": []
#         }
#
#         for ii in range(self.nlayers):
#             if Gammas[ii][0] <= 1. + 1e-5:
#                 raise ValueError("layer:{} starts with Gamma:{} too low!".format(ii, Gammas[ii][0]))
#             # print(ii)
#             self.int_data["Gamma"].append(interpolate.interp1d(Rs[ii], Gammas[ii], kind="linear"))
#             self.int_data["beta"].append(interpolate.interp1d(Rs[ii], betas[ii], kind="linear"))
#             self.int_data["cthetas"].append(interpolate.interp1d(Rs[ii], cthetas[ii], kind="linear"))
#             self.int_data["thickness"].append(interpolate.interp1d(Rs[ii], thickness[ii], kind="linear"))
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
#             R, Gamma, tt, thickness,
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
#
#     def _get_eats_power(self, freqprime, radii, ilayer):
#         rs = radii
#         freqs = np.full_like(radii, freqprime)
#         power = self.int_data["pprime"][ilayer]((rs, freqs))
#         return power#np.power(10., power)
#
#     def lightcurve(self, alpha_obs, timegrid, freq, z, d_l, jet='principle'):
#
#         light_curve = np.zeros([len(timegrid), self.nlayers])
#
#         for ii in tqdm(range(self.ncells)):
#             layer = self.layer[ii] - 1
#             # phi coordinate point of the cell
#             phi_cell = self.cphis[ii]
#             # theta coordiantes of the cell
#             theta_cellR = self.cthetas[layer]#[:, layer]  # Value! -> ARRAY
#             calphaR = obsangle(theta_cellR, phi_cell, alpha_obs)  # 0. for forward jet
#
#             # observer times during which radiation from 'phi, theta[]' elements arrive
#             ttobs = self.tts[layer] + self.Rs[layer] / cgs.c * (1. - calphaR)  # arr
#             # part of the 'emssion' observer time that falls within observation window
#             aval_times = (np.min(ttobs) < timegrid) & (timegrid <= np.max(ttobs))
#
#             # Radii values of elements emission from which is visible at 'ttobs'
#             Rint = interpolate.interp1d(ttobs, self.Rs[layer])
#             r_obs = Rint(timegrid[aval_times])  #
#
#
#             # Quantities visible at ttobs when rad. emitted at r_obs of the jet reaches observer
#             theta_obs = self.int_data["cthetas"][layer](r_obs)
#             calpha = obsangle(theta_obs, phi_cell, alpha_obs)  # 0. for forward
#             Gamma_obs = self.int_data["Gamma"][layer](r_obs)
#             beta_obs = self.int_data["beta"][layer](r_obs)
#             thickness = self.int_data["thickness"][layer](r_obs)
#
#             eats_facs = np.power(Gamma_obs * (1. - beta_obs * calpha), -3.)
#             dopfacs = dopplerFactor(calpha, beta_obs)
#             freqprime = (1. + z) * freq * Gamma_obs * (1 - beta_obs * calpha)
#
#             powers = self._get_eats_power(freqprime, r_obs, layer)
#             eats_power = r_obs ** 2 * thickness * eats_facs * powers
#             eats_flux = eats_power * (1. + z) / (2. * np.power(d_l, 2))
#
#             light_curve[aval_times, layer] = light_curve[aval_times, layer] + eats_flux
#
#         return np.sum(light_curve, axis=1)

# class TestEATS():
#
#     def __init__(
#             self,
#             Rs,
#             Gammas,
#             betas,
#             tts,
#             cthetas,
#             thickness,
#             spectra
#     ):
#         assert (200 > len(Rs) > 0)
#         self.nlayers = len(Rs)
#
#         self.int_data = {
#             "Gamma": [], "beta": [], "cthetas": [], "TTs": [], "pprime": [], "thickness": []
#         }
#         self.cthetas = cthetas
#         self.tts = tts
#         self.Rs = Rs
#
#         for ii in range(self.nlayers):
#             if Gammas[ii][0] <= 1. + 1e-5:
#                 raise ValueError("layer:{} starts with Gamma:{} too low!".format(ii, Gammas[ii][0]))
#             # print(ii)
#             self.int_data["Gamma"].append(interpolate.interp1d(Rs[ii], Gammas[ii], kind="linear"))
#             self.int_data["beta"].append(interpolate.interp1d(Rs[ii], betas[ii], kind="linear"))
#             self.int_data["cthetas"].append(interpolate.interp1d(Rs[ii], cthetas[ii], kind="linear"))
#             self.int_data["thickness"].append(interpolate.interp1d(Rs[ii], thickness[ii], kind="linear"))
#             self.int_data["pprime"].append(interpolate.RegularGridInterpolator((
#                 Rs[ii],
#                 freq_to_integrate),
#                 np.vstack((spectra[ii])))
#             )
#
#     def _cell_power(self, theta, phi, t, alpha, freq, obsangle_func):
#         pass
#
#
#     def _integrate_eats(self, t, alpha, freq, obsangle=obsangle):
#
#         grid_phi = np.linspace(-np.pi, np.pi, 50)
#         grid_theta = np.linspace(-np.pi / 2, np.pi / 2, 50)
#
#         integrand = lambda theta, phi: self._cell_power(theta, phi, t, alpha, freq, obsangle)
#
#         return np.trapz(
#             [np.trapz(
#                 [integrand(theta, phi) for phi in grid_phi], grid_phi)
#                 for theta in grid_theta], grid_theta
#         )
#
#     def _compute_power_from_layer(self, layer, alpha_obs, timegrid, freq_z, obsangle_func):
#
#         nR = len(self.cthetas)
#         nPhi = 2 * layer + 1
#
#         # alpha_obs = 0.
#         # import numpy as np
#         # nR, nPhi = 100, 100
#         grid_phi = np.linspace(-np.pi, np.pi, nPhi)
#         theta_cellR = self.cthetas[layer]  # [:, layer]  # Value! -> ARRAY
#         calphaR = np.vstack((obsangle(theta_cellR, [phi for phi in grid_phi], alpha_obs)))
#
#         # calphaR = np.zeros((nR, nPhi))
#         # calphaR[:, :] = obsangle(theta_cellR, grid_phi, alpha_obs)
#
#         # calphaR = obsangle(theta_cellR, phi_cell, alpha_obs)  # 0. for forward jet
#
#
#
#         # observer times during which radiation from 'phi, theta[]' elements arrive
#         #    ttobs = self.tts[layer] + self.Rs[layer] / cgs.c * (1. - calphaR)  # arr
#         ttobs = np.vstack((self.tts[layer] + self.Rs[layer] / cgs.c * (1. - calphaR[:, [iphi for iphi in range(nPhi)]])))  # arr
#         # part of the 'emssion' observer time that falls within observation window
#         aval_times = (np.min(ttobs) < timegrid) & (timegrid <= np.max(ttobs))
#
#         # Radii values of elements emission from which is visible at 'ttobs'
#         Rint = interpolate.interp1d(ttobs, self.Rs[layer])
#         r_obs = Rint(timegrid[aval_times])  #
#
#         # Quantities visible at ttobs when rad. emitted at r_obs of the jet reaches observer
#         theta_obs = self.int_data["cthetas"][layer](r_obs)
#         calpha = obsangle(theta_obs, phi_cell, alpha_obs)  # 0. for forward
#         Gamma_obs = self.int_data["Gamma"][layer](r_obs)
#         beta_obs = self.int_data["beta"][layer](r_obs)
#         thickness = self.int_data["thickness"][layer](r_obs)
#
#         delta_D = Gamma_obs * (1. - beta_obs * calpha)
#         freqprime = (1. + z) * delta_D * freq
#
#         powers = self._get_eats_power(freqprime, r_obs, layer)
#         eats_power = r_obs ** 2 * thickness * Gamma_obs * np.power(delta_D, -3) * powers
#
#         light_curve[aval_times, layer] = light_curve[aval_times, layer] + eats_power
#
#
#
#     def lightcurve(self, alpha_obs, timegrid, freq, z, d_l, jet='principle'):
#         fluxes = np.zeros((len(timegrid), self.nlayers))
#         freq_z = (1. + z) * freq
#         if jet=='principle':obsangle_func=obsangle
#         else: raise NameError("not implemented")
#         for ii in range(self.nlayers):
#             fluxes[:, ii] = self._compute_power_from_layer(
#                 ii,
#                 alpha_obs,
#                 timegrid,
#                 freq_z,
#                 obsangle_func=obsangle_func
#             )
#         fluxes *= (1. + z) / (2. * np.power(d_l, 2))
#         return np.sum(fluxes, axis=1)
