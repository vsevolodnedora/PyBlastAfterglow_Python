"""

"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate

from PyBlastAfterglow.dynamics import Driver_Peer_FS, Driver_Nava_FS, Driver_Nava_FSRS, rho_dlnrho1dR, EqOpts
from PyBlastAfterglow.electrons import Electron_BPL, Electron_BPL_Accurate
from PyBlastAfterglow.synchrotron import Synchrotron_Joh06, Synchrotron_WSPN99, Synchrotron_DM06, freq_to_integrate
from PyBlastAfterglow.structure import Structure_Angular #, Structure_Uniform
from PyBlastAfterglow.eats import EATS_StructuredLayersSource #, EATS_UniformSingleSource
from PyBlastAfterglow.shells import Shell_FS, Shell_FS_Electrons_Synchrotron
from PyBlastAfterglow.uutils import cgs

class StructuredJet:

    def __init__(self,
                 driver=(None, {}),
                 electrons=(None, {}),
                 synchrotron=(None, {}),
                 eats=(None, {}),
                 **kwargs):

        # set parameters
        if driver[0] == "Peer": o_driver, driver_kwargs = (Driver_Peer_FS, driver[1])
        elif driver[0] == "Nava": o_driver, driver_kwargs = (Driver_Nava_FS, driver[1])
        elif driver[0] == "NavaFSRS": o_driver, driver_kwargs = (Driver_Nava_FSRS, driver[1])
        else: raise NameError("Driver Name is not recognized: {}".format(driver[0]))

        if driver_kwargs["eq_dthetadr"] == "dthetadr_None": driver_kwargs["eq_dthetadr"] = EqOpts.dthetadr_None
        elif driver_kwargs["eq_dthetadr"] == "dthetadr_Adi": driver_kwargs["eq_dthetadr"] = EqOpts.dthetadr_Adi
        elif driver_kwargs["eq_dthetadr"] == "dthetadr_Adi_Rd": driver_kwargs["eq_dthetadr"] = EqOpts.dthetadr_Adi_Rd
        elif driver_kwargs["eq_dthetadr"] == "dthetadr_AA": driver_kwargs["eq_dthetadr"] = EqOpts.dthetadr_AA
        elif driver_kwargs["eq_dthetadr"] == "dthetadr_AA_Rd": driver_kwargs["eq_dthetadr"] = EqOpts.dthetadr_AA_Rd
        else: raise NameError("Driver kwarg 'eq_dthetadr' is not correct: {}".format(driver_kwargs["eq_dthetadr"]))

        if driver_kwargs["eq_gammaAdi"] == "gamma_adi_peer": driver_kwargs["eq_gammaAdi"] = EqOpts.gamma_adi_peer
        elif driver_kwargs["eq_gammaAdi"] == "gamma_adi_nava": driver_kwargs["eq_gammaAdi"] = EqOpts.gamma_adi_nava
        else: raise NameError("Driver kwarg 'eq_gammaAdi' is not correct: {}".format(driver_kwargs["eq_gammaAdi"]))

        if driver_kwargs["eq_rhoprime"] == "rho2_rel": driver_kwargs["eq_rhoprime"] = EqOpts.rho2_rel
        elif driver_kwargs["eq_rhoprime"] == "rho2_transrel": driver_kwargs["eq_rhoprime"] = EqOpts.rho2_transrel
        else: raise NameError("Driver kwarg 'eq_rhoprime' is not correct: {}".format(driver_kwargs["eq_gammaAdi"]))

        if electrons[0] == "Electron_BPL": o_electrons, electrons_kwargs = (Electron_BPL, electrons[1])
        elif electrons[0] == "Electron_BPL_Accurate": o_electrons, electrons_kwargs = (Electron_BPL_Accurate, electrons[1])
        else: raise NameError("Electrons name is not recognized: {}".format(electrons[0]))

        if synchrotron[0] == "Synchrotron_Joh06": o_synchrotron, synchrotron_kwargs = (Synchrotron_Joh06, synchrotron[1])
        elif synchrotron[0] == "Synchrotron_WSPN99": o_synchrotron, synchrotron_kwargs = (Synchrotron_WSPN99, synchrotron[1])
        elif synchrotron[0] == "Synchrotron_DM06": o_synchrotron, synchrotron_kwargs = (Synchrotron_DM06, synchrotron[1])
        else: raise NameError("Syncrotron Name is not recognized: {}".format(synchrotron[0]))

        if eats[0] == "EATS_StructuredLayersSource":
            o_eats, eats_kwargs = (EATS_StructuredLayersSource, eats[1])
        else:
            raise NameError("EATS Name is not recognized: {}".format(eats[0]))


        # structure object with angular distribution of jet parameters
        self.struct = kwargs["struct"]

        # create radial grid for dynamical evolution
        r_pars = kwargs["r_pars"]
        if not (len(r_pars) == 3):
            raise IOError("Wrong parameter 'r_pars' should be a tuple with 3 parameters:"
                          "(Rstart, Rstop, nR) \n"
                          "For example a common setup (1.e8, 1.e22, 500)")
        self.RRs = np.logspace(r_pars[0], r_pars[1], r_pars[2])

        self.dens_pars = kwargs["dens_pars"]
        if not (len(self.dens_pars) == 5):
            raise IOError("Wrong parameter 'dens_pars' should be a tuple with 5 parameters: "
                          "(n_ism, A0, s, R_EJ, R_ISM) \n"
                          "For uniform profile set only n_ism, other parameters are None, e.g.,"
                          "(1e-3, None, None, None, None)" )

        self.shells = []
        self.cthetas = []

        # extract methods and parameters
        # o_driver, driver_kwargs = driver[0], driver[-1]
        # o_electrons, electrons_kwargs = electrons[0], electrons[-1]
        # o_synchrotron, synchrotron_kwargs = synchrotron[0], synchrotron[-1]

        assert not (o_driver is None) # without a driver, shell cannot be evoloved
        if not "ncells" in driver_kwargs.keys(): driver_kwargs["ncells"] = self.struct.ncells
        else: driver_kwargs["ncells"] += self.struct.ncells # used for scaling the swept-up mass

        # evolve every layer/shell
        shell = None

        # print("Computing model '{}' layers ({})".format(self.struct.structure, self.struct.nlayers))

        for i_layer in range(self.struct.nlayers):

            if (self.struct.structure == "uniform") and not (shell is None):
                # all shells are identical in uniform model -- no need to compute them again
                self.shells.append(shell)
                continue

            shell = self.layer_evolution(
                i_layer,
                o_driver,
                driver_kwargs,
                o_electrons,
                electrons_kwargs,
                o_synchrotron,
                synchrotron_kwargs,
                print_text=False
            )

            self.shells.append(shell)

        # correct shells 'thetas' as they are a part of a structured jet
        self.cthetas = [self.struct.cthetas0[ii] +
                        0.5 * (2 * self.shells[ii].dyn.get("theta")[1:] -
                               2 * self.struct.theta0j) for ii
                        in range(self.struct.nlayers)]
        '''
        def get_thetas(self, joAngle):

            fac = arange(0,self.nlayers+1)/float(self.nlayers)        # Numerical factor for use during execution
            thetas  = 2.*arcsin(fac*sin(joAngle/4.))         # Calculate the propagation angle with respect to jet axis
            cthetas = 0.5*(thetas[1:]+thetas[:-1])

            return thetas, cthetas
        '''
        # thetas0, cthetas0 = self.get_thetas(self.initJoAngle)
        # cthetas[ii,:] = cthetas0[:] + 0.5*(self.joAngle[ii]-self.initJoAngle)
        # self.cthetas[:,ii] = self.cthetas0[ii] + 0.5*(self.joAngles[:,ii]-self.initJoAngle)

        self.eats = self.integrate_eats(o_eats, eats_kwargs)

        # print("Done computing model.")

    @classmethod
    def from_analytic_pars(cls,
                           nlayers=100,
                           EEc0=1.e52,
                           Gamc0=100.,
                           Mc0=None,
                           theta0j=np.pi / 2.,
                           thetaCo=np.pi / 2.,
                           kk=1.,
                           structure='uniform',
                           r_pars=(8., 22., 1000),
                           dens_pars=(1e-1, None, None, None, None),
                           driver=(None, {}),
                           electrons=(None, {}),
                           synchrotron=(None, {}),
                           eats=(None, {})
                           ):
        # initialize angular structure
        struct = Structure_Angular.from_analytic_pars(
            nlayers=nlayers,
            EEc0=EEc0, Gamc0=Gamc0, Mc0=Mc0,
            theta0j=theta0j, theta0c=thetaCo,
            kk=kk,
            structure=structure)

        pars = locals()
        for key in ["EEc0", "Gamc0", "theta0j", "thetaCo", "kk", "structure", "Mc0"]:
            pars.pop(key)
        pars["struct"] = struct
        return cls(**pars)

    @classmethod
    def from_analytic_pars_ejecta(cls,
                                  nlayers=100,
                                  Ek=1e49,
                                  beta=0.8,
                                  mass=1e23,
                                  theta0=np.pi/2.,
                                  structure='uniform',
                                  r_pars=(8., 22., 1000),
                                  dens_pars=(1e-1, None, None, None, None),
                                  driver=(None, {}),
                                  electrons=(None, {}),
                                  synchrotron=(None, {}),
                                  eats=(None, {})
                                  ):
        # initialize angular structure
        struct = Structure_Angular.from_analytic_pars_ejecta(
            nlayers=nlayers,
            Ek=Ek, beta=beta, mass=mass,
            theta0=theta0,
            structure=structure)

        pars = locals()
        for key in ["Ek", "beta", "mass", "theta0", "structure"]:
            pars.pop(key)
        pars["struct"] = struct
        return cls(**pars)
    @classmethod
    def from_profile(cls,
                     nlayers,
                     dist_cthetas,
                     dist_EEs,
                     dist_Gam0s,
                     dist_Beta0s,
                     dist_MM0s,
                     r_pars=(8., 22., 1000),
                     dens_pars=(1e-1, None, None, None, None),
                     driver=(None, {}),
                     electrons=(None, {}),
                     synchrotron=(None, {}),
                     eats=(None, {})
                     ):

        # initialize angular structure
        struct = Structure_Angular.from_profile(
            nlayers=nlayers,
            dist_cthetas=dist_cthetas,
            dist_EEs=dist_EEs,
            dist_Gam0s=dist_Gam0s,
            dist_Beta0s=dist_Beta0s,
            dist_MM0s=dist_MM0s
        )

        pars = locals()
        for key in ["dist_cthetas", "dist_EEs", "dist_Gam0s", "dist_Beta0s", "dist_MM0s"]:
            pars.pop(key)
        pars["struct"] = struct
        return cls(**pars)


    def layer_evolution(self,
                        i_layer,
                        o_driver,
                        driver_kwargs,
                        o_electrons,
                        electrons_kwargs,
                        o_synchrotron,
                        synchrotron_kwargs,
                        print_text=True):

        if print_text:
            print("\tlayer: {}/{}  [E0: {:.2e}  M0: {:2e}  G0: {:.3f}  theta0: {:.2f}]"
                  .format(i_layer, self.struct.nlayers, self.struct.cell_EEs[i_layer],
                          self.struct.cell_MM0s[i_layer], self.struct.cell_Gam0s[i_layer],
                          self.struct.theta0j / (np.pi / 180.)))

        # shell without electrons or radiation -- only dynamics
        if (o_electrons is None) and (o_synchrotron is None):
            shell = Shell_FS(
                Gamma0=self.struct.cell_Gam0s[i_layer],
                E0=self.struct.cell_EEs[i_layer],
                M0=self.struct.cell_MM0s[i_layer],
                thetaE=0.,
                theta0=self.struct.theta0j,
                r_grid=self.RRs,
                ncells=self.struct.ncells,
                dens_pars=self.dens_pars,
                driver=o_driver,
                driver_kwargs=driver_kwargs,
            )

        # shell with electrons but no radiation
        elif (o_electrons is None) and (not o_synchrotron is None):
            raise NameError("Not implemented")

        # shell with dynamics, electrons and radaition
        else:
            shell = Shell_FS_Electrons_Synchrotron(
                Gamma0=self.struct.cell_Gam0s[i_layer],
                E0=self.struct.cell_EEs[i_layer],
                M0=self.struct.cell_MM0s[i_layer],
                thetaE=0.,
                theta0=self.struct.theta0j,
                r_grid=self.RRs,
                ncells=self.struct.ncells,
                dens_pars=self.dens_pars,
                driver=o_driver,
                driver_kwargs=driver_kwargs,
                electrons=o_electrons,
                electron_kwargs=electrons_kwargs,
                synchrotron=o_synchrotron,
                synchrotron_kwargs=synchrotron_kwargs
            )

        return shell

    def integrate_eats(self, o_eats, eats_kwargs):

        # integrate over equal arrival time surfaces
        if not o_eats is None:
            eats = o_eats.from_model_objs(self.struct, self.shells, self.cthetas)
            # eats = o_eats(
            #     nlayers=self.struct.nlayers,
            #     ncells=self.struct.ncells,
            #     layers=self.struct.layer,
            #     cphis=self.struct.cphis,
            #     cthetas=self.cthetas,
            #     Rs=[self.shells[i].dyn.get("R")[1:] for i in range(self.struct.nlayers)],
            #     Gammas=[self.shells[i].dyn.get("Gamma")[1:] for i in range(self.struct.nlayers)],
            #     betas=[self.shells[i].dyn.get("beta")[1:] for i in range(self.struct.nlayers)],
            #     tts=[self.shells[i].dyn.get("tt")[1:] for i in range(self.struct.nlayers)],
            #     thickness=[self.shells[i].dyn.get("thickness")[1:] for i in range(self.struct.nlayers)],
            #     spectra=[self.shells[i].spectrum for i in range(self.struct.nlayers)],
            # )
        else:
            eats = None

        return eats

# class UniformJet:
#
#     def __init__(self,
#                  driver=(None, {}),
#                  electrons=(None, {}),
#                  synchrotron=(None, {}),
#                  eats=(None, {}),
#                  **kwargs):
#         # structure object with angular distribution of jet parameters
#         self.struct = kwargs["struct"]
#         # create radial grid for dynamical evolution
#         r_pars = kwargs["r_pars"]
#         self.RRs = np.logspace(r_pars[0], r_pars[1], r_pars[2])
#         self.dens_pars = kwargs["dens_pars"]
#
#         self.shells = []
#         self.cthetas = []
#
#         # extract methods and parameters
#         o_driver, driver_kwargs = driver[0], driver[-1]
#         o_electrons, electrons_kwargs = electrons[0], electrons[-1]
#         o_synchrotron, synchrotron_kwargs = synchrotron[0], synchrotron[-1]
#
#         assert not (o_driver is None)  # without a driver, shell cannot be evoloved
#
#         # evolve every layer/shell
#         shell = None
#         for i_shell in range(self.struct.nshells):
#
#             if (self.struct.structure == "uniform") and not (shell is None):
#                 # all shells are identical in uniform model -- no need to compute them again
#                 self.shells.append(shell)
#                 continue
#
#             shell = self.layer_evolution(
#                 i_layer,
#                 o_driver,
#                 driver_kwargs,
#                 o_electrons,
#                 electrons_kwargs,
#                 o_synchrotron,
#                 synchrotron_kwargs,
#                 print_text=True
#             )
#
#             self.shells.append(shell)
#
#         # correct shells 'thetas' as they are a part of a structured jet
#         self.cthetas = [self.struct.cthetas0[ii] +
#                         0.5 * (2 * self.shells[ii].dyn.get("theta")[1:] -
#                                2 * self.struct.theta0j) for ii
#                         in range(self.struct.nlayers)]
#         '''
#         def get_thetas(self, joAngle):
#
#             fac = arange(0,self.nlayers+1)/float(self.nlayers)        # Numerical factor for use during execution
#             thetas  = 2.*arcsin(fac*sin(joAngle/4.))         # Calculate the propagation angle with respect to jet axis
#             cthetas = 0.5*(thetas[1:]+thetas[:-1])
#
#             return thetas, cthetas
#         '''
#         # thetas0, cthetas0 = self.get_thetas(self.initJoAngle)
#         # cthetas[ii,:] = cthetas0[:] + 0.5*(self.joAngle[ii]-self.initJoAngle)
#         # self.cthetas[:,ii] = self.cthetas0[ii] + 0.5*(self.joAngles[:,ii]-self.initJoAngle)
#
#         self.eats = self.integrate_eats(eats)
#
#
#
#     @classmethod
#     def from_analytic_pars(cls):
#         return cls()
#
#     @classmethod
#     def from_profile(cls,
#                      nshells,
#                      dist_EEs,
#                      dist_Gam0s,
#                      dist_Beta0s,
#                      dist_MM0s,
#                      r_pars=(8., 22., 1000),
#                      dens_pars=(1e-1, None, None, None, None),
#                      driver=(None, {}),
#                      electrons=(None, {}),
#                      synchrotron=(None, {}),
#                      eats=(None, {})
#                      ):
#
#         # initialize angular structure
#         struct = Structure_Uniform.from_profile(
#             nshells=nshells,
#             dist_EEs=dist_EEs,
#             dist_Gam0s=dist_Gam0s,
#             dist_Beta0s=dist_Beta0s,
#             dist_MM0s=dist_MM0s
#         )
#
#         pars = locals()
#         for key in ["dist_cthetas", "dist_EEs", "dist_Gam0s", "dist_Beta0s", "dist_MM0s"]:
#             pars.pop(key)
#         pars["struct"] = struct
#         return cls(**pars)



# class StructuredJet:
#
#     def __init__(self,
#                  struct=None,
#                  r_pars=(8., 22., 1000),
#                  dens_pars=(1e-1, None, None, None, None),
#                  driver=None,
#                  driver_kwargs=None,
#                  electrons=None,
#                  electron_kwargs=None,
#                  synchrotron=None,
#                  synchrotron_kwarkgs=None,
#                  eats=None
#                  ):
#         # structure object with angular distribution of jet parameters
#         self.struct = struct
#
#         # create radial grid for dynamical evolution
#         self.RRs = np.logspace(r_pars[0], r_pars[1], r_pars[2])
#         self.dens_pars = dens_pars
#
#         # loop over the angular __models
#         self.shells = []
#         self.cthetas = []
#
#         if driver is None: raise NameError(" 'driver' cannot be set to None. Set of of the drivers from 'dynamics'")
#
#         for ii in range(self.struct.nlayers):
#
#             print("\tlayer: {}/{}  [E0: {:.2e}  M0: {:2e}  G0: {:.1f}  theta0: {:.2f}]"
#                   .format(ii, self.struct.nlayers, self.struct.cell_EEs[ii],
#                           self.struct.cell_MM0s[ii], self.struct.cell_Gam0s[ii],
#                           struct.theta0j / (np.pi / 180.)))
#
#             if struct.structure == "uniform" and ii > 0:
#                 self.shells.append(shell)
#                 continue
#
#             if (electrons is None) and (synchrotron is None):
#                 shell = Shell_FS(
#                     Gamma0=self.struct.cell_Gam0s[ii],
#                     E0=self.struct.cell_EEs[ii],
#                     M0=self.struct.cell_MM0s[ii],
#                     thetaE=0.,
#                     theta0=self.struct.theta0j,
#                     r_grid=self.RRs,
#                     ncells=self.struct.ncells,
#                     dens_pars=self.dens_pars,
#                     driver=driver,
#                     driver_kwargs=driver_kwargs,
#                 )
#             elif (electrons is None) and (not synchrotron is None):
#                 pass
#
#             else:
#                 shell = Shell_FS_Electrons_Synchrotron(
#                     Gamma0=self.struct.cell_Gam0s[ii],
#                     E0=self.struct.cell_EEs[ii],
#                     M0=self.struct.cell_MM0s[ii],
#                     thetaE=0.,
#                     theta0=self.struct.theta0j,
#                     r_grid=self.RRs,
#                     ncells=self.struct.ncells,
#                     dens_pars=self.dens_pars,
#                     driver=driver,
#                     driver_kwargs=driver_kwargs,
#                     electrons=electrons,
#                     electron_kwargs=electron_kwargs,
#                     synchrotron=synchrotron,
#                     synchrotron_kwargs=synchrotron_kwarkgs
#                 )
#
#             self.shells.append(shell)
#
#         self.cthetas = [self.struct.cthetas0[ii] + 0.5 * (2 * self.shells[ii].dyn.get("theta")[1:] - 2 * struct.theta0j) for ii
#                         in range(struct.nlayers)]
#         '''
#         def get_thetas(self, joAngle):
#
#             fac = arange(0,self.nlayers+1)/float(self.nlayers)        # Numerical factor for use during execution
#             thetas  = 2.*arcsin(fac*sin(joAngle/4.))         # Calculate the propagation angle with respect to jet axis
#             cthetas = 0.5*(thetas[1:]+thetas[:-1])
#
#             return thetas, cthetas
#         '''
#         # thetas0, cthetas0 = self.get_thetas(self.initJoAngle)
#         # cthetas[ii,:] = cthetas0[:] + 0.5*(self.joAngle[ii]-self.initJoAngle)
#         # self.cthetas[:,ii] = self.cthetas0[ii] + 0.5*(self.joAngles[:,ii]-self.initJoAngle)
#
#         # integrate over equal arrival time surfaces
#         if not eats is None:
#             self.eats = EATS_Integ_Struct(
#                 nlayers=self.struct.nlayers,
#                 ncells=self.struct.ncells,
#                 layers=self.struct.layer,
#                 cphis=self.struct.cphis,
#                 cthetas=self.cthetas,
#                 Rs=[self.shells[i].dyn.get("R")[1:] for i in range(self.struct.nlayers)],
#                 Gammas=[self.shells[i].dyn.get("Gamma")[1:] for i in range(self.struct.nlayers)],
#                 betas=[self.shells[i].dyn.get("beta")[1:] for i in range(self.struct.nlayers)],
#                 tts=[self.shells[i].dyn.get("tt")[1:] for i in range(self.struct.nlayers)],
#                 thickness=[self.shells[i].dyn.get("thickness")[1:] for i in range(self.struct.nlayers)],
#                 spectra=[self.shells[i].spectrum for i in range(self.struct.nlayers)],
#             )
#         else:
#             self.eats = None
#
#     @classmethod
#     def from_analytic_pars(cls,
#                      nlayers=100,
#                      EEc0=1.e52,
#                      Gamc0=100.,
#                      theta0j=np.pi / 2.,
#                      thetaCo=np.pi / 2.,
#                      kk=1.,
#                      structure='uniform',
#                      r_pars=(8., 22., 1000),
#                      dens_pars=(1e-1, None, None, None, None),
#                      driver=None,
#                      driver_kwargs=None,
#                      electrons=None,
#                      electron_kwargs=None,
#                      synchrotron=None,
#                      synchrotron_kwarkgs=None,
#                      eats=None
#                      ):
#         # initialize angular structure
#         struct = Structure_Angular.from_analytic_pars(
#             nlayers=nlayers,
#             EEc0=EEc0, Gamc0=Gamc0,
#             theta0j=theta0j, theta0c=thetaCo,
#             kk=kk,
#             structure=structure)
#         driver_kwargs["ncells"] = struct.ncells
#         pars = locals()
#         for key in ["EEc0", "Gamc0", "theta0j", "thetaCo", "kk", "structure"]:
#             pars.pop(key)
#         pars["struct"] = struct
#         return cls(**struct)
#
#     @classmethod
#     def from_profile(cls,
#                      nlayers,
#                      dist_cthetas,
#                      dist_EEs,
#                      dist_Gam0s,
#                      dist_Beta0s,
#                      dist_MM0s,
#                      r_pars=(8., 22., 1000),
#                      dens_pars=(1e-1, None, None, None, None),
#                      driver=None,
#                      driver_kwargs=None,
#                      electrons=None,
#                      electron_kwargs=None,
#                      synchrotron=None,
#                      synchrotron_kwarkgs=None,
#                      eats=None
#                      ):
#
#
# class StructuredJet:
#
#     def __init__(self,
#                  nlayers=100,
#                  EEc0=1.e52,
#                  Gamc0=100.,
#                  theta0j=np.pi/2.,
#                  thetaCo=np.pi/2.,
#                  kk=1.,
#                  structure='uniform',
#                  r_pars = (8., 22., 1000),
#                  dens_pars=(1e-1, None, None, None, None),
#                  driver=None,
#                  driver_kwargs=None,
#                  electrons=None,
#                  electron_kwargs=None,
#                  synchrotron=None,
#                  synchrotron_kwarkgs=None,
#                  eats=None
#                  ):
#         """
#         driver_kwargs={"aa": np.nan, "useSpread": False, "epsilon_e_rad": 0, "adiabLoss": True,
#                                 "tprompt": 1e3, "epsilon_e_rad_RS": 0., "adiabLoss_RS": True,
#                                 "ode_rtol": 1e-3, "ode_nsteps": 3000},
#         electrons=Electron_BPL,
#         electron_kwargs={"p": 2.2, "eps_e": 0.2, "eps_b": 1e-3},
#         synchrotron=Synchrotron_Joh06,
#         synchrotron_kwarkgs={"ssa": False},
#         """
#
#         # initialize structure and distributions
#         self.struct = Structure_Angular.from_analytic_pars(
#             nlayers=nlayers,
#             EEc0=EEc0, Gamc0=Gamc0,
#             theta0j=theta0j, theta0c=thetaCo,
#             kk = kk,
#             structure=structure)
#         driver_kwargs["ncells"] = self.struct.ncells
#
#         # create radial grid for dynamical evolution
#         self.RRs = np.logspace(r_pars[0], r_pars[1], r_pars[2])
#         self.dens_pars = dens_pars
#
#         # loop over the angular __models
#         self.shells = []
#         self.cthetas = []
#
#         if driver is None: raise NameError(" 'driver' cannot be set to None. Set of of the drivers from 'dynamics'")
#
#         for ii in range(self.struct.nlayers):
#
#             print("\tlayer: {}/{}  [E0: {:.2e}  M0: {:2e}  G0: {:.1f}  theta0: {:.2f}]"
#                   .format(ii, self.struct.nlayers, self.struct.cell_EEs[ii],
#                           self.struct.cell_MM0s[ii], self.struct.cell_Gam0s[ii],
#                           theta0j / (np.pi / 180.)))
#
#             if structure=="uniform" and ii > 0:
#                 self.shells.append(shell)
#                 continue
#
#             if (electrons is None) and (synchrotron is None):
#                 shell = Shell_FS(
#                     Gamma0=self.struct.cell_Gam0s[ii],
#                     E0=self.struct.cell_EEs[ii],
#                     M0=self.struct.cell_MM0s[ii],
#                     thetaE=0.,
#                     theta0=self.struct.theta0j,
#                     r_grid=self.RRs,
#                     ncells=self.struct.ncells,
#                     dens_pars=self.dens_pars,
#                     driver=driver,
#                     driver_kwargs=driver_kwargs,
#                 )
#             elif (electrons is None) and (not synchrotron is None):
#                 pass
#
#             else:
#                 shell = Shell_FS_Electrons_Synchrotron(
#                               Gamma0=self.struct.cell_Gam0s[ii],
#                               E0=self.struct.cell_EEs[ii],
#                               M0=self.struct.cell_MM0s[ii],
#                               thetaE=0.,
#                               theta0=self.struct.theta0j,
#                               r_grid=self.RRs,
#                               ncells=self.struct.ncells,
#                               dens_pars=self.dens_pars,
#                               driver=driver,
#                               driver_kwargs=driver_kwargs,
#                               electrons=electrons,
#                               electron_kwargs=electron_kwargs,
#                               synchrotron=synchrotron,
#                               synchrotron_kwargs=synchrotron_kwarkgs
#                               )
#
#             self.shells.append(shell)
#
#
#         self.cthetas = [ self.struct.cthetas0[ii] + 0.5 * ( 2 * self.shells[ii].dyn.get("theta")[1:] - 2 * theta0j) for ii in range(nlayers) ]
#         '''
#         def get_thetas(self, joAngle):
#
#             fac = arange(0,self.nlayers+1)/float(self.nlayers)        # Numerical factor for use during execution
#             thetas  = 2.*arcsin(fac*sin(joAngle/4.))         # Calculate the propagation angle with respect to jet axis
#             cthetas = 0.5*(thetas[1:]+thetas[:-1])
#
#             return thetas, cthetas
#         '''
#         # thetas0, cthetas0 = self.get_thetas(self.initJoAngle)
#         # cthetas[ii,:] = cthetas0[:] + 0.5*(self.joAngle[ii]-self.initJoAngle)
#         # self.cthetas[:,ii] = self.cthetas0[ii] + 0.5*(self.joAngles[:,ii]-self.initJoAngle)
#
#         # integrate over equal arrival time surfaces
#         if not eats is None:
#             self.eats = EATS_Integ_Struct(
#                 nlayers=self.struct.nlayers,
#                 ncells=self.struct.ncells,
#                 layers=self.struct.layer,
#                 cphis=self.struct.cphis,
#                 cthetas=self.cthetas,
#                 Rs=[self.shells[i].dyn.get("R")[1:] for i in range(self.struct.nlayers)],
#                 Gammas=[self.shells[i].dyn.get("Gamma")[1:] for i in range(self.struct.nlayers)],
#                 betas=[self.shells[i].dyn.get("beta")[1:] for i in range(self.struct.nlayers)],
#                 tts=[self.shells[i].dyn.get("tt")[1:] for i in range(self.struct.nlayers)],
#                 thickness=[self.shells[i].dyn.get("thickness")[1:] for i in range(self.struct.nlayers)],
#                 spectra=[self.shells[i].spectrum for i in range(self.struct.nlayers)],
#             )
#         else:
#             self.eats = None
#
#
