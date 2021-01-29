"""

"""

import numpy as np
from PyBlastAfterglow.uutils import cgs

class GenAngStruct:

    def __init__(self):
        pass

    @staticmethod
    def cell_in_layer(i_layer):
        return 2 * i_layer + 1

    @staticmethod
    def _number_of_cells_in_layers(nlayers):

        #ii_cells = lambda ii: 2 * ii + 1  # Origin of this EXACT equations is unknown
        cil = []
        for ii in range(nlayers):
            cil.append(GenAngStruct.cell_in_layer(ii))
        return (cil, np.sum(cil))

    @staticmethod
    def _generate_grid_cthetas(nlayers, theta0):
        fac = np.arange(0, nlayers + 1) / float(nlayers)
        thetas = 2. * np.arcsin(fac * np.sin(theta0 / 2.))
        cthetas = 0.5 * (thetas[1:] + thetas[:-1])
        return cthetas

    @staticmethod
    def _generate_azymuthal_grid(nlayers):

        cil, ncells = Structure_Angular._number_of_cells_in_layers(nlayers)

        layer = np.array([])
        cphis = np.array([])
        for ii in range(nlayers):  # Loop over layers and populate the arrays
            num = cil[ii]
            layer = np.append(layer, np.ones(num) * (ii + 1))  # Layer on which the cells are
            cphis = np.append(cphis, (np.arange(0, num)) * 2. * np.pi / num)
            layer = layer.astype('int')

        return (cphis, layer, ncells)


class Structure_Angular(GenAngStruct):

    def __init__(self,
                 nlayers,
                 cthetas0,
                 cell_EEs,
                 cell_Gam0s,
                 cell_Beta0s,
                 cell_MM0s
                 ):
        # fail-safe
        if nlayers > 1:
            assert (cthetas0[0] >= 0.)
            assert (cthetas0[-1] >= 0.)
            assert (cthetas0[-1] <= np.pi/2.)
            assert (cthetas0[0] < cthetas0[-1])
            assert nlayers == len(cthetas0)
            assert nlayers == len(cell_EEs)
            assert nlayers == len(cell_Gam0s)
            assert nlayers == len(cell_Beta0s)
            assert nlayers == len(cell_MM0s)

        # assign class attributes
        self.cell_EEs = cell_EEs
        self.cell_Gam0s = cell_Gam0s
        self.cell_Beta0s = cell_Beta0s
        self.cell_MM0s = cell_MM0s

        self.nlayers = nlayers
        self.cthetas0 = cthetas0

        self.cphis, self.layer, self.ncells = self._generate_azymuthal_grid(nlayers)

        super(Structure_Angular, self).__init__()


    @classmethod
    def from_analytic_pars(
            cls,
            nlayers,
            EEc0, Gamc0,
            theta0j, theta0c, Mc0=None,
            kk=1.,
            structure='gaussian'
    ):
        # from isotropic equivalent energy to energy per cell
        _, ncells = cls._number_of_cells_in_layers(nlayers)
        angExt0 = 2. * cgs.pi * (1. - np.cos(theta0j)) / ncells
        EEc0 = EEc0 * angExt0 / (4. * np.pi)
        # MMc0 = EEc0 / (Gamc0 * cgs.c ** 2.)

        # generate initial azymuthal angles of the grid
        cthetas0 = cls._generate_grid_cthetas(nlayers, theta0j)

        # create distributions of initial condiitons for jets to be launched at different angles (layers)
        if structure == 'uniform':
            cell_EEs = np.zeros_like(cthetas0) + EEc0
            cell_Gam0s = np.zeros_like(cthetas0) + Gamc0
            if not (Mc0 is None):
                cell_MM0s = np.zeros_like(cthetas0) + Mc0 * angExt0 / (4 * np.pi)#
            else:
                cell_MM0s = cell_EEs / (cell_Gam0s * cgs.c ** 2.)
        elif structure == 'gaussian':
            cell_EEs = EEc0 * np.exp(-1. * cthetas0 ** 2. / (theta0c ** 2.))  # Just for texting
            cell_Gam0s = 1. + (Gamc0 - 1) * np.exp(-1. * cthetas0 ** 2. / (2. * theta0c ** 2.))
            cell_MM0s = cell_EEs / (cell_Gam0s * cgs.c ** 2.)
        elif structure == 'power-law':
            cell_EEs = np.zeros(nlayers)
            cell_Gam0s = np.zeros(nlayers)
            cell_EEs[cthetas0 <= theta0c] = EEc0
            cell_Gam0s[cthetas0 <= theta0c] = Gamc0
            wings = cthetas0 > theta0c
            cell_EEs[wings] = EEc0 * (cthetas0[wings] / theta0c) ** (-1. * kk)
            cell_Gam0s[wings] = 1. + (Gamc0 - 1.) * (cthetas0[wings] / theta0c) ** (-1. * kk)
            cell_MM0s = cell_EEs / (cell_Gam0s * cgs.c ** 2.)
        else:
            raise NameError("structure: {} is not recognized".format(structure))
        cell_Beta0s = np.sqrt(1. - np.power(cell_Gam0s, -2))

        cls.theta0j = theta0j
        cls.structure = structure

        return cls(nlayers, cthetas0, cell_EEs, cell_Gam0s, cell_Beta0s, cell_MM0s)

        # return(cls, nlayers, cthetas0, cell_EEs, cell_Gam0s, cell_Beta0s, cell_MM0s)

    @classmethod
    def from_analytic_pars_ejecta(
            cls,
            nlayers,
            Ek,
            beta,
            mass,
            theta0=np.pi/2.,
            kk=1.,
            structure='uniform'
    ):
        # from isotropic equivalent energy to energy per cell
        _, ncells = cls._number_of_cells_in_layers(nlayers)
        # angExt0 = 2. * cgs.pi * (1. - np.cos(theta0j)) / ncells
        # EEc0 = EEc0 * angExt0 / (4. * np.pi)
        # MMc0 = EEc0 / (Gamc0 * cgs.c ** 2.)

        # generate initial azymuthal angles of the grid
        cthetas0 = cls._generate_grid_cthetas(nlayers, theta0)
        Gamma = np.float64(np.sqrt(1. / (1. - np.float64(beta) ** 2.)))
        Ek /= ncells
        mass /= ncells

        # create distributions of initial condiitons for jets to be launched at different angles (layers)
        if structure == 'uniform':
            cell_EEs = np.zeros_like(cthetas0) + Ek
            cell_Gam0s = np.zeros_like(cthetas0) + Gamma
            cell_MM0s = np.zeros_like(cthetas0) + mass
            cell_Beta0s = np.zeros_like(cthetas0) + beta
            # if not (Mc0 is None):
            #     cell_MM0s = np.zeros_like(cthetas0) + Mc0 * angExt0 / (4 * np.pi)  #
            # else:
            #     cell_MM0s = cell_EEs / (cell_Gam0s * cgs.c ** 2.)
        # elif structure == 'gaussian':
        #     cell_EEs = Ek * np.exp(-1. * cthetas0 ** 2. / (theta0c ** 2.))  # Just for texting
        #     cell_Gam0s = 1. + (Gamc0 - 1) * np.exp(-1. * cthetas0 ** 2. / (2. * theta0c ** 2.))
        #     cell_MM0s = cell_EEs / (cell_Gam0s * cgs.c ** 2.)
        # elif structure == 'power-law':
        #     cell_EEs = np.zeros(nlayers)
        #     cell_Gam0s = np.zeros(nlayers)
        #     cell_EEs[cthetas0 <= theta0c] = EEc0
        #     cell_Gam0s[cthetas0 <= theta0c] = Gamc0
        #     wings = cthetas0 > theta0c
        #     cell_EEs[wings] = EEc0 * (cthetas0[wings] / theta0c) ** (-1. * kk)
        #     cell_Gam0s[wings] = 1. + (Gamc0 - 1.) * (cthetas0[wings] / theta0c) ** (-1. * kk)
        #     cell_MM0s = cell_EEs / (cell_Gam0s * cgs.c ** 2.)
        else:
            raise NameError("structure: {} is not recognized".format(structure))
        # cell_Beta0s = np.sqrt(1. - np.power(cell_Gam0s, -2))

        cls.theta0j = theta0
        cls.structure = structure

        return cls(nlayers, cthetas0, cell_EEs, cell_Gam0s, cell_Beta0s, cell_MM0s)

    @classmethod
    def from_analytic_func(
            cls,
            **kwargs
    ):
        raise ValueError("not implemented")

    @classmethod
    def from_profile(
            cls,
            nlayers,
            dist_cthetas,
            dist_EEs,
            dist_Gam0s,
            dist_Beta0s,
            dist_MM0s
    ):
        cls.theta0j = dist_cthetas[-1]
        cls.structure = "profile"

        for i_layer in range(nlayers):
            dist_MM0s[i_layer] /= Structure_Angular.cell_in_layer(i_layer)
            dist_EEs[i_layer] /= Structure_Angular.cell_in_layer(i_layer)

        # _, ncells = Structure_Angular._number_of_cells_in_layers(nlayers)
        # dist_MM0s /= (ncells / nlayers)
        # dist_EEs /= (ncells / nlayers)

        return cls(nlayers,
                   dist_cthetas,
                   dist_EEs,
                   dist_Gam0s,
                   dist_Beta0s,
                   dist_MM0s
                   )


# class Structure_Uniform:
#
#     def __init__(
#             self,
#             **kwargs
#     ):
#
#         self.shell_EEs = kwargs["Eks"]
#         self.shell_Gam0s = kwargs["Gammas"]
#         self.shell_Beta0s = kwargs["betas"]
#         self.shell_MM0s = kwargs["masses"]
#
#         self.nshells = kwargs["nshells"]
#
#     @classmethod
#     def from_analytic_func(cls):
#         raise NameError("Not implemented")
#
#     @classmethod
#     def from_analytic_pars(cls):
#         raise NameError("Not implemented")
#
#     @classmethod
#     def from_profile(
#             cls,
#             nshells,
#             dist_EEs,
#             dist_Gam0s,
#             dist_Beta0s,
#             dist_MM0s
#     ):
#         cls.structure = "profile"
#         return cls(nshells, dist_EEs, dist_Gam0s, dist_Beta0s, dist_MM0s)

#
#     @staticmethod
#     def create_cell_distribution(
#             nlayers,
#             EEc0, Gamc0,
#             theta0j, thetaCo,
#             kk=1.,
#             structure='gaussian'
#
#     ):
#
#         theta0j = min(theta0j, np.sqrt(-2. * theta0c ** 2. * np.log(1e-8 / (Gamc0 - 1.))))
#
#         cthetas0, self.cphis, self.layer, self.ncells = self.create_angular_structure(nlayers, theta0j)
#
#         if structure == 'uniform':
#             cell_EEs = np.zeros_like(cthetas0) + EEc0
#             cell_Gam0s = np.zeros_like(cthetas0) + Gamc0
#
#         elif structure == 'gaussian':
#             cell_EEs = self.EEc0 * np.exp(-1. * cthetas0 ** 2. / (theta0c ** 2.))  # Just for texting
#             cell_Gam0s = 1. + (Gamc0 - 1) * np.exp(-1. * cthetas0 ** 2. / (2. * theta0c ** 2.))
#
#         elif structure == 'power-law':
#             cell_EEs = np.zeros(nlayers)
#             cell_Gam0s = np.zeros(nlayers)
#             cell_EEs[cthetas0 <= theta0c] = EEc0
#             cell_Gam0s[cthetas0 <= theta0c] = Gamc0
#             wings = cthetas0 > theta0c
#             cell_EEs[wings] = self.EEc0 * (cthetas0[wings] / theta0c) ** (-1. * kk)
#             cell_Gam0s[wings] = 1. + (Gamc0 - 1.) * (cthetas0[wings] / theta0c) ** (-1. * kk)
#
#         else:
#             raise NameError("structure: {} is not recognized".format(self.structure))
#
#         cell_Beta0s = np.sqrt(1. - np.power(cell_Gam0s, -2))
#         cell_MM0s = cell_EEs / (cell_Gam0s * cgs.c ** 2.)
#
#         return (cell_EEs, cell_Gam0s, cell_Beta0s, cell_MM0s)
#
#
#
#
#
#
#
#
#
#
# class Structure_Angular:
#
#     def __init__(self):
#         pass
#
#     @classmethod
#     def from_analytic_pars(
#             cls,
#             nlayers,
#             EEc0, Gamc0,
#             theta0j, thetaCo,
#             kk=1.,
#             structure='gaussian'
#     ):
#
#         def create_cell_distribution():
#
#             if self.structure == 'uniform':
#                 cell_EEs = np.zeros_like(self.cthetas0) + self.EEc0
#                 cell_Gam0s = np.zeros_like(self.cthetas0) + self.Gamc0
#
#             elif self.structure == 'gaussian':
#                 cell_EEs = self.EEc0 * np.exp(-1. * self.cthetas0 ** 2. / (self.theta0c ** 2.))  # Just for texting
#                 cell_Gam0s = 1. + (self.Gamc0 - 1) * np.exp(-1. * self.cthetas0 ** 2. / (2. * self.theta0c ** 2.))
#
#             elif self.structure == 'power-law':
#                 cell_EEs = np.zeros(self.nlayers)
#                 cell_Gam0s = np.zeros(self.nlayers)
#                 cell_EEs[self.cthetas0 <= self.theta0c] = self.EEc0
#                 cell_Gam0s[self.cthetas0 <= self.theta0c] = self.Gamc0
#                 wings = self.cthetas0 > self.theta0c
#                 cell_EEs[wings] = self.EEc0 * (self.cthetas0[wings] / self.theta0c) ** (-1. * self.kk)
#                 cell_Gam0s[wings] = 1. + (self.Gamc0 - 1.) * (self.cthetas0[wings] / self.theta0c) ** (-1. * self.kk)
#
#             else:
#                 raise NameError("structure: {} is not recognized".format(self.structure))
#
#             cell_Beta0s = np.sqrt(1. - np.power(cell_Gam0s, -2))
#             cell_MM0s = cell_EEs / (cell_Gam0s * cgs.c ** 2.)
#
#             return (cell_EEs, cell_Gam0s, cell_Beta0s, cell_MM0s)
#
#
#     @classmethod
#     def from_analytic_frofile(cls):
#         pass
#
#     @classmethod
#     def from_numeric_profile(cls):
#         pass
#
#     @staticmethod
#     def _phi_cells_in_theta_layer(nlayers):
#
#         ii_cells = lambda ii: 2 * ii + 1  # Origin of this EXACT equations is unknown
#         cil = []
#         for ii in range(nlayers):
#             cil.append(ii_cells(ii))
#         return (cil, np.sum(cil))
#
#     @staticmethod
#     def _generate_grid_cthetas(nlayers, theta0):
#         fac = np.arange(0, nlayers + 1) / float(nlayers)
#         thetas = 2. * np.arcsin(fac * np.sin(theta0 / 2.))
#         cthetas = 0.5 * (thetas[1:] + thetas[:-1])
#         return cthetas
#
#     def create_angular_structure(self, nlayers, theta0):
#
#         cil, ncells = Structure_Angular._phi_cells_in_theta_layer(nlayers)
#
#         cthetas = Structure_Angular._generate_grid_cthetas(nlayers, theta0)
#
#         layer = np.array([])
#         cphis = np.array([])
#         for ii in range(nlayers):  # Loop over layers and populate the arrays
#             num = cil[ii]
#             layer = np.append(layer, np.ones(num) * (ii + 1))  # Layer on which the cells are
#             cphis = np.append(cphis, (np.arange(0, num)) * 2. * np.pi / num)
#             layer = layer.astype('int')
#
#         return (cthetas, cphis, layer, ncells)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# class UniformOverlapping:
#
#     def __init__(self,
#                  nlayers,
#                  E0, Gam0,
#                  theta0j,
#                  structure='single'
#                  ):
#         # reading parameters
#         self.nlayers = nlayers
#         self.structure = structure
#
#         # single layer
#         self.E0 = E0 * (1 - np.cos(theta0j)) / (2 * np.pi)
#         self.Gamma0 = Gam0
#         self.theta0 = theta0j
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# class Structure:
#
#     def __init__(self,
#                  cthetas0, cphis, layer, ncells, nlayers,
#                  cell_EEs, cell_Gam0s, cell_Beta0s, cell_MM0s
#                  ):
#
#         self.cell_EEs = cell_EEs
#         self.cell_Gam0s = cell_Gam0s
#         self.cell_Beta0s = cell_Beta0s
#         self.cell_MM0s = cell_MM0s
#
#         self.nlayers = nlayers
#         self.cthetas0 = cthetas0
#         self.cphis = cphis
#         self.layer = layer
#         self.ncells = ncells
#
# class Structure_Angular_Analytic():
#
#     def __init__(self,
#                  nlayers,
#                  EEc0, Gamc0,
#                  theta0j, thetaCo,
#                  kk = 1.,
#                  structure='gaussian'
#                  ):
#
#         # cheks
#         assert not (thetaCo > theta0j)
#
#         # reading the parameters
#         self.nlayers = nlayers
#         self.structure = structure
#         self.kk = kk
#         self.EEc0 = EEc0
#         self.Gamc0 = Gamc0
#         self.theta0c = thetaCo
#         self.theta0j = min(theta0j, np.sqrt(-2. * self.theta0c ** 2. * np.log(1e-8 / (self.Gamc0 - 1.))))
#
#         # create angular grid
#         self.cthetas0, self.cphis, self.layer, self.ncells = self.create_angular_structure(nlayers, theta0j)
#
#         # create energy/LF distribution
#         self.angExt0 = 2. * cgs.pi * (1. - np.cos(theta0j)) / self.ncells
#         self.EEc0 = self.EEc0 * self.angExt0 / (4. * np.pi)
#         self.MMc0 = self.EEc0 / (self.Gamc0 * cgs.c ** 2.)
#         cell_EEs, cell_Gam0s, cell_Beta0s, cell_MM0s = self.create_cell_distribution()
#
#
#     @staticmethod
#     def create_angular_structure(nlayers, theta0):
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
#         cthetas = cthetas(nlayers, theta0)
#
#         layer = np.array([])
#         cphis = np.array([])
#         for ii in range(nlayers):  # Loop over layers and populate the arrays
#             num = cil[ii]
#             layer = np.append(layer, np.ones(num) * (ii + 1))  # Layer on which the cells are
#             cphis = np.append(cphis, (np.arange(0, num)) * 2. * np.pi / num)
#             layer = layer.astype('int')
#
#         return (cthetas, cphis, layer, ncells)
#
#     def create_cell_distribution(self):
#
#         if self.structure == 'uniform':
#             cell_EEs = np.zeros_like(self.cthetas0) + self.EEc0
#             cell_Gam0s = np.zeros_like(self.cthetas0) + self.Gamc0
#
#         elif self.structure == 'gaussian':
#             cell_EEs = self.EEc0 * np.exp(-1. * self.cthetas0 ** 2. / (self.theta0c ** 2.))  # Just for texting
#             cell_Gam0s = 1. + (self.Gamc0 - 1) * np.exp(-1. * self.cthetas0 ** 2. / (2. * self.theta0c ** 2.))
#
#         elif self.structure == 'power-law':
#             cell_EEs = np.zeros(self.nlayers)
#             cell_Gam0s = np.zeros(self.nlayers)
#             cell_EEs[self.cthetas0 <= self.theta0c] = self.EEc0
#             cell_Gam0s[self.cthetas0 <= self.theta0c] = self.Gamc0
#             wings = self.cthetas0 > self.theta0c
#             cell_EEs[wings] = self.EEc0 * (self.cthetas0[wings] / self.theta0c) ** (-1. * self.kk)
#             cell_Gam0s[wings] = 1. + (self.Gamc0 - 1.) * (self.cthetas0[wings] / self.theta0c) ** (-1. * self.kk)
#
#         else:
#             raise NameError("structure: {} is not recognized".format(self.structure))
#
#         cell_Beta0s = np.sqrt(1. - np.power(cell_Gam0s, -2))
#         cell_MM0s = cell_EEs / (cell_Gam0s * cgs.c ** 2.)
#
#         return (cell_EEs, cell_Gam0s, cell_Beta0s, cell_MM0s)



# class Structure_Angular:
#
#     def __init__(self,
#                  nlayers,
#                  EEc0, Gamc0,
#                  theta0j, thetaCo,
#                  kk=1.,
#                  structure='gaussian'
#                  ):
#
#         # cheks
#         assert not (thetaCo > theta0j)
#
#         # reading the parameters
#         self.nlayers = nlayers
#         self.structure = structure
#         self.kk = kk
#         self.EEc0 = EEc0
#         self.Gamc0 = Gamc0
#         self.theta0c = thetaCo
#         self.theta0j = min(theta0j, np.sqrt(-2. * self.theta0c ** 2. * np.log(1e-8 / (self.Gamc0 - 1.))))
#
#         # create angular grid
#         self.cthetas0, self.cphis, self.layer, self.ncells = self.create_angular_structure(nlayers, theta0j)
#
#         # create energy/LF distribution
#         self.angExt0 = 2. * cgs.pi * (1. - np.cos(theta0j)) / self.ncells
#         self.EEc0 = self.EEc0 * self.angExt0 / (4. * np.pi)
#         self.MMc0 = self.EEc0 / (self.Gamc0 * cgs.c ** 2.)
#         self.create_cell_distribution()
#
#     @staticmethod
#     def create_angular_structure(nlayers, theta0):
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
#         cthetas = cthetas(nlayers, theta0)
#
#         layer = np.array([])
#         cphis = np.array([])
#         for ii in range(nlayers):  # Loop over layers and populate the arrays
#             num = cil[ii]
#             layer = np.append(layer, np.ones(num) * (ii + 1))  # Layer on which the cells are
#             cphis = np.append(cphis, (np.arange(0, num)) * 2. * np.pi / num)
#             layer = layer.astype('int')
#
#         return (cthetas, cphis, layer, ncells)
#
#     def create_cell_distribution(self):
#
#         if self.structure == 'uniform':
#             self.cell_EEs = np.zeros_like(self.cthetas0) + self.EEc0
#             self.cell_Gam0s = np.zeros_like(self.cthetas0) + self.Gamc0
#
#         elif self.structure == 'gaussian':
#             self.cell_EEs = self.EEc0 * np.exp(-1. * self.cthetas0 ** 2. / (self.theta0c ** 2.))  # Just for texting
#             self.cell_Gam0s = 1. + (self.Gamc0 - 1) * np.exp(-1. * self.cthetas0 ** 2. / (2. * self.theta0c ** 2.))
#
#         elif self.structure == 'power-law':
#             self.cell_EEs = np.zeros(self.nlayers)
#             self.cell_Gam0s = np.zeros(self.nlayers)
#             self.cell_EEs[self.cthetas0 <= self.theta0c] = self.EEc0
#             self.cell_Gam0s[self.cthetas0 <= self.theta0c] = self.Gamc0
#             wings = self.cthetas0 > self.theta0c
#             self.cell_EEs[wings] = self.EEc0 * (self.cthetas0[wings] / self.theta0c) ** (-1. * self.kk)
#             self.cell_Gam0s[wings] = 1. + (self.Gamc0 - 1.) * (self.cthetas0[wings] / self.theta0c) ** (-1. * self.kk)
#
#         self.cell_Beta0s = np.sqrt(1. - np.power(self.cell_Gam0s, -2))
#         self.cell_MM0s = self.cell_EEs / (self.cell_Gam0s * cgs.c ** 2.)