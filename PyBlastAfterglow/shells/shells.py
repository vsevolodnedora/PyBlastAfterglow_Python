"""

"""

from PyBlastAfterglow.dynamics.dynamics import rho_dlnrho1dR

class Shell_FS:
    def __init__(
            self,
            Gamma0,
            E0,
            M0,
            thetaE,
            theta0,
            r_grid,
            ncells,
            dens_pars,
            driver,
            driver_kwargs
    ):
        # initial dynamics parameters
        self.Gamma0 = Gamma0
        self.thetaE = thetaE
        self.theta0 = theta0
        self.Rstart = r_grid[0]
        self.E0 = E0
        self.M0 = M0
        self.rho0, self.dlnrho1dR_0 = rho_dlnrho1dR(r_grid[0], *dens_pars)
        self.ncells = ncells

        # initialize the dynamics for external shock(s)
        self.dyn = driver.from_obj(self, **driver_kwargs)

        # run dynamical evolution
        for i in range(1, len(r_grid)):  # tqdm(range(1, len(r_grid))):

            rho, dlnrho1dR = rho_dlnrho1dR(r_grid[i], *dens_pars)

            self.dyn.evolove(r_grid[i], rho, dlnrho1dR)


class Shell_FS_Electrons_Synchrotron:
    def __init__(
            self,
            Gamma0,
            E0,
            M0,
            thetaE,
            theta0,
            r_grid,
            ncells,
            dens_pars,
            driver,
            driver_kwargs,
            electrons,
            electron_kwargs,
            synchrotron,
            synchrotron_kwargs
    ):
        # initial dynamics parameters
        self.Gamma0 = Gamma0
        self.thetaE = thetaE
        self.theta0 = theta0
        self.Rstart = r_grid[0]
        self.E0 = E0
        self.M0 = M0
        self.rho0, self.dlnrho1dR_0 = rho_dlnrho1dR(r_grid[0], *dens_pars)
        self.ncells = ncells

        # radiation [FS]
        self.spectrum = []

        # radiation [RS]

        # initialize the dynamics for external shock(s)
        self.dyn = driver.from_obj(self, **driver_kwargs)

        # initialize electrons for external forward shock
        self.ele = electrons.from_obj_fs(self.dyn, **electron_kwargs)

        # run dynamical evolution
        for i in range(1, len(r_grid)):  # tqdm(range(1, len(r_grid))):

            rho, dlnrho1dR = rho_dlnrho1dR(r_grid[i], *dens_pars)

            self.dyn.evolove(r_grid[i], rho, dlnrho1dR)

            # print(self.dyn.get_last("M2") / 4 / np.pi)
            # print(self.dyn.get_last("thickness") * self.dyn.get_last("Gamma") * self.dyn.get_last("R")**2 * self.dyn.get_last("rho2"))
            # print('\n')
            self.ele.compute_char_lfs_fs(self.dyn)

            rad = synchrotron.from_obj_fs(self.ele, self.dyn, **synchrotron_kwargs)

            self.spectrum.append(rad.get_spectrum())
