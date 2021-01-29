"""
    pass
"""

import numpy as np
from PyBlastAfterglow.uutils import cgs, axes_reshaper
from scipy import interpolate

gamma_to_integrate = np.logspace(1., 9., 200) # default arrays to be used for integration
freq_to_integrate = np.logspace(5., 24., 300)

class Synchrotron:
    def __init__(self):
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


def SED_SariPiran97(nu, nu_m, nu_c, p, PmaxF, PmaxS):
    """
    Following Sari & Piran 1997
    :param nu:
    :param nu_m:
    :param nu_c:
    :param p:
    :return:
    """

    if hasattr(nu, '__len__'):  # array treatment
        res = np.zeros(len(nu))
        sc = np.array(nu_m <= nu_c, dtype=bool)  # slow cooling
        fc = np.array(nu_m > nu_c, dtype=bool)  # fast cooling
        if len(sc) > 0:  # Slow cooling
            fil1 = sc & (nu < nu_m)
            fil2 = sc & ((nu >= nu_m) & (nu < nu_c))
            fil3 = sc & (nu >= nu_c)
            res[fil1] = (nu[fil1] / nu_m[fil1]) ** (1. / 3.) * PmaxS[fil1]
            res[fil2] = (nu[fil2] / nu_m[fil2]) ** (-1. * (p - 1.) / 2.) * PmaxS[fil2]
            res[fil3] = (nu_c[fil3] / nu_m[fil3]) ** (-1. * (p - 1.) / 2.) * \
                        (nu[fil3] / nu_c[fil3]) ** (-1. * p / 2.) * PmaxS[fil3]
        if len(fc) > 0:  # fast cooling
            fil1 = fc & (nu < nu_c)
            fil2 = fc & ((nu >= nu_c) & (nu < nu_m))
            fil3 = fc & (nu >= nu_m)
            res[fil1] = (nu[fil1] / nu_c[fil1]) ** (1. / 3.) * PmaxF[fil1]
            res[fil2] = (nu[fil2] / nu_c[fil2]) ** (-1. / 2.) * PmaxF[fil2]
            res[fil3] = (nu_m[fil3] / nu_c[fil3]) ** (-1. / 2.) * (nu[fil3] / nu_m[fil3]) ** (-p / 2.) * PmaxF[fil3]
        return res
    else:  # flaot treatment
        if nu_m <= nu_c:  # slow cooling
            res = 0
            if (nu < nu_m): res = (nu / nu_m) ** (1. / 3.)
            if (nu >= nu_m) & (nu < nu_c): res = (nu / nu_m) ** (-1. * (p - 1.) / 2.)
            if (nu >= nu_c): res = (nu_c / nu_m) ** (-1. * (p - 1.) / 2.) * (nu / nu_c) ** (-1. * p / 2.)
            res *= PmaxS
        else:  # fast cooling
            res = 0
            if (nu < nu_c): res = (nu / nu_c) ** (1. / 3.)
            if (nu >= nu_c) & (nu < nu_m): res = (nu / nu_c) ** (-1. / 2.)
            if (nu >= nu_m): res = (nu_m / nu_c) ** (-1. / 2.) * (nu / nu_m) ** (-p / 2.)
            res *= PmaxF
        return res

def Xintp(pp):
    data_xp = [
        1.4444980694980694, 0.9944136261210208,
        1.4820641682054725, 0.9951191166713027,
        1.4820641682054725, 0.9724149343868285,
        1.4927973392647305, 0.9572435579224101,
        1.5035305103239884, 0.9420721814579918,
        1.5196302669128754, 0.9230991471421099,
        1.5410966090313916, 0.9070816520832391,
        1.5536186419338591, 0.8936162230627284,
        1.5840292932684235, 0.8763981013871924,
        1.6194487577639751, 0.8477393594942589,
        1.6001290498573106, 0.863911976295446,
        1.6484283196239717, 0.8286159040949189,
        1.6698946617424877, 0.8123281211517009,
        1.7074607604498908, 0.7972487992565946,
        1.7128273459795198, 0.7808337068026454,
        1.7342936880980357, 0.7650864996281136,
        1.7632732499580324, 0.746467581612873,
        1.7986927144535838, 0.731088984437128,
        1.825525642101729, 0.7161350134448954,
        1.846991984220245, 0.7033609729980923,
        1.889924668457277, 0.688534311516599,
        1.922124181635051, 0.6747339605524508,
        1.954323694812825, 0.6609336095883027,
        1.975790036931341, 0.647438801449929,
        2.0187227211683734, 0.6351348268889329,
        2.0562888198757765, 0.6223979999914229,
        2.0992215041128084, 0.6092831617774097,
        2.1421541883498403, 0.5972494751007525,
        2.1904534581165014, 0.5847657003384353,
        2.2387527278831625, 0.5697232002710423,
        2.2977851687090816, 0.5576005927976534,
        2.360037560852778, 0.5433411441517004,
        2.421216635890549, 0.530382602844718,
        2.4856156622460968, 0.5171972547478341,
        2.550014688601645, 0.5050930581883057,
        2.609047129427564, 0.49560125278914946,
        2.678812741312741, 0.4848488873728878,
        2.7421384505623636, 0.47591959415041374,
        2.8076107940238373, 0.4680283630924301,
        2.8709365032734593, 0.4603604133302047,
        2.9374821638408593, 0.45062769916457235,
        3.004027824408259, 0.4440303244572723,
        3.061986948128252, 0.4389747658554187,
        3.1371191455430583, 0.43224107204123463,
        3.199371537686755, 0.42706329630907136,
        3.2659171982541544, 0.42100649737044926,
        3.3254862976330366, 0.4160537656811161,
        3.3947152509652505, 0.409916076237978,
        3.4542843503441327, 0.4048191910103307,
        3.5278065721000504, 0.39895570666724656,
        3.6018654524089304, 0.39424368788360376,
        3.6662644787644783, 0.3906085117000302,
        3.7306635051200265, 0.3871535274393493,
        3.7896959459459456, 0.3834458827650472,
        3.858388240725197, 0.37884158741356844,
        3.9335204381400026, 0.3742701966740961,
        4.004359367131106, 0.3703075412567437,
        4.068758393486654, 0.36766342064907986,
        4.138524005371831, 0.36473138468635924,
        4.201849714621453, 0.3623610774571778,
        4.272688643612556, 0.3594255160003136,
        4.3370876699681045, 0.35705168327698866,
        4.406853281853282, 0.3549305109672849,
        4.476618893738458, 0.3511876113515472,
        4.541017920094006, 0.3488137786282224,
        4.610783531979184, 0.34588174266550165,
        4.676255875440658, 0.3432881541405619,
        4.744948170219908, 0.3403055860951171,
        4.803980611045828, 0.33821966872684883,
        4.8791128084606346, 0.3361709649078737,
        4.932778663756924, 0.33293142751152094,
        5.0025442756421015, 0.33032373701000706,
        5.093776229645794, 0.3280779972405644,
        5.190374769179117, 0.3257785916158257,
        5.286973308712438, 0.3231188021453014,
        5.3835718482457615, 0.3202788207518845,
        5.480170387779083, 0.31815960705003843,
        5.576768927312406, 0.3156800095024068,
        5.673367466845727, 0.31320041195477544,
        5.7699660063790486, 0.31072081440714383,
        5.866564545912372, 0.30824121685951245,
        5.952429914386435, 0.6841999123278875,
        5.957796499916064, 0.3053467861676563
    ]
    data_xp = np.reshape(np.array(data_xp), (int(len(data_xp) / 2), 2))
    return interpolate.interp1d(data_xp[:, 0], data_xp[:, 1])(pp)

def PhiPint(pp):
    data_phip = [
        1.0062166154656431, 0.4114136714112666,
        1.0451629272536156, 0.41748522011949685,
        1.1309890587863694, 0.4352108868963911,
        1.2168151903191229, 0.4539524485203539,
        1.2572039580992425, 0.46103900983957213,
        1.3531272815770259, 0.48129823808157235,
        1.4641963929723543, 0.5034106767310167,
        1.5860839243087863, 0.5251294910490668,
        1.620702868120317, 0.5319293256420172,
        1.696431807708041, 0.5467827993381156,
        1.8175981110483992, 0.5661800094094774,
        1.8579868788285185, 0.5703882019953342,
        1.958958798278817, 0.5845489733286391,
        2.090222293564205, 0.6011745162193037,
        2.2315829807946232, 0.6141818128900468,
        2.277020344547257, 0.6178789702260863,
        2.3852045439582916, 0.6268107283083477,
        2.42342962775019, 0.6289642665816388,
        2.5344987391455187, 0.6375314406034999,
        2.584984698870668, 0.6420720891524804,
        2.6859566183209664, 0.6459045962072529,
        2.7364425780461157, 0.6483287971581736,
        2.8303464631348936, 0.6522841482352335,
        2.8828518612490486, 0.6548933614442702,
        2.9989695686168916, 0.6609515736837455,
        3.044406932369526, 0.6610930990550443,
        3.1605246397373694, 0.6649163426309681,
        3.205962003490004, 0.6670896576964045,
        3.311982518912817, 0.6690566030386015,
        3.367517074610481, 0.6708004529318601,
        3.473537590033295, 0.6738679510250482,
        3.518974953785929, 0.6750253712434158,
        3.6350926611537724, 0.6769861409330469,
        3.680530024906407, 0.6779742453435698,
        3.7966477322742502, 0.6801043308410457,
        3.842085096026884, 0.6809231194437237,
        3.948105611449698, 0.6828900647859206,
        3.998591571174847, 0.6846370025054622,
        4.109660682570175, 0.6848230440390058,
        4.155098046322809, 0.6858111484495286,
        4.271215753690653, 0.6867560232920911,
        4.316653117443288, 0.6867282328555452,
        4.4226736328661005, 0.6890338098134317,
        4.480227626952771, 0.6906917666722547,
        4.586392387974799, 0.6928037516261392,
        4.639763259684242, 0.6922196231170257,
        4.738715740745535, 0.6935813545077774,
        4.79122113885969, 0.6937524200838491,
        4.892193058309989, 0.6939954320123121,
        4.948737333202156, 0.6940624378426506,
        5.02682228424372, 0.6961141954800841,
        5.089088301238071, 0.6947215858265036,
        5.144622856935735, 0.6963807778158397,
        5.250643372358549, 0.6964852492717439,
        5.285983544166153, 0.6962943186799191,
        5.392004059588967, 0.6963987901358233,
        5.442490019314116, 0.6967065434886839,
        5.51821895890184, 0.6978454367493543,
        5.573753514599504, 0.6973035232367082,
        5.609093686407109, 0.698128487491952,
        5.7151142018299215, 0.6980636431400113,
        5.772668195916593, 0.6982316208898003,
        5.8766692729504, 0.6979648326989591,
        5.937252424620579, 0.6979277787835645
    ]
    data_phip = np.reshape(np.array(data_phip), (int(len(data_phip) / 2), 2))
    return interpolate.interp1d(data_phip[:, 0], data_phip[:, 1])(pp)

class Synchrotron_WSPN99(Synchrotron):

    def __init__(
            self,
            p,
            gamma_min,
            gamma_c,
            B,
            Ne,
            delta_shock=0.,
            ssa=False
    ):
        """
        :Ne: is the comoving NUMBER of electrons (not number density)
        """
        self.gamma_min = gamma_min
        self.gamma_c = gamma_c
        self.p = p
        self.B = B
        self.Ne = Ne
        # self.delta_shock = delta_shock
        self.ssa = ssa

        self.Xp = Xintp(p)
        self.PhiP = PhiPint(p)

        self.nu_m = 3. / (4. * np.pi) * self.Xp * gamma_min ** 2. * cgs.qe * B / (cgs.me * cgs.c)
        self.nu_c = 0.286 * 3. * gamma_c ** 2. * cgs.qe * B / (4. * np.pi * cgs.me * cgs.c)
        # self.pprimemax =  Ne * cgs.me * cgs.c ** 2 * cgs.sigmaT * B / (3 * cgs.qe)
        self.pprimemax = np.sqrt(3) * Ne* cgs.qe ** 3.* B / (cgs.me * cgs.c ** 2. * 4. * cgs.pi) * self.PhiP

        super(Synchrotron_WSPN99, self).__init__()

    @classmethod
    def from_obj_fs(cls, electons, dynamics, ssa=False):
        return cls(
            electons.p1,
            electons.get_last("gamma_min"),
            electons.get_last("gamma_c"),
            electons.get_last("B"),
            dynamics.get_last("M2")/cgs.mppme, #dynamics.get_last("rho2")/cgs.mppme,
            dynamics.get_last("thickness"),
            ssa
        )

    @staticmethod
    def evaluate_sed(
            nuprime,
            nu_m,
            nu_c,
            p,
            pmaxprime
    ):

        if hasattr(nuprime, '__len__') and (not hasattr(nu_m, '__len__')):
            nu_m = np.full_like(nuprime, nu_m)
            nu_c = np.full_like(nuprime, nu_c)
            pmaxprime = np.full_like(nuprime, pmaxprime)

        spectrum = SED_SariPiran97(nuprime, nu_m, nu_c, p, pmaxprime, pmaxprime)

        return nuprime * spectrum

    def sed(self, nuprime):
        return self.evaluate_sed(
            nuprime,
            self.nu_m,
            self.nu_c,
            self.p,
            self.pprimemax
        )
    # @staticmethod
    # def evaluate_sed_power(
    #         nuprime,
    #         gamma_min,
    #         gamma_c,
    #         p,
    #         B,
    #         Ne,
    #         delta_shock=0.,
    #         ssa=False
    # ):
    #     Xp = Xintp(p)
    #     PhiP = PhiPint(p)
    #
    #     nu_m = 3. / (4. * np.pi) * Xp * gamma_min ** 2. * cgs.qe * B / (cgs.me * cgs.c)
    #     nu_c = 0.286 * 3. * gamma_c ** 2. * cgs.qe * B / (4. * np.pi * cgs.me * cgs.c)
    #     pmaxprime = Ne * cgs.me * cgs.c ** 2 * cgs.sigmaT * B / (3 * cgs.qe)
    #     # pmaxprime = self.PhiP * (np.sqrt(3)) * num_e * cgs.qe ** 3 * B / (cgs.me * cgs.c ** 2)  # / (4. * cgs.pi)
    #
    #     if hasattr(nuprime, '__len__') and (not hasattr(nu_m, '__len__')):
    #         nu_m = np.full_like(nuprime, nu_m)
    #         nu_c = np.full_like(nuprime, nu_c)
    #         pmaxprime = np.full_like(nuprime, pmaxprime)
    #
    #     spectrum = SED_SariPiran97(nuprime, nu_m, nu_c, p, pmaxprime, pmaxprime)
    #
    #     return nuprime * spectrum
    #
    # def sed_power(self, nuprime):
    #     return self.evaluate_sed_power(
    #         nuprime,
    #         self.gamma_min,
    #         self.gamma_c,
    #         self.p,
    #         self.B,
    #         self.Ne,
    #         self.delta_shock,
    #         self.ssa
    #     )


def SSA_joh06(nu, nu_m, nu_c, p, alpha0F, alpha0S):
    """
            See description of the synchrotron self-absorption in afterglow context:
            https://iopscience.iop.org/article/10.1086/308052/fulltext/39011.text.html

            See description also in PhD thesis (p.19)
            https://www.imprs-astro.mpg.de/sites/default/files/varela_karla.pdf

            Soruce of these exact formulas:
            N/A

    """
    if hasattr(nu, '__len__'):  # array treatment
        assert len(nu) > 0
        alpha_out = np.zeros(len(nu))
        sc = np.array(nu_m <= nu_c, dtype=bool)  # slow cooling
        fc = np.array(nu_m > nu_c, dtype=bool)  # fast cooling
        if any(sc) > 0:  # Slow cooling
            fill1 = sc & (nu <= nu_m)
            fill2 = sc & ((nu_m < nu) & (nu <= nu_c))
            fill3 = sc & (nu_c < nu)
            alpha_out[fill1] = (nu[fill1] / nu_m[fill1]) ** (-5 / 3.) * alpha0S[fill1]
            alpha_out[fill2] = (nu[fill2] / nu_m[fill2]) ** (-(p + 4) / 2) * alpha0S[fill2]
            alpha_out[fill3] = (nu_c[fill3] / nu_m[fill3]) ** (-(p + 4) / 2) * \
                               (nu[fill3] / nu_c[fill3]) ** (-(p + 5) / 2) * alpha0S[fill3]
        if any(fc) > 0:  # fast cooling
            fill1f = fc & (nu <= nu_c)
            fill2f = fc & ((nu_c < nu) & (nu <= nu_m))
            fill3f = fc & (nu_m < nu)
            alpha_out[fill1f] = (nu[fill1f] / nu_c[fill1f]) ** (-5 / 3.) * alpha0F[fill1f]
            alpha_out[fill2f] = (nu[fill2f] / nu_c[fill2f]) ** (-3) * alpha0F[fill2f]
            alpha_out[fill3f] = (nu_m[fill3f] / nu_c[fill3f]) ** (-3) * \
                                (nu[fill3f] / nu_m[fill3f]) ** (-(p + 5) / 2) * alpha0F[fill3f]
        return alpha_out
    else:  # float treatment
        if nu_m <= nu_c:  # slow cooling
            if (nu <= nu_m):
                alpha_out = (nu / nu_m) ** (-5 / 3.) * alpha0S
            elif (nu_m < nu) and (nu <= nu_c):
                alpha_out = (nu / nu_m) ** (-(p + 4) / 2) * alpha0S
            elif (nu_c < nu):
                alpha_out = (nu_c / nu_m) ** (-(p + 4) / 2) * (nu / nu_c) ** (-(p + 5) / 2) * alpha0S
            else:
                raise ValueError()
        else:  # fast cooling
            if (nu <= nu_c):
                alpha_out = (nu / nu_c) ** (-5 / 3.) * alpha0F
            elif (nu_c < nu) and (nu <= nu_m):
                alpha_out = (nu / nu_c) ** (-3) * alpha0F
            elif (nu_m < nu):
                alpha_out = (nu_m / nu_c) ** (-3) * (nu / nu_m) ** (-(p + 5) / 2) * alpha0F
            else:
                raise ValueError()
        return alpha_out

def SED_joh06(nu, nu_m, nu_c, p, PmaxF, PmaxS):
    kappa1 = 2.37 - 0.3 * p
    kappa2 = 14.7 - 8.68 * p + 1.4 * p ** 2
    kappa3 = 6.94 - 3.844 * p + 0.62 * p ** 2
    kappa4 = 3.5 - 0.2 * p

    kappa13 = -kappa1 / 3.
    kappa12 = kappa1 / 2.
    kappa11 = -1. / kappa1
    kappa2p = kappa2 * (p - 1) / 2.
    kappa12inv = -1. / kappa2
    kappa33 = -kappa3 / 3
    kappa3p = kappa3 * (p - 1) / 2.
    kappa13inv = -1. / kappa3
    kappa42 = kappa4 / 2
    kappa14 = -1. / kappa4

    if hasattr(nu, '__len__'):  # array

        fc = nu_m > nu_c  # fast cooling
        sc = nu_m <= nu_c  # slow cooling

        P_out = np.zeros(len(nu_m))

        P_out[fc] = PmaxF[fc] * \
                    ((nu[fc] / nu_c[fc]) ** (kappa13) + (nu[fc] / nu_c[fc]) ** (kappa12)) ** (kappa11) * \
                    (1 + (nu[fc] / nu_m[fc]) ** (kappa2p)) ** (kappa12inv)

        P_out[sc] = PmaxS[sc] * \
                    ((nu[sc] / nu_m[sc]) ** (kappa33) + (nu[sc] / nu_m[sc]) ** (kappa3p)) ** (kappa13inv) * \
                    (1 + (nu[sc] / nu_c[sc]) ** (kappa42)) ** (kappa14)
        return P_out
    else:  # float
        sc = bool(nu_m <= nu_c)
        fc = bool(nu_m > nu_c)

        if fc:  # fast cooling
            P_out = PmaxF * \
                    ((nu / nu_c) ** (kappa13) + (nu / nu_c) ** (kappa12)) ** (kappa11) * \
                    (1 + (nu / nu_m) ** (kappa2p)) ** (kappa12inv)
        elif sc:  # slow cooling
            P_out = PmaxS * \
                    ((nu / nu_m) ** (kappa33) + (nu / nu_m) ** (kappa3p)) ** (kappa13inv) * \
                    (1 + (nu / nu_c) ** (kappa42)) ** (kappa14)
        else:
            raise ValueError("Neither fast nor slow cooling")
        return P_out

class Synchrotron_Joh06(Synchrotron):

    def __init__(
            self,
            p,
            gamma_min,
            gamma_c,
            B,
            R,
            Gamma,
            ne,
            delta_shock,
            ssa=False
    ):
        """
        :ne: is the comoving number density of electrons
        """

        self.gamma_min =gamma_min
        self.gamma_c = gamma_c
        self.p = p
        self.B = B
        self.ne = ne
        self.delta_shock = delta_shock
        self.ssa = ssa

        phipF = 1.89 - 0.935 * p + 0.17 * p ** 2
        phipS = 0.54 + 0.08 * p
        XpF = 0.455 + 0.08 * p  # for self-absorption
        XpS = 0.06 + 0.28 * p  # for self-absorption

        gamToNuFactor = (3. / (4. * np.pi)) * (cgs.qe * B) / (cgs.me * cgs.c)

        rhoprim = ne * cgs.mppme

        if gamma_min < gamma_c:
            # slow cooling
            self.nu_m = XpS * gamma_min ** 2 * gamToNuFactor
            self.nu_c = XpS * gamma_c ** 2 * gamToNuFactor
            _phip = 11.17 * (p - 1) / (3 * p - 1) * phipS
            self.pprimemax = _phip * cgs.qe ** 3 * ne * B / (cgs.me * cgs.c ** 2)
            # for synch. self absorption
            _alpha = 7.8 * phipS * XpS ** (-(4 + p) / 2.) * (p + 2) * (p - 1) * cgs.qe / cgs.mp / (p + 2 / 3.)
            self.alpha = _alpha * rhoprim * gamma_min ** (-5) / B

        else:
            # fast cooling
            self.nu_m = XpF * gamma_min ** 2 * gamToNuFactor
            self.nu_c = XpF * gamma_c ** 2 * gamToNuFactor
            _phip = 2.234 * phipF
            self.pprimemax = _phip * cgs.qe ** 3 * ne * B / (cgs.me * cgs.c ** 2)
            # for synch. self absorption
            _alpha = 11.7 * phipF * XpF ** (-3) * cgs.qe / cgs.mp
            self.alpha = _alpha * rhoprim * gamma_c ** (-5) / B

        self.Fmax = self.pprimemax * R ** 2 * self.delta_shock * Gamma

        super(Synchrotron_Joh06, self).__init__()

    @classmethod
    def from_obj_fs(cls, electons, dynamics, ssa=False):
        return cls(
            electons.p1,
            electons.get_last("gamma_min"),
            electons.get_last("gamma_c"),
            electons.get_last("B"),
            dynamics.get_last("R"),
            dynamics.get_last("Gamma"),
            dynamics.get_last("rho2") / cgs.mppme,
            dynamics.get_last("thickness"),
            ssa
        )

    def __call__(self, nupime):
        return self.sed(nupime)

    @staticmethod
    def evaluate_tau_ssa(
            nuprime, # Hz
            nu_m, # Gauss
            nu_c, # cm ; 2 * R_b for the blob
            alpha, # object
            delta_shock,
            p
    ):

        if hasattr(nuprime, '__len__') and ( (not hasattr(nu_m, '__len__')) or (not hasattr(alpha, '__len__')) ):
            nu_m = np.full_like(nuprime, nu_m)
            nu_c = np.full_like(nuprime, nu_c)
            alpha = np.full_like(nuprime, alpha)

        alpha_sed = SSA_joh06(nuprime, nu_m, nu_c, p, alpha, alpha)
        alpha_sed *= delta_shock / 2.

        if hasattr(alpha_sed, '__len__'):
            atten=np.ones_like(alpha_sed)
            msc = alpha_sed > 1e-2
            atten[msc] = (1. - np.exp(-alpha_sed[msc])) / alpha_sed[msc]
            msc2 = (1e-8<alpha_sed) & (alpha_sed<1e-2)
            atten[msc2] = \
                (alpha_sed[msc2] - alpha_sed[msc2] ** 2 / 2 + alpha_sed[msc2] ** 4 / 4 - alpha_sed[msc2] ** 6 / 6) / \
                alpha_sed[msc2]
        else:
            if alpha_sed > 1e-2:
                atten = (1. - np.exp(-alpha_sed)) / alpha_sed
            elif 1e-8 < alpha_sed <= 1e-2:
                atten = (alpha_sed - alpha_sed ** 2 / 2 + alpha_sed ** 4 / 4 - alpha_sed ** 6 / 6) / alpha_sed
            else:
                atten = 1.

        return atten

    @staticmethod
    def evaluate_sed(
            nuprime,
            nu_m,
            nu_c,
            Fmax,
            p,
            alpha=None,
            delta_shock=0.,
            ssa=False
    ):
        """
            Based on eqs from https://arxiv.org/abs/astro-ph/0605299
            and https://arxiv.org/abs/1805.05875
            by Johanneson et al
        """
        if hasattr(nuprime, '__len__') and (not hasattr(nu_m, '__len__')):
            nu_m = np.full_like(nuprime, nu_m)
            nu_c = np.full_like(nuprime, nu_c)
            Fmax = np.full_like(nuprime, Fmax)

        spectrum = SED_joh06(nuprime, nu_m, nu_c, p, Fmax, Fmax)

        if ssa:
            atten = Synchrotron_Joh06.evaluate_tau_ssa(
                nuprime,
                nu_m,
                nu_c,
                alpha,
                delta_shock,
                p
            )

            spectrum *= atten

        return nuprime * spectrum

    def sed(self, nuprime):
        return self.evaluate_sed(
            nuprime,
            self.nu_m,
            self.nu_c,
            self.Fmax,
            self.p,
            self.alpha,
            self.delta_shock,
            self.ssa
        )

    def tau_attenuation(self, nuprime):
        return self.evaluate_tau_ssa(
            nuprime,  # Hz
            self.nu_m,  # Gauss
            self.nu_c,  # cm ; 2 * R_b for the blob
            self.alpha,  # object
            self.delta_shock,
            self.p
        )


def R(x):
    """
    Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(x, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)

def nu_synch_peak(B, gamma):
    """
    observed peak frequency for monoenergetic electrons
    Eq. 7.19 in [DermerMenon2009]_
    :B: float
    magnetic field strength in Gauss
    """
    # B = B_to_cgs(B)
    nu_peak = (cgs.qe * B / (2 * np.pi * cgs.me * cgs.c)) * np.power(gamma, 2)
    return nu_peak#.to("Hz")

def calc_x(B, epsilon, gamma):
    """
    ratio of the frequency to the critical synchrotron frequency from
    Eq. 7.34 in [DermerMenon2009]_, argument of R(x),
    note B has to be in cgs Gauss units
    """
    x = (
        4
        * np.pi
        * epsilon
        * np.power(cgs.me, 2)
        * np.power(cgs.c, 3)
        / (3 * cgs.qe * B * cgs.h * np.power(gamma, 2))
    )
    return x#.to_value("")

def single_electron_synch_power(B, epsilon, gamma):
    """
    angle-averaged synchrotron power for a single electron,
    to be folded with the electron distribution
    :B: float
        magnetic field strength in Gauss
    """
    x = calc_x(B, epsilon, gamma)
    prefactor = np.sqrt(3) * np.power(cgs.qe, 3) * B / cgs.h
    return prefactor * R(x)

def tau_to_attenuation(tau):
    """
    Converts the synchrotron self-absorption optical depth to an attenuation
    Eq. 7.122 in [DermerMenon2009]_.
    """
    u = 1 / 2 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
    return np.where(tau < 1e-3, 1., 3 * u / tau)

class Synchrotron_DM06(Synchrotron):
    """Class for synchrotron radiation computation

    Parameters
    ----------
    ssa : bool
        whether or not to consider synchrotron self absorption (SSA).
    integrator : (`~uutils.trapz_loglog`, `~numpy.trapz`)
        function to be used for the integration
	"""

    def __init__(self,
                 electrons,
                 electron_pars,
                 thickness,
                 B,
                 R,
                 Gamma,
                 ssa=False,
                 integrator=np.trapz
                 ):

        self.B = B
        self.R = R
        self.Gamma = Gamma
        self.electrons = electrons
        self.electron_pars = electron_pars
        self.thickness = thickness
        self.ssa = ssa
        self.integrator = integrator

        super(Synchrotron_DM06, self).__init__()


    def __call__(self, nupime):
        return self.sed(nupime)

    @staticmethod
    def evaluate_tau_ssa(
            nuprim, # Hz
            B, # Gauss
            length, # cm ; 2 * R_b for the blob
            n_e, # object
            *args, # for electron distribution
            integrator=np.trapz,
            gamma=gamma_to_integrate, # integration limits
    ):
        """ Computes the syncrotron self-absorption opacity for a general set
            of model parameters, see
            :func:`~agnpy:sycnhrotron.Synchrotron.evaluate_sed_flux`
            for parameters defintion.
            Eq. before 7.122 in [DermerMenon2009]_.
        """
        # conversions
        epsilon = nuprim * cgs.h / cgs.mec2 # = nu_to_epsilon_prime(nu, z, delta_D)
        # multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        SSA_integrand = n_e.evaluate_SSA_integrand(_gamma, *args)
        integrand = SSA_integrand * single_electron_synch_power(B, _epsilon, _gamma)
        integral = integrator(integrand, gamma, axis=0)
        prefactor_k_epsilon = (
                -1 / (8 * np.pi * cgs.me * np.power(epsilon, 2)) * np.power(cgs.lambda_c / cgs.c, 3)
        )
        k_epsilon = (prefactor_k_epsilon * integral)#.to("cm-1")
        return (k_epsilon * length) #.to_value("") # dimensionless

    @staticmethod
    def evaluate_sed(
            nuprim,
            B,
            R,
            Gamma,
            length,
            n_e_obj,
            *args,
            ssa=False,
            integrator=np.trapz,
            gamma=gamma_to_integrate,
    ):
        # conversions
        epsilon = nuprim * cgs.h / cgs.mec2
        # reshape for multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        N_e = 1. * n_e_obj.evaluate(_gamma, *args) # 1. is volume * n_e which is number of electrons
        integrand = N_e * single_electron_synch_power(B, _epsilon, _gamma)
        emissivity = integrator(integrand, gamma, axis=0)
        sed = epsilon * emissivity * (R ** 2 * length * Gamma)

        if ssa:
            tau = Synchrotron_DM06.evaluate_tau_ssa(
                nuprim,
                B,
                R,
                Gamma,
                length,
                n_e_obj,
                *args,
                integrator=integrator,
                gamma=gamma,
            )
            attenuation = tau_to_attenuation(tau)
            sed *= attenuation

        return sed

    def sed(self, nuprime):
        return self.evaluate_sed(
            nuprime,
            self.B,
            self.R,
            self.Gamma,
            self.thickness,
            self.electrons,
            *self.electron_pars,
            ssa=False,
            integrator=np.trapz,
            gamma=gamma_to_integrate,
        )

    def tau_attenuation(self, nuprim):
        tau = Synchrotron_DM06.evaluate_tau_ssa(
            nuprim,
            self.B,
            self.thickness,
            self.electrons,
            *self.electron_pars,
            integrator=np.trapz,
            gamma=gamma_to_integrate,
        )
        return tau_to_attenuation(tau)


''' [below] to be removed [below]'''

# class SynchComov:
#
#     def __init__(
#             self,
#             freqprime=np.logspace(8.,22.)
#     ):
#         self.nu_to_integrate = freqprime
#         self.sed = np.zeros_like(self.nu_to_integrate)
#
#     def compute_sed(self, B, Ne, electrons):
#         gm = electrons.get("gamma_min")
#         gM = electrons.get("gamma_max")
#         gc = electrons.get("gamma_c")
#
# class Radiation:
#     def __init__(
#             self,
#             redshift_z,
#             lum_distance,
#             alpha_obs,
#             freq_obs,
#             theta_offset=None
#     ):
#         self.z = redshift_z
#         self.d_l = lum_distance
#         self.alpha = alpha_obs
#         self.freqs = freq_obs
#         self.theta0 = theta_offset if (not theta_offset is None) else 0.
#
#
#
#     def compute(self):
#         pass
#
# class DermerMenon2009(Radiation):
#
#     def __init__(self):
#
#         super(DermerMenon2009, self).__init__()
#
#     @staticmethod
#     def R(x):
#         """
#         Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the
#         approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
#         """
#         term_1_num = 1.808 * np.power(x, 1 / 3)
#         term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
#         term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
#         term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
#         return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)
#
#     @staticmethod
#     def nu_synch_peak(B, gamma):
#         """
#         observed peak frequency for monoenergetic electrons
#         Eq. 7.19 in [DermerMenon2009]_
#         :B: float
#         magnetic field strength in Gauss
#         """
#         # B = B_to_cgs(B)
#         nu_peak = (cgs.qe * B / (2 * np.pi * cgs.me * cgs.c)) * np.power(gamma, 2)
#         return nu_peak  # .to("Hz")
#
#     @staticmethod
#     def calc_x(B, epsilon, gamma):
#         """
#         ratio of the frequency to the critical synchrotron frequency from
#         Eq. 7.34 in [DermerMenon2009]_, argument of R(x),
#         note B has to be in cgs Gauss units
#         """
#         x = (
#                 4
#                 * np.pi
#                 * epsilon
#                 * np.power(cgs.me, 2)
#                 * np.power(cgs.c, 3)
#                 / (3 * cgs.qe * B * cgs.h * np.power(gamma, 2))
#         )
#         return x  # .to_value("")
#
#     @staticmethod
#     def single_electron_synch_power(B, epsilon, gamma):
#         """
#         angle-averaged synchrotron power for a single electron,
#         to be folded with the electron distribution
#         :B: float
#             magnetic field strength in Gauss
#         """
#         x = calc_x(B, epsilon, gamma)
#         prefactor = np.sqrt(3) * np.power(cgs.qe, 3) * B / cgs.h
#         return prefactor * R(x)
#
#     @staticmethod
#     def tau_to_attenuation(tau):
#         """
#         Converts the synchrotron self-absorption optical depth to an attenuation
#         Eq. 7.122 in [DermerMenon2009]_.
#         """
#         u = 1 / 2 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
#         return np.where(tau < 1e-3, 1., 3 * u / tau)
#
# class WSPN99():
#     """
#         ALL quantities here are primed -- in comoving frame
#         'pmax', 'nu_min', 'nu_c' 'Xintp', 'PhiPint' are from WIJERS 1999, https://arxiv.org/abs/astro-ph/9805341
#         All quantities are in comoving frame of refence
#     """
#
#     def __init__(self, p):
#
#         self.Xp = self.Xintp(p) # dimensionless spectral peak from WIJERS 1999
#         self.PhiP = self.PhiPint(p)
#
#     @staticmethod
#     def Xintp(pp):
#         data_xp = [
#             1.4444980694980694, 0.9944136261210208,
#             1.4820641682054725, 0.9951191166713027,
#             1.4820641682054725, 0.9724149343868285,
#             1.4927973392647305, 0.9572435579224101,
#             1.5035305103239884, 0.9420721814579918,
#             1.5196302669128754, 0.9230991471421099,
#             1.5410966090313916, 0.9070816520832391,
#             1.5536186419338591, 0.8936162230627284,
#             1.5840292932684235, 0.8763981013871924,
#             1.6194487577639751, 0.8477393594942589,
#             1.6001290498573106, 0.863911976295446,
#             1.6484283196239717, 0.8286159040949189,
#             1.6698946617424877, 0.8123281211517009,
#             1.7074607604498908, 0.7972487992565946,
#             1.7128273459795198, 0.7808337068026454,
#             1.7342936880980357, 0.7650864996281136,
#             1.7632732499580324, 0.746467581612873,
#             1.7986927144535838, 0.731088984437128,
#             1.825525642101729, 0.7161350134448954,
#             1.846991984220245, 0.7033609729980923,
#             1.889924668457277, 0.688534311516599,
#             1.922124181635051, 0.6747339605524508,
#             1.954323694812825, 0.6609336095883027,
#             1.975790036931341, 0.647438801449929,
#             2.0187227211683734, 0.6351348268889329,
#             2.0562888198757765, 0.6223979999914229,
#             2.0992215041128084, 0.6092831617774097,
#             2.1421541883498403, 0.5972494751007525,
#             2.1904534581165014, 0.5847657003384353,
#             2.2387527278831625, 0.5697232002710423,
#             2.2977851687090816, 0.5576005927976534,
#             2.360037560852778, 0.5433411441517004,
#             2.421216635890549, 0.530382602844718,
#             2.4856156622460968, 0.5171972547478341,
#             2.550014688601645, 0.5050930581883057,
#             2.609047129427564, 0.49560125278914946,
#             2.678812741312741, 0.4848488873728878,
#             2.7421384505623636, 0.47591959415041374,
#             2.8076107940238373, 0.4680283630924301,
#             2.8709365032734593, 0.4603604133302047,
#             2.9374821638408593, 0.45062769916457235,
#             3.004027824408259, 0.4440303244572723,
#             3.061986948128252, 0.4389747658554187,
#             3.1371191455430583, 0.43224107204123463,
#             3.199371537686755, 0.42706329630907136,
#             3.2659171982541544, 0.42100649737044926,
#             3.3254862976330366, 0.4160537656811161,
#             3.3947152509652505, 0.409916076237978,
#             3.4542843503441327, 0.4048191910103307,
#             3.5278065721000504, 0.39895570666724656,
#             3.6018654524089304, 0.39424368788360376,
#             3.6662644787644783, 0.3906085117000302,
#             3.7306635051200265, 0.3871535274393493,
#             3.7896959459459456, 0.3834458827650472,
#             3.858388240725197, 0.37884158741356844,
#             3.9335204381400026, 0.3742701966740961,
#             4.004359367131106, 0.3703075412567437,
#             4.068758393486654, 0.36766342064907986,
#             4.138524005371831, 0.36473138468635924,
#             4.201849714621453, 0.3623610774571778,
#             4.272688643612556, 0.3594255160003136,
#             4.3370876699681045, 0.35705168327698866,
#             4.406853281853282, 0.3549305109672849,
#             4.476618893738458, 0.3511876113515472,
#             4.541017920094006, 0.3488137786282224,
#             4.610783531979184, 0.34588174266550165,
#             4.676255875440658, 0.3432881541405619,
#             4.744948170219908, 0.3403055860951171,
#             4.803980611045828, 0.33821966872684883,
#             4.8791128084606346, 0.3361709649078737,
#             4.932778663756924, 0.33293142751152094,
#             5.0025442756421015, 0.33032373701000706,
#             5.093776229645794, 0.3280779972405644,
#             5.190374769179117, 0.3257785916158257,
#             5.286973308712438, 0.3231188021453014,
#             5.3835718482457615, 0.3202788207518845,
#             5.480170387779083, 0.31815960705003843,
#             5.576768927312406, 0.3156800095024068,
#             5.673367466845727, 0.31320041195477544,
#             5.7699660063790486, 0.31072081440714383,
#             5.866564545912372, 0.30824121685951245,
#             5.952429914386435, 0.6841999123278875,
#             5.957796499916064, 0.3053467861676563
#         ]
#         data_xp = np.reshape(np.array(data_xp), (int(len(data_xp) / 2), 2))
#         return interpolate.interp1d(data_xp[:, 0], data_xp[:, 1])(pp)
#
#     @staticmethod
#     def PhiPint(pp):
#         data_phip = [
#             1.0062166154656431, 0.4114136714112666,
#             1.0451629272536156, 0.41748522011949685,
#             1.1309890587863694, 0.4352108868963911,
#             1.2168151903191229, 0.4539524485203539,
#             1.2572039580992425, 0.46103900983957213,
#             1.3531272815770259, 0.48129823808157235,
#             1.4641963929723543, 0.5034106767310167,
#             1.5860839243087863, 0.5251294910490668,
#             1.620702868120317, 0.5319293256420172,
#             1.696431807708041, 0.5467827993381156,
#             1.8175981110483992, 0.5661800094094774,
#             1.8579868788285185, 0.5703882019953342,
#             1.958958798278817, 0.5845489733286391,
#             2.090222293564205, 0.6011745162193037,
#             2.2315829807946232, 0.6141818128900468,
#             2.277020344547257, 0.6178789702260863,
#             2.3852045439582916, 0.6268107283083477,
#             2.42342962775019, 0.6289642665816388,
#             2.5344987391455187, 0.6375314406034999,
#             2.584984698870668, 0.6420720891524804,
#             2.6859566183209664, 0.6459045962072529,
#             2.7364425780461157, 0.6483287971581736,
#             2.8303464631348936, 0.6522841482352335,
#             2.8828518612490486, 0.6548933614442702,
#             2.9989695686168916, 0.6609515736837455,
#             3.044406932369526, 0.6610930990550443,
#             3.1605246397373694, 0.6649163426309681,
#             3.205962003490004, 0.6670896576964045,
#             3.311982518912817, 0.6690566030386015,
#             3.367517074610481, 0.6708004529318601,
#             3.473537590033295, 0.6738679510250482,
#             3.518974953785929, 0.6750253712434158,
#             3.6350926611537724, 0.6769861409330469,
#             3.680530024906407, 0.6779742453435698,
#             3.7966477322742502, 0.6801043308410457,
#             3.842085096026884, 0.6809231194437237,
#             3.948105611449698, 0.6828900647859206,
#             3.998591571174847, 0.6846370025054622,
#             4.109660682570175, 0.6848230440390058,
#             4.155098046322809, 0.6858111484495286,
#             4.271215753690653, 0.6867560232920911,
#             4.316653117443288, 0.6867282328555452,
#             4.4226736328661005, 0.6890338098134317,
#             4.480227626952771, 0.6906917666722547,
#             4.586392387974799, 0.6928037516261392,
#             4.639763259684242, 0.6922196231170257,
#             4.738715740745535, 0.6935813545077774,
#             4.79122113885969, 0.6937524200838491,
#             4.892193058309989, 0.6939954320123121,
#             4.948737333202156, 0.6940624378426506,
#             5.02682228424372, 0.6961141954800841,
#             5.089088301238071, 0.6947215858265036,
#             5.144622856935735, 0.6963807778158397,
#             5.250643372358549, 0.6964852492717439,
#             5.285983544166153, 0.6962943186799191,
#             5.392004059588967, 0.6963987901358233,
#             5.442490019314116, 0.6967065434886839,
#             5.51821895890184, 0.6978454367493543,
#             5.573753514599504, 0.6973035232367082,
#             5.609093686407109, 0.698128487491952,
#             5.7151142018299215, 0.6980636431400113,
#             5.772668195916593, 0.6982316208898003,
#             5.8766692729504, 0.6979648326989591,
#             5.937252424620579, 0.6979277787835645
#         ]
#         data_phip = np.reshape(np.array(data_phip), (int(len(data_phip) / 2), 2))
#         return interpolate.interp1d(data_phip[:, 0], data_phip[:, 1])(pp)
#
#     def nu_min(self, gamma_min, B):
#         """ the charactersitic frequency: minimum
#             nu = (Gamma * (1. - beta)) ** (-1) * nu_prime
#         """
#         nu_m = 3./ (4. * np.pi) * self.Xp * gamma_min ** 2. * cgs.qe * B / (cgs.me * cgs.c)
#         return nu_m
#
#     def nu_c(self, gamma_c, B):
#         """ the characteristic frequency, cooling
#             nu = (Gamma*(1.-beta))**(-1.) * nu_prime
#         """
#         nu_c   = 0.286 * 3. * gamma_c**2. * cgs.qe * B/(4.*np.pi*cgs.me*cgs.c)
#         return nu_c
#
#     def pmax(self, Ne, B):
#         # pmax = Ne * cgs.me * cgs.c ** 2 * cgs.sigmaT * B / (3 * cgs.qe) # SPN 97
#         pmax = self.PhiP * np.sqrt(3) * Ne * cgs.qe ** 3 * B / (cgs.me * cgs.c ** 2) # Widges 99
#         return pmax
#
#
#     @staticmethod
#     def SED_SariPiran97(nu, nu_m, nu_c, p, PmaxF, PmaxS):
#         """
#         Following Sari & Piran 1997
#         :param nu:
#         :param nu_m:
#         :param nu_c:
#         :param p:
#         :return:
#         """
#
#         if hasattr(nu, '__len__'):  # array treatment
#             res = np.zeros(len(nu))
#             sc = np.array(nu_m <= nu_c, dtype=bool)  # slow cooling
#             fc = np.array(nu_m > nu_c, dtype=bool)  # fast cooling
#             if len(sc) > 0:  # Slow cooling
#                 fil1 = sc & (nu < nu_m)
#                 fil2 = sc & ((nu >= nu_m) & (nu < nu_c))
#                 fil3 = sc & (nu >= nu_c)
#                 res[fil1] = (nu[fil1] / nu_m[fil1]) ** (1. / 3.) * PmaxS[fil1]
#                 res[fil2] = (nu[fil2] / nu_m[fil2]) ** (-1. * (p - 1.) / 2.) * PmaxS[fil2]
#                 res[fil3] = (nu_c[fil3] / nu_m[fil3]) ** (-1. * (p - 1.) / 2.) * \
#                             (nu[fil3] / nu_c[fil3]) ** (-1. * p / 2.) * PmaxS[fil3]
#             if len(fc) > 0:  # fast cooling
#                 fil1 = fc & (nu < nu_c)
#                 fil2 = fc & ((nu >= nu_c) & (nu < nu_m))
#                 fil3 = fc & (nu >= nu_m)
#                 res[fil1] = (nu[fil1] / nu_c[fil1]) ** (1. / 3.) * PmaxF[fil1]
#                 res[fil2] = (nu[fil2] / nu_c[fil2]) ** (-1. / 2.) * PmaxF[fil2]
#                 res[fil3] = (nu_m[fil3] / nu_c[fil3]) ** (-1. / 2.) * (nu[fil3] / nu_m[fil3]) ** (-p / 2.) * PmaxF[fil3]
#             return res
#         else:  # flaot treatment
#             if nu_m <= nu_c:  # slow cooling
#                 res = 0
#                 if (nu < nu_m): res = (nu / nu_m) ** (1. / 3.)
#                 if (nu >= nu_m) & (nu < nu_c): res = (nu / nu_m) ** (-1. * (p - 1.) / 2.)
#                 if (nu >= nu_c): (nu / nu_m) ** (-1. * (p - 1.) / 2.) * (nu / nu_c) ** (-1. * p / 2.)
#                 res *= PmaxS
#             else:  # fast cooling
#                 res = 0
#                 if (nu < nu_c): res = (nu / nu_c) ** (1. / 3.)
#                 if (nu >= nu_c) & (nu < nu_m): res = (nu / nu_c) ** (-1. / 2.)
#                 if (nu >= nu_m): res = (nu_m / nu_c) ** (-1. / 2.) * (nu / nu_m) ** (-p / 2.)
#                 res *= PmaxF
#             return res


# """
#
# """
#
# import numpy as np
# from uutils import cgs
#
# from dynamics import Peer_rhs
#
# class SimplePlasma:
#
#     def __init__(
#             self,
#             p,
#             eps_e,
#             eps_b,
#             key_dic
#     ):
#         self.key_dic = key_dic
#         self.eps_e = eps_e
#         self.eps_b = eps_b
#         self.p = p
#
#         self.gamma_min = np.zeros(1)
#         self.gamma_c = np.zeros(1)
#         self.gamma_max = np.zeros(1)
#         self.B = np.zeros(1)
#
#     @classmethod
#     def from_obj(cls, shell, **kwargs):
#         if "rs" in kwargs.keys() and kwargs["rs"] == True:
#             key_dic = {
#                 "Gamma": "Gamma",
#                 "GammaSh": "Gamma43",
#                 "U_e": "U_e_RS",
#             }
#             return cls(shell.p_rs, shell.eps_e_rs, shell.eps_b_rs, key_dic)
#         else:
#             key_dic = {
#                 "Gamma": "Gamma",
#                 "GammaSh": "Gamma",
#                 "U_e": "U_e",
#             }
#             return cls(shell.p, shell.eps_e, shell.eps_b, key_dic)
#
#
#     def get_B(self, U_b):
#         # magnetic field in the plasma
#         return np.sqrt(8. * np.pi * U_b)
#
#     def get_gamma_min(self, Gamma):
#         return cgs.mp / cgs.me * (self.p - 2.) / (self.p - 1.) * self.eps_e * (Gamma - 1.)
#
#     def get_gamma_c(self, Gamma, tt, B):
#         return 6. * cgs.pi * cgs.me * cgs.c / (cgs.sigmaT * Gamma * tt * np.power(B,2.))
#
#     def get_gamma_max(self, B):
#         return (6. * np.pi * cgs.qe / cgs.sigmaT / B) ** .5
#
#     def evaluate(self, get_func):
#
#         GammaSh = get_func(self.key_dic["GammaSh"])[-1]
#         Gamma = get_func(self.key_dic["Gamma"])[-1]
#         tt = get_func("tt")[-1]
#         U_e = get_func(self.key_dic["U_e"])[-1]
#         U_b = self.eps_b * U_e
#
#         self.B = np.append(self.B, self.get_B(U_b)) # U_b_rs
#         self.gamma_min = np.append(self.gamma_min, self.get_gamma_min(GammaSh)) # Gamma43
#         self.gamma_c = np.append(self.gamma_c, self.get_gamma_c(Gamma, tt, self.B[-1])) # BRS
#         self.gamma_max = np.append(self.gamma_max, self.get_gamma_max(self.B[-1]))
#
#     def get(self, v_n):
#         if v_n == "gamma_min":
#             return self.gamma_min
#         elif v_n == "gamma_max":
#             return self.gamma_max
#         elif v_n == "gamma_c":
#             return self.gamma_c
#         elif v_n == "B":
#              return self.B
#
#
#     @property
#     def parameters(self):
#         return [self.B, self.p, self.gamma_min, self.gamma_c, self.gamma_max]
#
#     def __str__(self):
#         return (
#                 f"* plasma properties \n"
#                 + f" - B: {self.B:.2e}\n"
#                 + f" - p: {self.p:.2f}\n"
#                 + f" - gamma_min: {self.gamma_min:.2e}\n"
#                 + f" - gamma_c: {self.gamma_c:.2e}\n"
#                 + f" - gamma_max: {self.gamma_max:.2e}\n"
#         )
#
#
# class PlasmaSimple:
#
#     def __init__(
#             self,
#             Gamma,
#             rho,
#             U_e,
#             tt,
#             eps_e,
#             eps_b,
#             p
#     ):
#
#
#
#         # self.Gamma = Gamma
#         # self.rho = rho
#         # self.tt = tt
#         # self.U_e = U_e
#         # self.eps_e = eps_e
#         # self.eps_b = eps_b
#         # self.p = p
#
#         self.U_e = U_e
#         self.U_b = eps_b * U_e
#
#         self.B = self.get_B(self.U_b)
#         self.gamma_min = self.get_gamma_min(Gamma, p, eps_e)
#         self.gamma_c = self.get_gamma_c(Gamma, tt, self.B)
#         self.gamma_max = self.gamma_max(self.B)
#
#
#
#     def get_B(self, U_b):
#         # magnetic field in the plasma
#         return np.sqrt(8. * np.pi * U_b)
#
#     def get_gamma_min(self, Gamma, p, eps_e):
#         return cgs.mp / cgs.me * (p - 2.) / (p - 1.) * eps_e * (Gamma - 1.)
#
#     def get_gamma_c(self, Gamma, tt, B):
#         return 6. * cgs.pi * cgs.me * cgs.c / (cgs.sigmaT * Gamma * tt * B ** 2.)
#
#     def get_gamma_max(self, B):
#         return (6. * np.pi * cgs.qe / cgs.sigmaT / B) ** .5
#
#     @classmethod
#     def from_no_energy(
#             cls,
#             Gamma,
#             rho,
#             tt,
#             eps_e,
#             eps_b,
#             p
#     ):
#         # if energy is not given, compute
#         nn = rho / cgs.mppme
#         beta = np.sqrt(1. - np.power(float(Gamma), -2))
#         TT = Peer_rhs.normT(Gamma, beta)
#         ada = Peer_rhs.adabatic_index(TT)
#         eT = (ada * Gamma + 1.) / (ada - 1) * (Gamma - 1.) * nn * cgs.mp * cgs.c ** 2.
#
#         U_e = eT  # assumption
#
#         return cls(Gamma, rho, U_e, tt, eps_e, eps_b, p)
#
#
# class PlasmaP:
#
#     def __init__(
#             self,
#             Gamma,
#             rho,
#             eps_e,
#             eps_b,
#     ):
#         # if energy is not given, compute
#         nn = rho / cgs.mppme
#         beta = np.sqrt(1. - np.power(float(Gamma), -2))
#         TT = Peer_rhs.normT(Gamma, beta)
#         ada = Peer_rhs.adabatic_index(TT)
#         eT = (ada * Gamma + 1.) / (ada - 1) * (Gamma - 1.) * nn * cgs.mp * cgs.c ** 2.
#
#         U_e = eT # assumption
#
#         U_b = eps_b * U_e
#         B = np.sqrt(8. * np.pi * U_b) * cgs.c  # magnetic field strength
#
#     def gamma_min(self):
#         return cgs.mp / cgs.me * (pp - 2.) / (pp - 1.) * epsE * (Gam - 1.)
#
#     def gamma_max(self):
#         return 1e8
#
#     def gamma_c(self):
#         return 6. * cgs.pi * cgs.me * cgs.c / (cgs.sigmaT * Gam * tt * B ** 2.)
#
# class PlasmaN:
#
#     def __init__(
#             self,
#             Gamma,
#             rho,
#             Eint,
#             eps_b
#     ):
#
#         rhoprim = 4. * rho * Gamma # comoving density
#         V2 = M2 / rhoprim # comoving volume
#         U_e = Eint / V2 # comoving energy density (electrons)
#         U_b = eps_b * U_e # comoving energy density (MF)
#         B = np.sqrt(8. * np.pi * U_b) #* cgs.c # magnetic field strength
#
#
#
#
#     def get_B(self):
#
#         rhoprim = 4. * rho * Gamma  # comoving density
#         V2 = M2 / rhoprim  # comoving volume
#         U_e = Eint2 / V2  # comoving energy density (electrons)
#         U_b = eps_b * U_e  # comoving energy density (MF)
#         B = np.sqrt(8. * np.pi * U_b) * cgs.c  # magnetic field strength
#
#     def get_gamma_min(self):
#         """
#             minimumum injected lorentz factor
#             Kumar 2014 : arXiv:1410.0679 eq.64 & eq.65
#         """
#         assert p != 2.
#         gp = (p - 2) / (p - 1)
#         nppe = 1. # np / me difference in number density
#         (p - 2) / (p - 1) * (1 + cgs.mp / cgs.me * epsilone * (Gamma - 1.)) * nppe
#
#     def get_gamma_c(self):
#         """ compute weighted cooling lorentz factor """
#         gamma_c_w_fac = 6 * np.pi * cgs.me * cgs.c / cgs.sigmaT
#         dm2 = np.diff(M2[:i])
#         gamma_c_w_array = gamma_c_w_fac / B[:i] ** 2 / tcomoving[:i]
#         gamma_c_w = np.sum(dm2 * gamma_c_w_array) / (M2[i] - M2[0])
#         return gamma_c_w
#
#     def get_gamma_max(self, B):
#         """ maximum lorentz factor """
#         return (6. * np.pi * cgs.qe / cgs.sigmaT / B) ** .5
#
#
