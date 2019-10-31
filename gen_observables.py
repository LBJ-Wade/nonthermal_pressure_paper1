import sys
import numpy as np
import colossus
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_so, profile_nfw, mass_defs
from pathlib import Path
from os.path import expanduser
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from os.path import isfile

print("Finished imports", flush=True)

# constants and directories:

cosmo_to_evcm3 = 4.224e-10
mu_plasma = 0.6
Gmp_by_kB = 5.21e-4 # in kpc * K / Msun
boltzmann = 8.617e-8 # in keV per Kelvin
sigmaT_by_mec2 = 1.697E-18 # kpc^2 s^2 / Msun / km^2
mp_kev_by_kms2 = 1.044E-5 # in KeV / (km/s)^2
G = colossus.utils.constants.G
cm_per_km = 1e5
km_per_kpc = colossus.utils.constants.KPC / cm_per_km # KPC was in cm
s_per_Gyr = colossus.utils.constants.GYR
yr_per_Gyr = 1E9

home_dir = Path(expanduser('~'))
multimah_root = home_dir / 'scratch60/frank_mah/output'

# TUNABLE PARAMETERS

beta_def = 1.0
eta_def  = 0.7
Nmah = 9999
Nradii = 500
N_r500_mult = 20
zi=6.
zobs = 0.0

# adding additional cosmologies based on conversation with Daisuke

# four extremes with low/high Om0 and sigma8
# note that the Omega_Lambda is changed accordingly by varying Om0
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['Om0'] = 0.1
cosmology.addCosmology('planck18_vlO', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['Om0'] = 0.5
cosmology.addCosmology('planck18_vhO', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['sigma8'] = 0.5
cosmology.addCosmology('planck18_vlS', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['sigma8'] = 1.2
cosmology.addCosmology('planck18_vhS', fiducial_params)


# six low/high Om0, sigma8, H0
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['Om0'] = 0.25
cosmology.addCosmology('planck18_lO', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['Om0'] = 0.35
cosmology.addCosmology('planck18_hO', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['sigma8'] = 0.7
cosmology.addCosmology('planck18_lS', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['sigma8'] = 0.9
cosmology.addCosmology('planck18_hS', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['H0'] = 65
cosmology.addCosmology('planck18_lH', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['H0'] = 75
cosmology.addCosmology('planck18_hH', fiducial_params)

cname = sys.argv[1] # load in cosmology name from the ones above
cosmo = cosmology.setCosmology(cname)
cbf = cosmo.Ob0 / cosmo.Om0

radii_definitions = [('vir', 1), ('500c', 1), ('500c', 2), ('500c', 3), ('500c', 4), ('500c', 5),
                     ('200m', 0.5), ('200m', 0.875), ('200m', 1.0), ('200m', 1.25), ('200m', 1.625),
                     ('200m', 2.0)]


def zhao_vdb_conc(t, t04):
    return 4.0 * (1.0 + (t / (3.40*t04))**6.5)**(1.0/8.0)

# CAN CHANGE THIS IF WE IMPLEMENT LUDLOW
conc_model=zhao_vdb_conc

def nfw_prof(r, rhos, rs):
    return rhos / ((r/rs) * (1. + r/rs)**2)

def NFWf(x):
    return np.log(1. + x) - x/(1. + x)

def NFWM(r, M, z, c, R):
    return M * NFWf(c*r/R) / NFWf(c)

def t_d(r, M, z, c, R, beta=beta_def):
    Menc = NFWM(r, M, z, c, R)
    t_dyn = 2. * np.pi * (r**3 / (G*Menc))**(1./2.) * km_per_kpc / (cosmology.getCurrent().H0 / 100.)
    return beta * t_dyn / s_per_Gyr / 2.

def Gamma(c_nfw):
    return 1.15 + 0.01*(c_nfw - 6.5)

def eta0(c_nfw):
    return 0.00676*(c_nfw - 6.5)**2 + 0.206*(c_nfw - 6.5) + 2.48

def multimah_multiM(z_obs, cosmo, Nmah):
    # loads in an array of MAH from Frank's MAH code, specify Nmah = number of MAH to get
    mah_dir = multimah_root / ('%s' % (cosmo.name))
    fn = mah_dir / 'mah_data.npz'
    if(isfile(fn)):
        d = np.load(fn)
        return d['dat'], d['redshifts'], d['lbtime'], d['masses']
    else:
        masses = 10**np.loadtxt(mah_dir / 'halomasses.dat')[:Nmah]
        dat1 = np.loadtxt(mah_dir / 'MAH0001.dat')
        redshifts = dat1[:,1]
        lbtime = dat1[:,2]
        nz = len(dat1)
        dat = np.zeros((Nmah, nz))
        std = np.zeros((nz,2))
        for i in range(0,Nmah):
            dat[i,:] = 10**np.loadtxt(mah_dir / ('MAH%04d.dat' %(i+1)), usecols=3) * masses[i]
        np.savez(fn, dat=dat, redshifts=redshifts, lbtime=lbtime, masses=masses)
        return dat, redshifts, lbtime, masses

def sig2_tot(r, M, c, R):
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gamma(c) - 1.) / Gamma(c)) * 3. *eta0(c)**-1 * (phi0 - phir)
    return (1.0 / rho0_by_P0) * theta

def p_2_y(r,p):
    '''
    Discrete Integration for y, calculate int P dl, l=np.sqrt(r3d^2-r2d^2).
    If P is in unit of P200m, r in unit of R200m, y is in unit of P200m*R200m.
    r is 3D radius (r3d), P is pressure, ind is index of 2D radius array (r2d) you want y value for.
    Assume 2D radius array is same as 3D radius. Assume equal log space.
    '''
    yv = np.zeros(len(r)-1)
    dlogr = np.log(r[2]/r[1])
    for i in range(0,len(r)-1):
        yv[i] = np.sum(p[i+1:]*r[i+1:]**2*dlogr/np.sqrt(r[i+1:]**2-r[i]**2))
    return sigmaT_by_mec2 * yv # this is in units of h

# This outputs in units of kpc^2 / h, standard unit is Mpc^2, verified magnitudes
def YSZ(yprof, rads, Rx):
    # interpolate the yprof
    yprof_interp = interp(rads, yprof, k=3)
    Y = 2.0 * np.pi * quad(lambda x: yprof_interp(x) * x, 0, Rx)[0]
    return Y

def gen_obs(cosmo, beta=beta_def, eta=eta_def):

    mah, redshifts, lbtime, masses = multimah_multiM(zobs, cosmo, Nmah)
    print("Loaded MAH", flush=True)
    zi_snap = np.where(redshifts <= zi)[0][-1] + 1 #first snap over z=6
    t0 = cosmo.age(0) # this way we can easily get proper times using the lookback times from Frank's files

    n_steps = zi_snap
    
    rads = np.logspace(np.log10(0.01),np.log10(N_r500_mult), Nradii) # y_SZ goes out to 20x R_500c for LOS integration

    ds2dt    = np.zeros((n_steps, Nradii))
    sig2tots = np.zeros((n_steps, Nradii))
    sig2nth  = np.zeros((n_steps, Nradii))
    cvirs    = np.zeros(Nmah)
    # The values that we will return and column_stack
    YSZv     = np.zeros((Nmah, len(radii_definitions)))
    Tmgasv   = np.zeros((Nmah, len(radii_definitions)))
    Mgasv    = np.zeros((Nmah, len(radii_definitions)))
    mass_enc = np.zeros((Nmah, len(radii_definitions)))

    for mc in range(0,Nmah):
        if(mc % 100 == 0):
            print(mc, flush=True)
        # get cvir so that we can get Rdef
        t04_ind = np.where(mah[mc,:] > 0.04*masses[mc])[0][-1]
        cvir = conc_model(t0 - lbtime[0], t0 - lbtime[t04_ind])
        Mdf, Rdef, _ = mass_defs.changeMassDefinition(masses[mc], c=cvir, z=zobs, mdef_in='vir', mdef_out='500c')
        rds  = rads*Rdef #convert to physical units; using r500c, this goes out to 20x R500c
        # doing it this way ensures that we're using the same fractional radii for each cluster

        # integrate time to z=0 in order to get f_nth profile
        for i in range(zi_snap,0,-1):
            z_1 = redshifts[i] #first redshift
            z_2 = redshifts[i-1] #second redshift, the one we are actually at
            dt = lbtime[i] - lbtime[i-1] # in Gyr
            mass_1 = mah[mc, i] #dat = np.zeros((Nmah, nz))
            mass_2 = mah[mc, i-1]
            dM = mass_2 - mass_1
            dMdt = dM/dt # since the (i+1)th is computed between i+1 and i
            Rvir_1 = mass_so.M_to_R(mass_1, z_1, 'vir')
            Rvir_2 = mass_so.M_to_R(mass_2, z_2, 'vir')

            time_1 = t0 - lbtime[i]
            time_2 = t0 - lbtime[i-1]
            m04_1  = 0.04 * mass_1
            m04_2  = 0.04 * mass_2
            t04_ind_1 = np.where(mah[mc,:] > m04_1)[0][-1]
            t04_ind_2 = np.where(mah[mc,:] > m04_2)[0][-1]
            t04_1 = t0 - lbtime[t04_ind_1]
            t04_2 = t0 - lbtime[t04_ind_2]

            c_1 = conc_model(time_1, t04_1)
            c_2 = conc_model(time_2, t04_2)
            if(i==1): # final
                cvirs[mc] = c_2
                assert cvirs[mc] == cvir
            sig2tots[i-1,:] = sig2_tot(rds, mass_2, c_2, Rvir_2) # this function takes radii in physical kpc/h
            if(i==zi_snap):
                ds2dt[i-1,:] = (sig2tots[i-1,:] - sig2_tot(rds, mass_1, c_1, Rvir_1)) / dt # see if this works better, full change
                sig2nth[i-1,:] = eta * sig2tots[i-1,:] # starts at z_i = 6 roughly
            else:
                ds2dt[i-1,:] = (sig2tots[i-1,:] - sig2tots[i,:]) / dt
                td = t_d(rds, mass_2, z_2, c_2, Rvir_2, beta=beta_def) #t_d at z of interest z_2
                sig2nth[i-1,:] = sig2nth[i,:] + ((-1. * sig2nth[i,:] / td) + eta * ds2dt[i-1,:])*dt
                sig2nth[i-1, sig2nth[i-1,:] < 0] = 0 #can't have negative sigma^2_nth at any point in time
        fnth = sig2nth[0,:] / sig2tots[0,:]
        # Now, we have fnth, so we can compute the pressure profile and use it to compute the thermal pressure profile
        Rvir = mass_so.M_to_R(masses[mc], zobs, 'vir')
        assert Rvir == Rvir_2 # the final one, it should

        # compute rho_gas profile, use it to compute M_gas within Rdef and T_mgas within Rdef
        rho0_by_P0 = 3*eta0(cvirs[mc])**-1 * Rvir/(G*masses[mc])
        phi0 = -1. * (cvirs[mc] / NFWf(cvirs[mc]))
        phir = lambda rad: -1. * (cvirs[mc] / NFWf(cvirs[mc])) * (np.log(1. + cvirs[mc]*rad/Rvir) / (cvirs[mc]*rad/Rvir))
        theta = lambda rad: 1. + ((Gamma(cvirs[mc]) - 1.) / Gamma(cvirs[mc])) * 3. *eta0(cvirs[mc])**-1 * (phi0 - phir(rad))

        rho0_nume = nume = cbf * masses[mc]
        rho0_denom = 4. * np.pi * quad(lambda x: theta(x)**(1.0 / (Gamma(cvirs[mc]) - 1.0)) * x**2, 0, Rvir)[0]
        rho0 = rho0_nume / rho0_denom
        rhogas = lambda rad: rho0 * theta(rad)**(1.0 / (Gamma(cvirs[mc]) - 1.0))

        Tg = mu_plasma * mp_kev_by_kms2 * (1. - fnth) * sig2tots[0,:]
        Tgf = interp(rds, Tg) # interpolator for Tgas

        # for computing the enclosed mass out to arbitrary radii
        rhos, rs = profile_nfw.NFWProfile.fundamentalParameters(masses[mc], cvir, zobs, 'vir')

        Ptot = rhogas(rds) * sig2tots[0,:]
        Pth  = Ptot * (1.0 - fnth)
        # compute ySZ profile
        yprof = p_2_y(rds, Pth)


        ### BELOW HERE IS WHERE WE CAN LOOP OVER DIFFERENT RADII ####

        # Loop over Rdef values, make them tuples
        for itR in range(0,len(radii_definitions)):
            mdef, mult = radii_definitions[itR]
            Mdf, Rdef, _ = mass_defs.changeMassDefinition(masses[mc], c=cvir, z=zobs, mdef_in='vir', mdef_out=mdef)
            Rdef = mult*Rdef

            # integrate ySZ profile out to Rdef
            YSZv[mc, itR] = YSZ(yprof, rds[:-1], Rdef) # uses an interpolator
            Mgasv[mc, itR] = 4.0 * np.pi * quad(lambda x: rhogas(x) * x**2, 0, Rdef)[0]
            Tweighted = 4. * np.pi * quad(lambda x: Tgf(x) * rhogas(x) * x**2, 0, Rdef)[0]
            Tmgasv[mc, itR] = Tweighted/Mgasv[mc, itR]
            mass_enc[mc, itR] = quad(lambda x: 4. * np.pi * x**2 * nfw_prof(x, rhos, rs), 0, Rdef)[0]


    return np.stack((mass_enc, Tmgasv, Mgasv, YSZv))
    # the masses should be same as Mvirs and they're the same for all cosmologies anyway

print("Finished load-in stuff", flush=True)
    
data = gen_obs(cosmo, beta=beta_def, eta=eta_def)
np.save('%s_data.npy' % cname, data)
