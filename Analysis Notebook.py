#!/usr/bin/env python
# coding: utf-8

# # Modeling the nonthermal pressure fraction and $Y_\mathrm{SZ}-M$ relation

# ## Imports and cosmologies

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import colossus
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import concentration, mass_so, profile_nfw, mass_defs
from plotter import plot, loglogplot
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from scipy.optimize import curve_fit, root_scalar
from astropy.cosmology import FlatLambdaCDM, z_at_value
from sklearn.linear_model import LinearRegression
import astropy.units as u
from scipy.stats import spearmanr
from matplotlib.ticker import MultipleLocator
import subprocess
from numba import jit, njit, prange
from os import getcwd
from os.path import isfile
from scipy.integrate import quad
import warnings
import seaborn as sns
from pathlib import Path
from os.path import expanduser
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


home_dir = Path(expanduser('~'))
multimah_root = home_dir / 'frank_mah/output'
obs_data_dir = home_dir / 'research/nth_frac_cosmology/obs_data'
fig_dir = home_dir / 'research/nth_frac_cosmology/figures'


# In[12]:


# global variables
# free parameters of Shi+14 model; can be changed later
beta_def = 1.0
eta_def = 0.7

G = colossus.utils.constants.G
cm_per_km = 1e5
km_per_kpc = colossus.utils.constants.KPC / cm_per_km  # KPC was in cm
s_per_Gyr = colossus.utils.constants.GYR
yr_per_Gyr = 1E9

# can reset cosmology on-the-fly, will start with WMAP 5
# but note that paper uses planck18
cosmo = cosmology.setCosmology('WMAP5')
cosmo_astro = FlatLambdaCDM(H0=cosmology.getCurrent().H0 * u.km / u.s / u.Mpc, Tcmb0=cosmology.getCurrent(
).Tcmb0 * u.K, Om0=cosmology.getCurrent().Om0, Neff=cosmology.getCurrent().Neff, Ob0=cosmology.getCurrent().Ob0)

# adding a modified EdS
eds_params = cosmology.cosmologies['EdS']
eds_params['Ob0'] = 0.001
cosmology.addCosmology('near_EdS', eds_params)

# let's make some various cosmologies perturbed from planck18 fiducial
# we can go up and down in sigma_8
# then go up and down in omega_m
# then go up and down in h
# can always change the fiducial if we want
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['Om0'] = 0.1
cosmology.addCosmology('pl18_lowOm0', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['Om0'] = 0.5
cosmology.addCosmology('pl18_hiOm0', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()  # reset to default
fiducial_params['sigma8'] = 0.5
cosmology.addCosmology('pl18_lows8', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['sigma8'] = 1.2
cosmology.addCosmology('pl18_his8', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()  # reset to default
fiducial_params['H0'] = 60
cosmology.addCosmology('pl18_lowH0', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['H0'] = 80
cosmology.addCosmology('pl18_hiH0', fiducial_params)

print(cosmology.getCurrent())


# In[5]:


# cosmologies to test the effect of varying S8 instead...
# S_8 = sigma_8 * sqrt(Omega_m / 0.3)

cosmology.setCosmology('planck18')


def S8_cosmo(cosmo):
    return cosmo.sigma8 * np.sqrt(cosmo.Om0 / 0.3)


print(S8_cosmo(cosmology.getCurrent()))

# so 0.825 is S8 for planck18
# we need to now vary this up and down by varying sigma8 and Om0

fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['sigma8'] = 1.0
fiducial_params['Om0'] = 0.3
cosmology.addCosmology('pl18_s8_1_1', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['sigma8'] = 0.8
fiducial_params['Om0'] = 0.469
cosmology.addCosmology('pl18_s8_1_2', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['sigma8'] = 0.6
fiducial_params['Om0'] = 0.3
cosmology.addCosmology('pl18_s8_p6_1', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['sigma8'] = 0.5
fiducial_params['Om0'] = 0.432
cosmology.addCosmology('pl18_s8_p6_2', fiducial_params)

# dictionary of names for different cosmologies
cosmo_dict = {'near_EdS': 'eeddss', 'WMAP5': 'WMAP05', 'planck18': 'plnk18',
              'pl18_lowOm0': 'p8loOm', 'pl18_hiOm0': 'p8hiOm', 'pl18_lows8': 'p8los8',
              'pl18_his8': 'p8his8', 'pl18_lowH0': 'p8loH0', 'pl18_hiH0': 'p8hiH0',
              'pl18_s8_1_1': 'p8s811', 'pl18_s8_1_2': 'p8s812', 'pl18_s8_p6_1': 'p8s861',
              'pl18_s8_p6_2': 'p8s862'}

cosmo = cosmology.setCosmology('WMAP5')


# In[6]:


def zhao_vdb_conc(t, t04):
    return 4.0 * (1.0 + (t / (3.40*t04))**6.5)**(1.0/8.0)


# In[7]:


# masses corresponding to each peak height at the redshifts of interest
cosmo = cosmology.setCosmology('planck18')
for z in [0., 0.5, 1., 1.5, 2., 2.5, 3.]:
    print("z:", z)
    for nu in [1.16, 2.01, 4.10]:
        print(np.log10(peaks.massFromPeakHeight(nu, z)))


# ## Komatsu and Seljak model

# In[9]:


# computing t_dis from t_orb
# the masses are in Msun/h
# the lengths for haloes are in kpc/h


def NFWf(x):
    return np.log(1. + x) - x/(1. + x)

# accepts radius in physical kpc/h


def NFWM(r, M, z, c, R):
    return M * NFWf(c*r/R) / NFWf(c)

# dissipation timescale


def t_d(r, M, z, c, R, beta=beta_def):
    Menc = NFWM(r, M, z, c, R)
    t_dyn = 2. * np.pi * (r**3 / (G*Menc))**(1./2.) *         km_per_kpc / (cosmology.getCurrent().H0 / 100.)
    return beta * t_dyn / s_per_Gyr / 2.


# In[10]:


# look at t_ddis vs. r for a 10^14 Msun/h halo to verify the results

mass = 1e14  # Msun/h
Rvir = mass_so.M_to_R(mass, 0.0, 'vir')
nr = 30
rads = np.logspace(np.log10(0.01*Rvir), np.log10(Rvir), nr)

dt = mass_so.dynamicalTime(0.0, 'vir', definition='orbit')
print(dt / 2)


loglogplot()
plt.plot(rads/Rvir, t_d(rads, mass, 0.0, c=6, R=Rvir),
         label=r'$t_\mathrm{d}(r)$')
plt.plot([1.], [dt/2], '*', label=r'Colossus Orbit $t_\mathrm{dyn}$')
plt.legend()
plt.xlabel(r'$r / r_\mathrm{vir}$')
plt.ylabel(r'$t_\mathrm{d}$ [Gyr]')
# Note that this disagrees by ~factor of 2~ with Shi and Komatsu 2014, which is explained in their erratum


# In[11]:


# Komatsu and Seljak Model
# Here, c_nfw is defined using the virialization condition of Lacey & Cole (1993) and Nakamura & Suto (1997)


def Gamma(c_nfw):
    return 1.15 + 0.01*(c_nfw - 6.5)


def eta0(c_nfw):
    return 0.00676*(c_nfw - 6.5)**2 + 0.206*(c_nfw - 6.5) + 2.48


def NFWPhi(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    if(type(r) != np.ndarray and r == 0):
        return -1. * (G * M / R) * (c / NFWf(c))
    else:
        return -1. * (G * M / R) * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))

# this agrees with Komatsu and Seljak 2001 eqn 19 for theta:


def theta(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    # the rho0/P0 is actually 3eta^-1(0) * R/(GM) from Komatsu and Seljak 2001
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    return 1. + ((Gamma(c) - 1.) / Gamma(c))*rho0_by_P0*(NFWPhi(0, M, z, conc_model='diemer19', mass_def='vir')-NFWPhi(r, M, z, conc_model='diemer19', mass_def='vir'))

# arbitrary units for now while we figure out what to do with the normalization:


def rho_gas_unnorm(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    return theta(r, M, z, conc_model, mass_def)**(1.0 / (Gamma(c) - 1.0))

# in km/s:


def sig2_tot_obsolete(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    return (1.0 / rho0_by_P0) * theta(r, M, z, conc_model, mass_def)

# the complete sig2_tot that only makes one of each relevant function call:


def sig2_tot(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gamma(c) - 1.) / Gamma(c)) *         3. * eta0(c)**-1 * (phi0 - phir)
    return (1.0 / rho0_by_P0) * theta

# concentration increases with time
# radius also increases with time (hence why c increases)


# ## Functions for calls to average MAH codes

# In[24]:


zhao_exec_name = 'mandc.x'
vdb_exec_name = 'getPWGH'

# masses are mvir in Msun/h


def zhao_mah(Mobs, z_obs, cosmo):
    lgMobs = np.log10(Mobs)
    zpt = '%05d' % (np.round(z_obs, decimals=1)*100)
    mpt = '%05d' % (np.round(lgMobs, decimals=1)*100)
    df_name = 'mchistory_%s.%s.%s' % (cosmo_dict[cosmo.name], zpt, mpt)
    if(isfile(df_name)):  # we've already generated this run
        data = np.loadtxt(df_name, skiprows=1)
    else:
        instring = '%s\n%.3f %.3f\n1\n%.3f\n%.3f\n%.3f\n%.4f %1.3f\n1\n%1.1f\n%2.1f' % (
            cosmo_dict[cosmo.name], cosmo.Om0, cosmo.Ode0, cosmo.H0/100., cosmo.sigma8, cosmo.ns, cosmo.Ob0, cosmo.Tcmb0, z_obs, lgMobs)
        command = "echo '%s' | %s/%s" % (instring, getcwd(), zhao_exec_name)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        data = np.loadtxt(df_name, skiprows=1)
    zeds = data[:, 0]
    mass = data[:, 1]
    conc = data[:, 2]
    times = data[:, -1] / yr_per_Gyr / (cosmo.H0/100.)
    dMdt = (mass[1:] - mass[:-1]) / (times[1:] - times[:-1])
    # setting the dMdt at present day to zero since we don't need to evolve past z=0
    dMdt = np.insert(dMdt, len(dMdt), 0)
    out = np.column_stack((zeds, mass, conc, dMdt))
    out = np.flip(out, axis=0)
    return(out)

# same units for both, dM/dt in Msun/h / Gyr, mass in Msun/h:


def vdb_mah(Mobs, z_obs, cosmo, tp='average', return_sigma_D=False):
    if(tp == 'median'):
        med_or_avg = 0
    elif(tp == 'average'):
        med_or_avg = 1
    df_out_name = 'PWGH_%s.dat' % tp  # name used by Frank's code
    lgMobs = np.log10(Mobs)
    zpt = '%05d' % (np.round(z_obs, decimals=1)*100)
    mpt = '%09d' % (int(lgMobs*1e7))
    # name we will save file as
    df_name = 'PWGH_%s.%s.%s' % (cosmo_dict[cosmo.name], zpt, mpt)
    if(isfile(df_name) and return_sigma_D == False):  # we've already generated this run
        data = np.loadtxt(df_name)
    else:
        instring = '%.3f\n%.3f\n%.3f\n%.3f\n%.4f\n%1.1E\n%1.1f\n%1d' % (
            cosmo.Om0, cosmo.H0/100., cosmo.sigma8, cosmo.ns, cosmo.Ob0*(cosmo.H0/100.)**2, Mobs, z_obs, med_or_avg)
        command = "echo '%s' | %s/%s; mv %s %s" % (
            instring, getcwd(), vdb_exec_name, df_out_name, df_name)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        data = np.loadtxt(df_name)
    zeds = data[:, 1]
    mass = 10**data[:, 3] * Mobs
    conc = data[:, 6]
    dMdt = data[:, 7] * yr_per_Gyr
    out = np.column_stack((zeds, mass, conc, dMdt))
    out = np.flip(out, axis=0)
    if(return_sigma_D == False):
        return(out)
    else:
        return out, data[:, 8][::-1], data[:, 9][::-1]


# In[25]:


# concentration vs. mass and redshift

# we can generate some cvir interpolators for the redshifts of interest
cosmo = cosmology.setCosmology('planck18')
zeds = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
nm = 30
masses = np.logspace(11.5, 16, nm)  # just to cover full range
concs = np.zeros(nm)
conc_interps = {}
loglogplot()
for j, z in enumerate(zeds):
    for i, m in enumerate(masses):
        dat = vdb_mah(m, z, cosmo)
        concs[i] = dat[-1, 2]
    conc_interps[z] = interp(masses, concs)

    plt.plot(masses, concs, label=r'%.2f' % z)
    #plt.plot(masses, conc_interps[z](masses))

plt.xlabel(r'$M_\mathrm{vir}$')
plt.ylabel(r'$c_\mathrm{vir}$ from Zhao+09')
plt.legend()


# In[26]:


# function to solve for Mvir (and hence c_vir) given M_200m and z_obs


def vir_from_other(m_other, mult_other, c_or_m, zobs, cosmo, r_other_mult_max=1.5):
    def_other = str(mult_other) + c_or_m
    if(c_or_m == 'c'):
        rho_other = cosmo.rho_c(zobs)
    elif(c_or_m == 'm'):
        rho_other = cosmo.rho_m(zobs)
    else:
        print("Must be mean or crit for density type!")

    conc_func = conc_interps[zobs]

    r_other = mass_so.M_to_R(m_other, zobs, def_other)
    lhs = r_other**3 * mult_other * rho_other /         (mass_so.deltaVir(zobs) * cosmo.rho_c(zobs))

    # will need to use mass_so.R_to_M(r, zobs, 'vir'), NFWf(c_vir)
    # solving for the value of rvir that gets func to zero

    def root_func(rvir):
        mvir = mass_so.R_to_M(rvir, zobs, 'vir')
        cvir = conc_func(mvir)
        #print(rvir, cvir, mvir, zobs)
        return lhs - (rvir**3 * NFWf(cvir * r_other / rvir) / NFWf(cvir))

    # find root of root_func
    # will be in units of kpc/h
    rvir = root_scalar(root_func, bracket=(1., r_other*r_other_mult_max)).root
    return mass_so.R_to_M(rvir, zobs, 'vir')


# ## Pipeline to compute nonthermal pressure fraction from average MAHs

# In[19]:


def sig2_tot(r, M, c, R):
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gamma(c) - 1.) / Gamma(c)) *         3. * eta0(c)**-1 * (phi0 - phir)
    return (1.0 / rho0_by_P0) * theta


# In[23]:


def t_d(r, M, z, c, R, beta=beta_def):
    Menc = NFWM(r, M, z, c, R)
    t_dyn = 2. * np.pi * (r**3 / (G*Menc))**(1./2.) *         km_per_kpc / (cosmology.getCurrent().H0 / 100.)
    return beta * t_dyn / s_per_Gyr / 2.

# another possible definition for the dissipation timescale, Brunt-Vaisala timescale:


def t_BV(r, M, z, c, R, fnth, gamma, beta):
    g = -G*M * NFWf(c*r/R) / (NFWf(c) * r**2)
    Gm = Gamma(c)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gm - 1.) / Gm) * 3. * eta0(c)**-1 * (phi0 - phir)
    lnK = ((Gm - gamma) / (Gm - 1.) * np.log(theta)) +        np.log(1. - fnth)
    lnK_interp = interp(r, lnK)
    dlnK_dr = lnK_interp.derivative(1)
    N_BV = np.sqrt(-1. * g / gamma * dlnK_dr(r))
    return beta * 1. / N_BV * km_per_kpc / (cosmology.getCurrent().H0 / 100.) / s_per_Gyr


def dlnK_dlnr(r, M, z, c, R, fnth, gamma, beta):
    g = -G*M * NFWf(c*r/R) / (NFWf(c) * r**2)
    Gm = Gamma(c)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gm - 1.) / Gm) * 3. * eta0(c)**-1 * (phi0 - phir)
    lnK = ((Gm - gamma) / (Gm - 1.) * np.log(theta)) +        np.log(1. - fnth)
    lnK_interp = interp(np.log(r), lnK)
    dlnK_dlnr = lnK_interp.derivative(1)
    return(dlnK_dlnr(np.log(r)))

#takes in Mobs, zobs, cosmo
# returns f_nth, sig2nth, sig2tot at z=zobs


def gen_fnth(Mobs, zobs, cosmo, mah_retriever=vdb_mah, mass_def='vir', conc_model='duffy08', beta=beta_def, eta=eta_def, nrads=30, zi=30., r_mult=1., timescale='td', init_eta=eta_def, conc_test_flag=False, psires=1e-4, dsig_pos=False, return_full=False):
    data = mah_retriever(Mobs, zobs, cosmo)
    # This below is defunct, we switched to setting initial time based on m/M0
    #first_snap_to_use = np.where(data[:,0] <= zi)[0][0] - 1
    # first snap where mass is above psi_res
    first_snap_to_use = np.where(data[:, 1]/Mobs >= psires)[0][0]
    data = data[first_snap_to_use:]

    n_steps = data.shape[0] - 1

    Robs = mass_so.M_to_R(Mobs, zobs, mass_def)

    rads = np.logspace(np.log10(0.01*Robs), np.log10(r_mult*Robs), nrads)

    ds2dt = np.zeros((n_steps, nrads))
    sig2tots = np.zeros((n_steps, nrads))
    sig2nth = np.zeros((n_steps, nrads))

    # this process could be made more pythonic so that the looping is faster
    for i in range(0, n_steps):
        z_1 = data[i, 0]  # first redshift
        z_2 = data[i+1, 0]  # second redshift, the one we are actually at
        dt = cosmo.age(z_2) - cosmo.age(z_1)  # in Gyr
        mass_1 = data[i, 1]
        mass_2 = data[i+1, 1]
        dM = mass_2 - mass_1
        dMdt = data[i+1, 3]  # since the (i+1)th is computed between i+1 and i
        R_1 = mass_so.M_to_R(mass_1, z_1, mass_def)
        R_2 = mass_so.M_to_R(mass_2, z_2, mass_def)
        if(conc_model == 'vdb'):
            c_1 = data[i, 2]
            c_2 = data[i+1, 2]
        else:
            c_1 = concentration.concentration(
                mass_1, mass_def, z_1, model=conc_model)
            c_2 = concentration.concentration(
                mass_2, mass_def, z_2, model=conc_model)

        if(conc_test_flag):
            # set concentrations to 4 if t004 is further back than the furthest timestep we have data for
            # just to verify that results aren't affected
            t04_ind = np.where(data[:i+1, 1] > 0.04 * mass_2)[0][0]
            if(t04_ind == 0):
                print(i, z_1)
                c_1 = 4.
                c_2 = 4.

        sig2tots[i, :] = sig2_tot(rads, mass_2, c_2, R_2)  # second timestep
        if(i == 0):
            ds2dt[i, :] = (sig2tots[i, :] -
                           sig2_tot(rads, mass_1, c_1, R_1)) / dt
            sig2nth[i, :] = init_eta * sig2tots[i, :]
        else:
            ds2dt[i, :] = (sig2tots[i, :] - sig2tots[i-1, :]) / dt
            if(dsig_pos):
                # another check to make sure rare cases where dsig2dt is negative (turbulent energy removed)
                # doesn't affect results; it doesn't, since in general the halo should be growing in mass
                # and this should almost never happen
                ds2dt[i, ds2dt[i, :] < 0] = 0.
            if(timescale == 'td'):
                # t_d at z of interest z_2
                td = t_d(rads, mass_2, z_2, c_2, R_2, beta=beta_def)
            elif(timescale == 'tBV'):
                td = t_BV(rads, mass_2, z_2, c_2, R_2,
                          sig2nth[i-1, :] / sig2tots[i-1, :], 5./3., beta)
            sig2nth[i, :] = sig2nth[i-1] +                 ((-1. * sig2nth[i-1, :] / td) + eta * ds2dt[i, :])*dt
            # can't have negative sigma^2_nth at any time
            sig2nth[i, sig2nth[i, :] < 0] = 0
    if(return_full == False):
        fnth = sig2nth[-1, :] / sig2tots[-1, :]
        return fnth, rads, sig2nth[-1, :], sig2tots[-1, :], data[-1, 0], data[-1, 2]
    else:
        fnth = sig2nth / sig2tots
        # return redshifts+concs too
        return fnth, rads, sig2nth[-1, :], sig2tots[-1, :], data[:, 0], data[:, 2]


# In[28]:


# Nelson+14 fitting formula


def fnth_nelson(r_by_r200m):
    A = 0.452
    B = 0.841
    gam = 1.628
    return 1. - A*(1. + np.exp(-1.*(r_by_r200m / B)**gam))


# In[31]:


# looking a bit more at the Brunt-Vaisala frequency

mass = 5e15  # Msun/h
zobs = 0
conc = 10
Rvir = mass_so.M_to_R(mass, zobs, 'vir')
m200m, R200m, _ = mass_defs.changeMassDefinition(
    mass, conc, zobs, 'vir', '200m')
nr = 30
rads = np.logspace(np.log10(0.01*R200m), np.log10(R200m), nr)

dt = mass_so.dynamicalTime(zobs, 'vir', definition='orbit')
print(dt / 2)


loglogplot()
plt.plot(rads/R200m, t_d(rads, mass, zobs, c=conc, R=Rvir),
         label=r'$\beta t_\mathrm{orb}(r)/2$, $\beta=1$')
plt.plot(rads/R200m, t_BV(rads, mass, zobs, c=conc, R=Rvir, fnth=fnth_nelson(rads/R200m), gamma=5./3.,
                          beta=2.4), label=r'$\beta_\mathrm{BV}t_\mathrm{BV}(r)$, $f_\mathrm{nth,N+14}$, $\beta_\mathrm{BV}=2.4$')
plt.plot(rads/R200m, t_BV(rads, mass, zobs, c=conc, R=Rvir, fnth=0, gamma=5./3., beta=2.8),
         label=r'$\beta_\mathrm{BV}t_\mathrm{BV}(r)$, $f_\mathrm{nth}=0$, $\beta_\mathrm{BV}=2.8$')

plt.legend(frameon=False)
plt.xlabel(r'$r / r_\mathrm{200m}$')
plt.ylabel(r'$t_\mathrm{d}$ [Gyr]')
plt.savefig(fig_dir / 't_BV_vs_t_orb.pdf')


# In[32]:


# differences in BV frequency arise due to fnth profile used

loglogplot()
plt.plot(rads/R200m, dlnK_dlnr(rads, mass, zobs, c=conc, R=Rvir,
                               fnth=fnth_nelson(rads/R200m), gamma=5./3., beta=2.4), label=r'$f_\mathrm{nth,N+14}(r)$')
plt.plot(rads/R200m, dlnK_dlnr(rads, mass, zobs, c=conc, R=Rvir,
                               fnth=0, gamma=5./3., beta=2.8), label=r'$f_\mathrm{nth}(r)=0$')
plt.legend(frameon=False)
plt.xlabel(r'$r/r_\mathrm{200m}$')
plt.ylabel(r'${\rm d}\ln K / {\rm d}\ln r$')
plt.ylim(1e-1, 2.1)
plt.savefig(fig_dir / 'dlnK_dlnr.pdf')
print(dlnK_dlnr(rads, mass, zobs, c=conc, R=Rvir,
                fnth=fnth_nelson(rads/R200m), gamma=5./3., beta=2.4)[-1])
print(dlnK_dlnr(rads, mass, zobs, c=conc,
                R=Rvir, fnth=0, gamma=5./3., beta=2.8)[-1])
print(1.384/0.928)


# ## Nonthermal pressure fractions from average MAHs, comparison to Nelson et al. (2014), and HSE mass biases

# In[38]:


# a couple different cosmologies we want to test

fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['Om0'] = 0.25
cosmology.addCosmology('planck18_lO', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['Om0'] = 0.35
cosmology.addCosmology('planck18_hO', fiducial_params)
fiducial_params = cosmology.cosmologies['planck18'].copy()
fiducial_params['Om0'] = 0.5
cosmology.addCosmology('planck18_vhO', fiducial_params)
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

eds_params = cosmology.cosmologies['EdS']
eds_params['Ob0'] = 0.001
cosmology.addCosmology('near_EdS', eds_params)


cosmo_dict = {'near_EdS': 'eeddss', 'WMAP5': 'WMAP05', 'planck18': 'plnk18',
              'planck18_lO': 'p8loOm', 'planck18_hO': 'p8hiOm',
              'planck18_lS': 'p8los8', 'planck18_hS': 'p8his8',
              'planck18_lH': 'p8loH0', 'planck18_hH': 'p8hiH0',
              'planck18_vhO': 'p8vhiO'
              }


# In[39]:


# compute f_nth fitting function


def fnth_fit(rad_nu, A, B, C, D, E, F):
    r_by_r200m = rad_nu[:, 0]
    nuvals = rad_nu[:, 1]
    return 1. - A*(1. + np.exp(-1.*(r_by_r200m / B)**C)) * (nuvals/4.1)**(D / (1. + (r_by_r200m / E)**F))


def compute_fitting_func(cosmo, mah_retriever=vdb_mah, mass_def='vir', conc_model='vdb', beta=beta_def, eta=eta_def, r_mult=1.):
    print(cosmo)

    zobs = 1
    nnu = 15
    nu_200m = np.linspace(1.0, 4.2, nnu)

    rad_arr = []
    fnth_arr = []
    nu_arr = []

    for i, nu in enumerate(nu_200m):
        m200m = peaks.massFromPeakHeight(nu, zobs)
        # converting to m_vir from m_200m
        mvir = vir_from_other(m200m, 200, 'm', zobs, cosmo)
        fnth, rads, _, _, zz, conc = gen_fnth(
            mvir, zobs, cosmo, mah_retriever, mass_def, conc_model, beta, eta, r_mult=r_mult, zi=30.)
        r200m = mass_so.M_to_R(m200m, zobs, '200m')
        msk = rads / r200m <= 2.0
        rad_arr.extend(rads[msk]/r200m)
        fnth_arr.extend(fnth[msk])
        nu_arr.extend(np.repeat(nu, len(rads[msk])))

    rad_arr = np.array(rad_arr)
    fnth_arr = np.array(fnth_arr)
    nu_arr = np.array(nu_arr)
    in_arr = np.column_stack((rad_arr, nu_arr))
    # output is the fnth
    # now we want to fit all of the curves using r/r_200m and nu_200m as the inputs
    params = curve_fit(fnth_fit, in_arr, fnth_arr, p0=[
                       0.452, 0.841, 1.628, 1.0, 0.2, -1.0])[0]
    return params, in_arr, fnth_arr


cosmo = cosmology.setCosmology('planck18')
params, in_arr, fnth_arr = compute_fitting_func(
    cosmo, mah_retriever=vdb_mah, mass_def='vir', r_mult=2.)
print(params)


# In[40]:


# check how accurate our model is relative to the data

pred_arr = fnth_fit(in_arr, params[0], params[1],
                    params[2], params[3], params[4], params[5])
nnu = 15
nu_200m = np.linspace(1.0, 4.2, nnu)
rel_err = (pred_arr - fnth_arr) / fnth_arr

cols = sns.cubehelix_palette(nnu)

plot(semilogx=True)
for i in range(0, len(nu_200m)):
    plt.plot(in_arr[30*i:30*(i+1), 0], rel_err[30*i:30*(i+1)],
             color=cols[i], label=r'$%.1f$' % nu_200m[i])

plt.legend(frameon=False, ncol=2,
           title=r'$\nu_\mathrm{200m}$', title_fontsize=18, bbox_to_anchor=(1.2, 1))
plt.xlim(10**-1, 2)
plt.ylim(-0.2, 0.2)
plt.xlabel(r'$r/r_\mathrm{200m}$')
plt.ylabel(r'Fractional Error')

# it does good enough for the purposes of any analysis at this stage in f_nth research


# In[48]:


def fnth_fit(r_by_r200m, nu, A, B, gam, beta, rc, alpha):
    return 1. - A*(1. + np.exp(-1.*(r_by_r200m / B)**gam)) * (nu/4.1)**(beta / (1. + (r_by_r200m / rc)**alpha))


def plot_fixed_nu_fixed_M(masses_200m, zeds, cosmo, mah_retriever=vdb_mah, mass_def='vir', conc_model='vdb', beta=beta_def, eta=eta_def, r_mult=1., init_eta=eta_def, psires=1e-4):
    print(cosmo)  # just to be clear
    fig, ax = plt.subplots(nrows=len(masses), ncols=2, figsize=(
        13, 10), sharex=True, sharey=True, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for i in range(0, 2):
        for j in range(0, len(masses)):
            ax[j, i].yaxis.set_ticks_position('both')
            ax[j, i].xaxis.set_ticks_position('both')
            ax[j, i].tick_params(axis='both', which='minor',
                                 colors='black', width=1.0, length=2.0)
            ax[j, i].set_ylim(0., 0.6)
            ax[j, i].set_xlim(0.0, 1.5)
            ax[j, i].xaxis.set_minor_locator(MultipleLocator(0.05))
            ax[j, i].yaxis.set_minor_locator(MultipleLocator(0.05))

    zobs = 0
    cols = sns.cubehelix_palette(len(zeds))

    for i, m200m in enumerate(masses_200m):
        nu200m = peaks.peakHeight(m200m, zobs)  # fixed nu_200m
        for j, z in enumerate(zeds):
            # since we put in M_vir, this is still M_vir
            m200m_newZ = peaks.massFromPeakHeight(nu200m, z)
            # converting to m_vir from m_200m, based on Zhao+09 concentrations
            mvir = vir_from_other(m200m_newZ, 200, 'm', z, cosmo)
            fnth, rads, _, _, zz, conc = gen_fnth(
                mvir, z, cosmo, mah_retriever, mass_def, conc_model, beta, eta, r_mult=r_mult, zi=30., init_eta=init_eta, psires=psires)
            r200m = mass_so.M_to_R(m200m_newZ, z, '200m')
            if(j == 0):
                ax[i, 1].text(
                    0.1, 0.5, r'Fixed $\nu_\mathrm{200m} = %.2f$' % nu200m, fontsize=16)
                ax[i, 0].text(
                    0.1, 0.5, r'Fixed $\log(M_\mathrm{200m}/[h^{-1}M_\odot])=%.1f$' % np.log10(m200m), fontsize=16)
            ax[i, 1].plot(rads/r200m, fnth, label='$%.0f$' %
                          z, color=cols[::-1][j])

            # now we need to do fixed mass on the left panel
            # converting from m_200m at z=0 to m_vir at each redshift
            mvir = vir_from_other(m200m, 200, 'm', z, cosmo)
            fnth, rads, _, _, zz, conc = gen_fnth(
                mvir, z, cosmo, mah_retriever, mass_def, conc_model, beta, eta, r_mult=r_mult, init_eta=init_eta)
            r200m = mass_so.M_to_R(m200m, z, '200m')
            ax[i, 0].plot(rads/r200m, fnth, label='$%.1f$' %
                          np.log10(m200m), color=cols[::-1][j])

        ax[i, 1].plot(rads/r200m, fnth_nelson(rads/r200m),
                      '-.', color='k', label='Nelson+14')
        ax[i, 1].plot(rads/r200m, fnth_fit(rads/r200m, nu200m, params[0], params[1], params[2], params[3],
                                           params[4], params[5]), linestyle='dotted', color='k', label='This work', linewidth=2)

    for i in range(0, len(masses)):
        ax[i, 0].set_ylabel(r'$f_\mathrm{nth}$')
    ax[len(masses)-1, 0].set_xlabel(r'$r/r_\mathrm{200m}$')
    ax[len(masses)-1, 1].set_xlabel(r'$r/r_\mathrm{200m}$')

    handles, labels = ax[0, 1].get_legend_handles_labels()
    ax[2, 1].legend(handles[4:], labels[4:], frameon=False, fontsize=16, loc=4)
    ax[2, 0].legend(handles[0:4], labels[0:4], frameon=False,
                    fontsize=16, title=r'$z=$', title_fontsize=18)
    # plt.savefig(fig_dir / 'fnth_vs_radius_fixed_nu.pdf', bbox_inches='tight')

    return fig, ax


cosmo = cosmology.setCosmology('planck18')
zeds = [0, 1., 2.0, 3.0]
masses = np.logspace(13, 15.4, 3)


plot_fixed_nu_fixed_M(masses, zeds, cosmo, mah_retriever=vdb_mah,
                      mass_def='vir', r_mult=2., init_eta=0.7, psires=1e-300)


# In[49]:


# Hydrostatic mass bias as a function of M500c (the commonly observed radius)

from matplotlib.patches import Rectangle


def nfw_prof(r, rhos, rs):
    return rhos / ((r/rs) * (1. + r/rs)**2)


def plot_hse_bias(mah_retriever=vdb_mah, mass_def='vir', conc_model='vdb', beta=beta_def, eta=eta_def, r_mult=1.):
    fig, ax = plot()
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis='both', which='minor',
                   colors='black', width=1.0, length=2.0)
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

    cosmo = cosmology.setCosmology('planck18')
    print(cosmo)

    nm = 15

    zeds = np.array([0.0, 1.0, 2.0, 3.0])
    m500c_vals = np.logspace(13, 15.4, nm)

    cols = sns.cubehelix_palette(len(zeds))

    for i, zobs in enumerate(zeds):
        ratios_500c = np.zeros(nm)
        for j, m500c in enumerate(m500c_vals):

            # convert from M500c to Mvir
            mvir = vir_from_other(m500c, 500, 'c', zobs,
                                  cosmo, r_other_mult_max=2.5)
            r500c = mass_so.M_to_R(m500c, zobs, '500c')
            fnth, rads, sig2nth, sig2tots, zz, conc = gen_fnth(
                mvir, zobs, cosmo, mah_retriever, mass_def, conc_model, beta, eta, r_mult=r_mult, nrads=200, return_full=False, zi=30.)

            # for computing the enclosed mass out to arbitrary radii
            rhos, rs = profile_nfw.NFWProfile.fundamentalParameters(
                mvir, conc, zobs, 'vir')
            # need M(<5R500c) for gas mass normalization
            Rvir = mass_so.M_to_R(mvir, zobs, 'vir')
            m200m, r200m, _ = mass_defs.changeMassDefinition(
                mvir, conc, zobs, 'vir', '200m')
            M2R200m = quad(lambda x: 4. * np.pi * x**2 *
                           nfw_prof(x, rhos, rs), 0, 2.0*r200m)[0]

            # compute rho_gas profile, use it to compute M_gas within Rdef and T_mgas within Rdef
            rho0_by_P0 = 3*eta0(conc)**-1 * Rvir/(G*m)
            phi0 = -1. * (conc / NFWf(conc))
            def phir(rad): return -1. * (conc / NFWf(conc)) *                 (np.log(1. + conc*rad/Rvir) / (conc*rad/Rvir))

            def theta(rad): return 1. + ((Gamma(conc) - 1.) /
                                         Gamma(conc)) * 3. * eta0(conc)**-1 * (phi0 - phir(rad))
            cbf = cosmo.Ob0 / cosmo.Om0
            rho0_nume = nume = cbf * M2R200m
            rho0_denom = 4. * np.pi *                 quad(lambda x: theta(x)**(1.0 / (Gamma(conc) - 1.0))
                     * x**2, 0, 2.0*r200m)[0]
            # This now pegs the gas mass to be equal to cosmic baryon fraction at 2R500m
            # NOTE: Both rho0_nume and rho_denom need to be changed if the radius is changed
            rho0 = rho0_nume / rho0_denom
            def rhogas(rad): return rho0 *                 theta(rad)**(1.0 / (Gamma(conc) - 1.0))

            Ptot = rhogas(rads) * sig2tots
            Pth = Ptot * (1.0 - fnth)

            # these radii are in real units of Robs=Rvir
            Ptot_interp = interp(rads, Ptot, k=4)
            Pth_interp = interp(rads, Pth, k=4)
            fnth_interp = interp(rads, fnth, k=4)

            ratios_500c[j] = Pth_interp.derivative(1)(
                r500c) / Ptot_interp.derivative(1)(r500c)

            assert ratios_500c[j] >= 1.0 - fnth_interp(r500c)

        ax.plot(np.log10(m500c_vals), ratios_500c, label=r'$%.0f$' %
                zobs, color=cols[::-1][i])

    rect = Rectangle((14.0, 0.8), 1.5, 0.1, edgecolor='k', linewidth=2,
                     facecolor='none', hatch='/', zorder=-32, label='$0.25$, Henson+17')
    ax.add_patch(rect)

    ax.set_ylabel(r'$M^\mathrm{HSE}_\mathrm{500c} / M_\mathrm{500c}$')
    ax.set_xlabel(r'$\log(M_\mathrm{500c}/[h^{-1}M_\odot])$')
    leg = ax.legend(frameon=False, fontsize=16)
    ax.text(13.05, 0.71, r'$z=$', fontsize=18)
    #plt.savefig(fig_dir / 'hse_bias.pdf', bbox_inches='tight')
    return fig, ax


plot_hse_bias(r_mult=2.)

# This looks good and is roughly in agreement with if we do it using Duffy+08 instead


# ## Function for calls to Monte Carlo MAH codes and MAH variance illustration

# In[47]:


def multimah(Mobs, z_obs, cosmo, Nmah, tp='average'):
    # loads in an array of MAH from Frank's MAH code, specify Nmah = number of MAH to get
    mass_int = int(np.log10(Mobs)*10)
    z_int = int(z_obs*100)
    mah_dir = multimah_root / ('%s/m%03d/z%03d' %
                               (cosmo.name, mass_int, z_int))
    dat1 = np.loadtxt(mah_dir / 'MAH0001.dat')
    redshifts = dat1[:, 1]
    lbtime = dat1[:, 2]
    nz = len(dat1)
    dat = np.zeros((Nmah, nz))
    std = np.zeros((nz, 2))
    for i in range(0, Nmah):
        dat[i, :] = np.loadtxt(mah_dir / ('MAH%04d.dat' % (i+1)), usecols=3)
    dat = 10**dat
    if(tp == 'full'):
        # return the full array instead of giving standard deviations
        return dat*Mobs, redshifts, lbtime
    elif(tp == 'average'):
        mah = np.average(dat, axis=0)
        std[:, 0] = np.std(dat, axis=0)
        std[:, 1] = std[:, 0]
    elif(tp == 'logaverage'):
        mah = np.average(np.log10(dat), axis=0)
        std[:, 0] = np.std(np.log10(dat), axis=0)
        std[:, 1] = std[:, 0]
        mah = 10**mah
        std = 10**std
    elif(tp == 'median'):
        mah = np.median(dat, axis=0)
        std[:, 0] = mah - np.percentile(dat, 16, axis=0)
        std[:, 1] = np.percentile(dat, 84, axis=0) - mah
    mah = mah * Mobs
    std = std * Mobs
    return mah, std, redshifts, lbtime


# In[50]:


# MC MAH for a 10^14 Msun/h (mvir) halo
# M(z) / M(z=0) vs. 1+z, both logged
from matplotlib import lines as mlines
print(cosmo)
fig, ax = plot()
mvir = 10**14
M200m, r200m, _ = mass_defs.changeMassDefinition(mvir, 5.0, 0.0, 'vir', '200m')
print(np.log10(M200m))
vd = vdb_mah(mvir, 0.0, cosmo)
mah, redshifts, lbtime = multimah(mvir, 0.0, cosmo, 1000, tp='full')

msk = redshifts < 8.

for i in range(0, 59):
    plt.plot(np.log10(1+redshifts[msk]), np.log10(mah[i, msk] /
                                                  mah[i, 0]), color='k', linewidth=1, linestyle='dashed')

plt.plot(np.log10(1+redshifts[msk]), np.log10(mah[59, msk] / mah[59, 0]), color='k', linewidth=1,
         linestyle='dashed', label=r'\noindent Parkinson et al. (2008) \vspace{1.5mm} \\ Monte Carlo')

msk = vd[:, 0] < 8.

plt.plot(np.log10(1+vd[msk, 0]), np.log10(vd[msk, 1]/vd[-1, 1]), color='r', linewidth=3,
         label=r'\noindent van den Bosch et al. (2014) \vspace{1.5mm} \\Universal Model of Average')
plt.xlabel(r'$\log[1+z]$')
plt.ylabel(r'$\log[M_\mathrm{vir}(z) / M_\mathrm{vir}(z=0)]$')

handles, labels = ax.get_legend_handles_labels()
handles.insert(1, mlines.Line2D([], [], linestyle=''))
labels.insert(1, '')
leg = ax.legend(handles, labels, frameon=False)

cb = leg._legend_box._children[-1]._children[0]
for ib in cb._children:
    ib.align = "center"

plt.text(
    0.3, -0.2, r'$\log(M_\mathrm{vir}(z=0) / [h^{-1}M_\odot]) = 14$', fontsize=18)

#plt.savefig(fig_dir / 'mahs.pdf', bbox_inches='tight')


# ## Analysis of Monte Carlo MAH-based scaling relation results

# The data generated for the analysis below comes from the Monte Carlo MAHs generated using Frank's code.

# In[191]:


# make the list of random masses to generate MC MAHs
if(False):
    Nt = 9999
    logMvals = 12.0 + 3.5*np.random.rand(Nt)
    np.savetxt('logMvals.txt', logMvals, fmt='%1.5f')


# In[51]:


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


# In[54]:


# function to compute scatter, slope, normalization given masses and an observable (and a zero point, set to 1 for now)
def compute_fit(masses, obs, zero_point=1.):
    coeffs = np.polyfit(np.log10(masses/zero_point), np.log10(obs), deg=1)
    preds = 10**(coeffs[0]*np.log10(masses) + coeffs[1])
    resids = np.log(preds / obs)
    fractional_errors = preds/obs - 1
    scatter = np.std(resids)
    pc_scatter = 100. * scatter
    fractional_scatter = np.std(fractional_errors)
    robust_scatter = (np.percentile(resids, 84) -
                      np.percentile(resids, 16)) / 2.0
    robust_fractional_scatter = (np.percentile(
        fractional_errors, 84) - np.percentile(fractional_errors, 16)) / 2.0
    pc_rbscatter = 100. * robust_scatter
    return coeffs[0], coeffs[1], pc_scatter, pc_rbscatter, scatter, robust_scatter


radii_definitions = [('vir', 1), ('500c', 1), ('500c', 2), ('500c', 3), ('500c', 4), ('500c', 5),
                     ('200m', 0.3), ('200m', 0.5), ('200m',
                                                    0.875), ('200m', 1.0), ('200m', 1.25),
                     ('200m', 1.625), ('200m', 2.0)]
cosmos = ['planck18', 'planck18_lO', 'planck18_hO', 'planck18_lS',
          'planck18_hS', 'planck18_lH', 'planck18_hH']
# cosmology labels for plotting
fancy_cosmos = ["Planck '18", r'$\Omega_\mathrm{m}=0.25$',
                r'$\Omega_\mathrm{m}=0.35$', r'$\sigma_8 = 0.7$', r'$\sigma_8 = 0.9$', r'$h = 0.65$',
                r'$h=0.75$']

current_palette = sns.color_palette()

scaling_factors = (1., 0., 0.4)

cosmo = cosmology.setCosmology('planck18')


# In[56]:


# Compute just the properties for the z=0 case, see whta happens when we add x% scatter to the halo masses
cosmo = cosmology.setCosmology('planck18')
data = np.load(obs_data_dir / ('planck18_data.npz'))['data']
ap = 6
print(radii_definitions[ap])
msk = data[0,:,ap] > 1e14
slope, norm, pc_scatter, pc_rbscatter, scatter, rbscatter = compute_fit(
                    data[0,msk,ap], data[3,msk,ap], zero_point=10**14)
print(pc_rbscatter)
data[0,msk,ap] = data[0,msk,ap] * (1.0 + np.random.normal(0.0, 0.05, size=len(data[0,msk,ap])))
slope, norm, pc_scatter, pc_rbscatter, scatter, rbscatter = compute_fit(
                    data[0,msk,ap], data[3,msk,ap], zero_point=10**14)
print(pc_rbscatter)


# In[60]:


# UPDATED FIGURE THAT ONLY SHOWS Y_SZ


def plot_3x1_means_varyCosm():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(
        15, 5.5), sharex=True, gridspec_kw={'wspace': 0.25, 'hspace': 0.075})
    for i in range(0, 3):
        ax[i].yaxis.set_ticks_position('both')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].tick_params(axis='both', which='minor',
                          colors='black', width=1.0, length=2.0)
        ax[i].set_xlim(0.2, 2.1)
        ax[i].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax[i].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.25))
    ax[1].yaxis.set_minor_locator(MultipleLocator(0.005))
    ax[2].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax[0].plot([0.4], [6.2], '*', color='b', markersize=8, label='Pike+14 NR')

    # loop over cosmology
    for i, cs in enumerate(cosmos):
        cosmo = cosmology.setCosmology(cs)
        # load in the data
        data = np.load(obs_data_dir / ('%s_data.npz' % cs))['data']

        # loop over observables (mass_enc, Tmgasv, Mgasv, YSZv)
        for j in range(3, 4):
            # loop over the radius, these will be the r200m multiples
            mult = np.zeros(7)
            slopes = np.zeros(7)
            norms = np.zeros(7)
            pc_scatters = np.zeros(7)
            pc_rbscatters = np.zeros(7)
            rbscatters = np.zeros(7)
            scatters = np.zeros(7)
            msk = data[0, :, 9] > 1e14
            for k in range(6, 13):
                mult[k-6] = radii_definitions[k][1]
                slopes[k-6], norms[k-6], pc_scatters[k-6], pc_rbscatters[k-6], scatters[k-6], rbscatters[k-6] = compute_fit(
                    data[0, msk, k], data[j, msk, k], zero_point=10**14)
            scat_interp = interp(mult, pc_rbscatters)
            slope_interp = interp(mult, slopes)
            norm_interp = interp(mult, norms)
            mult_arr = np.linspace(0.3, 2.0, 60)
            ax[0].plot(mult_arr, scat_interp(mult_arr),
                       color=current_palette[i], label=fancy_cosmos[i])
            ax[1].plot(mult_arr, slope_interp(mult_arr),
                       color=current_palette[i], label=fancy_cosmos[i])
            ax[2].plot(mult_arr, norm_interp(mult_arr),
                       color=current_palette[i], label=fancy_cosmos[i])

    # Self-similar slopes
    ax[1].axhline(5.0/3.0, color='k', linestyle='dashed')

    ax[0].text(
        0.4, 9.15, r'$Y_\mathrm{SZ}(<R_\mathrm{ap})$ -- $M(<r_\mathrm{ap})$', fontsize=16)
    handles, labels = ax[0].get_legend_handles_labels()
    ax[2].legend(handles, labels, frameon=False, labelspacing=0.3, fontsize=16)
    for i in range(0, 3):
        ax[i].set_xlabel(r'$r_\mathrm{ap}/r_\mathrm{200m}$')
    ax[0].set_ylabel(r'Percent Scatter')
    ax[1].set_ylabel(r'Slope')
    ax[2].set_ylabel(r'Normalization')
    ax[0].text(0.4, 8.65, r'$z=0$', fontsize=18)
    ax[2].text(0.4, 0.4, r'$\log[h^{-1}\mathrm{kpc}^2]$', fontsize=18)
    ax[0].text(
        0.5, 4.5, r'$\log(M_\mathrm{200m}/[h^{-1}M_\odot]) \geq 14$', fontsize=18)

    #plt.savefig(fig_dir / 'mass_obs_relations.pdf', bbox_inches='tight')


plot_3x1_means_varyCosm()


# In[61]:


def plot_3x3_means_varyComplexity():
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(
        15, 13), sharex=True, gridspec_kw={'wspace': 0.25, 'hspace': 0.075})
    for i in range(0, 3):
        for j in range(0, 3):
            ax[i, j].yaxis.set_ticks_position('both')
            ax[i, j].xaxis.set_ticks_position('both')
            ax[i, j].tick_params(axis='both', which='minor',
                                 colors='black', width=1.0, length=2.0)
            ax[i, j].set_xlim(0.2, 2.1)
            ax[i, j].xaxis.set_minor_locator(MultipleLocator(0.1))
            ax[i, j].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[0, 0].yaxis.set_minor_locator(MultipleLocator(0.5))
    ax[0, 1].yaxis.set_minor_locator(MultipleLocator(0.1))
    ax[0, 2].yaxis.set_minor_locator(MultipleLocator(0.25))
    ax[1, 0].yaxis.set_minor_locator(MultipleLocator(0.005))
    ax[1, 1].yaxis.set_minor_locator(MultipleLocator(0.0005))
    ax[1, 2].yaxis.set_minor_locator(MultipleLocator(0.005))
    ax[2, 0].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax[2, 1].yaxis.set_minor_locator(MultipleLocator(0.01))
    ax[2, 2].yaxis.set_minor_locator(MultipleLocator(0.05))

    cosmo = cosmology.setCosmology('planck18')
    labs = ['Full Model', 'Fixed $c_\mathrm{vir}$',
            'Fixed $c_\mathrm{vir}$ and $t_\mathrm{dis}$']
    cols = ['black', 'red', 'blue']

    # loop over cosmology
    for i, direct in enumerate(['', '_fixedc', '_fixedc_fixedT']):
        # load in the data
        data = np.load(
            obs_data_dir / ('redshifts%s/z000_data.npz' % direct))['data']

        # temporary check when I added in Y_{SZ}(r spherical) to understand projection effects
        # if(i==0):
        #    tmp = data[3].copy()
        #    data[3] = data[4]
        #    data[4] = tmp

        # loop over observables (mass_enc, Tmgasv, Mgasv, YSZv)
        for j in range(1, 4):
            # loop over the radius, these will be the r200m multiples
            mult = np.zeros(7)
            slopes = np.zeros(7)
            norms = np.zeros(7)
            pc_scatters = np.zeros(7)
            pc_rbscatters = np.zeros(7)
            rbscatters = np.zeros(7)
            scatters = np.zeros(7)
            msk = data[0, :, 9] > 1e14
            for k in range(6, 13):
                mult[k-6] = radii_definitions[k][1]
                slopes[k-6], norms[k-6], pc_scatters[k-6], pc_rbscatters[k-6], scatters[k-6], rbscatters[k-6] = compute_fit(
                    data[0, msk, k], data[j, msk, k], zero_point=10**14)
                # test to estimate enhanced scatter due to realistic Mgas, as requested by Gus
                if(j == 3 and i == 0):
                    # Tmgasv
                    sigY = np.sqrt(
                        rbscatters[k-6]**2. + (0.036)**2. + (2. * 0.48 * rbscatters[k-6]*0.036))
                    print('Mgas-enhanced scatter at r200m multiple in YSZ', mult[k-6], sigY)
            scat_interp = interp(mult, pc_rbscatters)
            slope_interp = interp(mult, slopes)
            norm_interp = interp(mult, norms)
            mult_arr = np.linspace(0.3, 2.0, 60)
            ax[0, j-1].plot(mult_arr, scat_interp(mult_arr),
                            color=cols[i], label=labs[i])
            ax[1, j-1].plot(mult_arr, slope_interp(mult_arr),
                            color=cols[i], label=labs[i])
            ax[2, j-1].plot(mult_arr, norm_interp(mult_arr),
                            color=cols[i], label=labs[i])

    # Self-similar slopes
    ax[1, 0].axhline(2.0/3.0, color='k', linestyle='dashed')
    ax[1, 1].axhline(1, color='k', linestyle='dashed')
    ax[1, 2].axhline(5.0/3.0, color='k', linestyle='dashed')

    ax[0, 0].set_title(
        r'$T_\mathrm{mg}(<r_\mathrm{ap})$ vs. $M(<r_\mathrm{ap})$')
    ax[0, 1].set_title(
        r'$M_\mathrm{gas}(<r_\mathrm{ap})$ vs. $M(<r_\mathrm{ap})$')
    ax[0, 2].set_title(
        r'$Y_\mathrm{SZ}(<R_\mathrm{ap})$ vs. $M(<r_\mathrm{ap})$')
    ax[0, 2].legend(frameon=False)
    for i in range(0, 3):
        ax[2, i].set_xlabel(r'$r_\mathrm{ap}/r_\mathrm{200m}$')
    ax[0, 0].set_ylabel(r'Percent Scatter')
    ax[1, 0].set_ylabel(r'Slope')
    ax[2, 0].set_ylabel(r'Normalization')
    ax[0, 1].text(1.65, 0.55, r'$z=0$', fontsize=18)
    ax[2, 0].text(0.4, -0.26, r'$\log[\mathrm{keV}]$', fontsize=18)
    ax[2, 1].text(1.3, 13.105, r'$\log[h^{-1}M_\odot]$', fontsize=18)
    ax[2, 2].text(0.4, 0.32, r'$\log[h^{-1}\mathrm{kpc}^2]$', fontsize=18)
    ax[0, 1].text(
        0.45, 0.65, r'$\log(M_\mathrm{200m}/[h^{-1}M_\odot]) \geq 14$', fontsize=18)

    #plt.savefig(fig_dir / 'mass_obs_relations_complexity.pdf', bbox_inches='tight')


plot_3x3_means_varyComplexity()


# In[62]:


def plot_projections():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(
        15, 5), sharex=True, gridspec_kw={'wspace': 0.25, 'hspace': 0.075})
    for i in range(0, 3):
        ax[i].yaxis.set_ticks_position('both')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].tick_params(axis='both', which='minor',
                          colors='black', width=1.0, length=2.0)
        ax[i].set_xlim(0.2, 2.1)
        ax[i].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax[i].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.005))
    ax[1].yaxis.set_minor_locator(MultipleLocator(0.0005))
    ax[2].yaxis.set_minor_locator(MultipleLocator(0.005))

    cosmo = cosmology.setCosmology('planck18')
    labs = ['Full Model', 'Fixed $c_\mathrm{vir}$',
            'Fixed $c_\mathrm{vir}$ and $t_\mathrm{dis}$']
    cols = ['black', 'red', 'blue']
    # loop over cosmology
    for i, direct in enumerate(['']):
        # load in the data
        data = np.load(
            obs_data_dir / ('redshifts%s/z000_data.npz' % direct))['data']
        if(i == 0):
            tmp = data[3].copy()
            data[3] = data[4]
            data[4] = tmp

        # loop over observables (mass_enc, Tmgasv, Mgasv, YSZv)
        for j in range(1, 4):
            # loop over the radius, these will be the r200m multiples
            mult = np.zeros(7)
            slopes = np.zeros(7)
            norms = np.zeros(7)
            pc_scatters = np.zeros(7)
            pc_rbscatters = np.zeros(7)
            rbscatters = np.zeros(7)
            scatters = np.zeros(7)
            msk = data[0, :, 9] > 1e14
            for k in range(6, 13):
                mult[k-6] = radii_definitions[k][1]
                slopes[k-6], norms[k-6], pc_scatters[k-6], pc_rbscatters[k-6], scatters[k-6], rbscatters[k-6] = compute_fit(
                    data[0, msk, k], data[j, msk, k], zero_point=10**14)
            scat_interp = interp(mult, pc_rbscatters)
            slope_interp = interp(mult, slopes)
            norm_interp = interp(mult, norms)
            mult_arr = np.linspace(0.3, 2.0, 60)
            ax[j-1].plot(mult_arr, slope_interp(mult_arr),
                         color=cols[i], label=labs[i])

    # Self-similar slopes
    ax[1].axhline(1, color='k', linestyle='dashed')

    ax[0].set_title(r'$T_\mathrm{mg}(<r_\mathrm{ap})$ vs. $M(<r_\mathrm{ap})$')
    ax[1].set_title(
        r'$M_\mathrm{gas}(<r_\mathrm{ap})$ vs. $M(<r_\mathrm{ap})$')
    ax[2].set_title(r'$Y_\mathrm{SZ}(<r_\mathrm{ap})$ vs. $M(<r_\mathrm{ap})$')
    ax[2].legend(frameon=False)
    for i in range(0, 3):
        ax[i].set_xlabel(r'$r_\mathrm{ap}/r_\mathrm{200m}$')
    ax[0].set_ylabel(r'Slope')
    ax[1].text(1.65, 0.996, r'$z=0$', fontsize=18)

    #plt.savefig(fig_dir / 'mass_obs_relations_complexity_3d.pdf', bbox_inches='tight')


plot_projections()

# this one is just to verify the projection effects from 2D YSZ
# verifies that when YSZ uses spherical aperture, the slope of Tmg and Mgas adds perfectly to YSZ


# In[65]:


def plot_3x3_means_varyZ():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(
        15, 5.5), sharex=True, gridspec_kw={'wspace': 0.25, 'hspace': 0.075})
    for i in range(0, 3):
        ax[i].yaxis.set_ticks_position('both')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].tick_params(axis='both', which='minor',
                          colors='black', width=1.0, length=2.0)
        ax[i].set_xlim(0.2, 2.1)
        ax[i].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax[i].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.25))
    ax[1].yaxis.set_minor_locator(MultipleLocator(0.005))
    ax[2].yaxis.set_minor_locator(MultipleLocator(0.05))

    cosmo = cosmology.setCosmology('planck18')

    zeds = [0., 1.0, 2.0, 3.0]
    cols = sns.cubehelix_palette(len(zeds))

    # loop over cosmology
    for i, z in enumerate(zeds):
        # load in the data
        data = np.load(
            obs_data_dir / ('redshifts/z%03d_data.npz' % int(100*z)))['data']

        mult = np.zeros(7)
        slopes = np.zeros(7)
        norms = np.zeros(7)
        pc_scatters = np.zeros(7)
        pc_rbscatters = np.zeros(7)
        rbscatters = np.zeros(7)
        scatters = np.zeros(7)
        msk = data[0, :, 9] > 1e14
        for k in range(6, 13):
            mult[k-6] = radii_definitions[k][1]
            slopes[k-6], norms[k-6], pc_scatters[k-6], pc_rbscatters[k-6], scatters[k-6], rbscatters[k-6] = compute_fit(
                data[0, msk, k], data[3, msk, k], zero_point=10**14 / ((1+z)**(3./2.))**scaling_factors[2])
        scat_interp = interp(mult, pc_rbscatters)
        slope_interp = interp(mult, slopes)
        norm_interp = interp(mult, norms)
        mult_arr = np.linspace(0.3, 2.0, 60)
        ax[0].plot(mult_arr, scat_interp(mult_arr),
                   color=cols[::-1][i], label=r'$%.0f$' % z)
        ax[1].plot(mult_arr, slope_interp(mult_arr),
                   color=cols[::-1][i], label=r'$%.0f$' % z)
        ax[2].plot(mult_arr, norm_interp(mult_arr),
                   color=cols[::-1][i], label=r'$%.0f$' % z)

    handles, labels = ax[0].get_legend_handles_labels()
    ax[1].legend(handles, labels, frameon=False,
                 title=r'$z=$', title_fontsize=18, loc=1)
    for i in range(0, 3):
        ax[i].set_xlabel(r'$r_\mathrm{ap}/r_\mathrm{200m}$')
    ax[1].axhline(5.0/3.0, color='k', linestyle='dashed')
    ax[0].set_ylabel(r'Percent Scatter')
    ax[1].set_ylabel(r'Slope')
    ax[2].set_ylabel(r'Normalization')
    ax[0].text(
        0.3, 9.35, r'$Y_\mathrm{SZ}(<R_\mathrm{ap})$ -- $M(<r_\mathrm{ap})(1+z)^{3/5}$', fontsize=16)
    ax[0].set_ylim(4, 9.85)
    ax[2].text(0.4, 0.25, r'$\log[h^{-1}\mathrm{kpc}^2]$', fontsize=18)
    ax[0].text(
        0.5, 5.0, r'$\log(M_\mathrm{200m}/[h^{-1}M_\odot]) \geq 14$', fontsize=18)

    #plt.savefig(fig_dir / 'mass_obs_relations_zeds.pdf', bbox_inches='tight')


plot_3x3_means_varyZ()


# In[66]:


def plot_3x3_means_varyM_masscuts():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(
        15, 5.5), sharex=True, gridspec_kw={'wspace': 0.25, 'hspace': 0.075})
    for i in range(0, 3):
        ax[i].yaxis.set_ticks_position('both')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].tick_params(axis='both', which='minor',
                          colors='black', width=1.0, length=2.0)
        ax[i].set_xlim(0.2, 2.1)
        ax[i].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax[i].xaxis.set_major_locator(MultipleLocator(0.5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.25))
    ax[1].yaxis.set_minor_locator(MultipleLocator(0.005))
    ax[2].yaxis.set_minor_locator(MultipleLocator(0.05))

    # loop over cosmology
    cosmo = cosmology.setCosmology('planck18')
    # load in the data
    data = np.load(obs_data_dir / ('planck18_data.npz'))['data']

    msks = [data[0, :, 9] >= 1e12, data[0, :, 9] >= 1e13,
            data[0, :, 9] >= 10**13.5, data[0, :, 9] >= 2e14]
    msk_labels = [r'$12.0$', r'$13.0$', r'$13.5$', r'$14.0$']
    cols = sns.cubehelix_palette(len(msks))

    mult = np.zeros(7)
    slopes = np.zeros(7)
    norms = np.zeros(7)
    pc_scatters = np.zeros(7)
    pc_rbscatters = np.zeros(7)
    rbscatters = np.zeros(7)
    scatters = np.zeros(7)
    for i, msk in enumerate(msks):
        for k in range(6, 13):
            mult[k-6] = radii_definitions[k][1]
            slopes[k-6], norms[k-6], pc_scatters[k-6], pc_rbscatters[k-6], scatters[k-6], rbscatters[k-6] = compute_fit(
                data[0, msk, k], data[3, msk, k], zero_point=10**14)
        scat_interp = interp(mult, pc_rbscatters)
        slope_interp = interp(mult, slopes)
        norm_interp = interp(mult, norms)
        mult_arr = np.linspace(0.3, 2.0, 60)
        ax[0].plot(mult_arr, scat_interp(mult_arr),
                   color=cols[i], label=msk_labels[i])
        ax[1].plot(mult_arr, slope_interp(mult_arr),
                   color=cols[i], label=msk_labels[i])
        ax[2].plot(mult_arr, norm_interp(mult_arr),
                   color=cols[i], label=msk_labels[i])

    handles, labels = ax[0].get_legend_handles_labels()
    ax[2].legend(handles, labels, frameon=False,
                 title=r'$\log(M_\mathrm{200m}/[h^{-1}M_\odot]) \geq$', title_fontsize=18)
    for i in range(0, 3):
        ax[i].set_xlabel(r'$r_\mathrm{ap}/r_\mathrm{200m}$')
    ax[1].axhline(5.0/3.0, color='k', linestyle='dashed')
    ax[0].set_ylabel(r'Percent Scatter')
    ax[1].set_ylabel(r'Slope')
    ax[2].set_ylabel(r'Normalization')
    ax[0].text(
        0.3, 9.3, r'$Y_\mathrm{SZ}(<R_\mathrm{ap})$ -- $M(<r_\mathrm{ap})$', fontsize=16)
    ax[0].set_ylim(4, 9.8)
    ax[2].text(0.4, 0.4, r'$\log[h^{-1}\mathrm{kpc}^2]$', fontsize=18)
    ax[0].text(0.3, 8.8, r'$z=0$', fontsize=18)

    #plt.savefig(fig_dir / 'ysz_m_masscut.pdf', bbox_inches='tight')


plot_3x3_means_varyM_masscuts()


# In[67]:


# load in MC-generated mass accretion histories
# we want to look at the delta_Y / Y vs. MAH, but we can see how this looks for different apertures


def multimah_multiM(z_obs, cosmo, Nmah):
    # loads in an array of MAH from Frank's MAH code, specify Nmah = number of MAH to get
    z_int = int(100*z_obs)
    mah_dir = obs_data_dir / 'redshifts/mah_data'
    fn = mah_dir / ('z%03d_data.npz' % int(100*z_obs))
    if(isfile(fn)):
        d = np.load(fn)
        return d['dat'], d['redshifts'], d['lbtime'], d['masses']
    else:
        masses = 10**np.loadtxt(mah_dir / 'halomasses.dat')[:Nmah]
        dat1 = np.loadtxt(mah_dir / 'MAH0001.dat')
        redshifts = dat1[:, 1]
        lbtime = dat1[:, 2]
        nz = len(dat1)
        dat = np.zeros((Nmah, nz))
        std = np.zeros((nz, 2))
        for i in range(0, Nmah):
            dat[i, :] = 10**np.loadtxt(mah_dir / ('MAH%04d.dat' %
                                                  (i+1)), usecols=3) * masses[i]
        np.savez(fn, dat=dat, redshifts=redshifts,
                 lbtime=lbtime, masses=masses)
        return dat, redshifts, lbtime, masses


cosmo = cosmology.setCosmology('planck18')
mah, zeds, lbtimes, mvirs = multimah_multiM(0.0, cosmo, 9999)

# using this, we will be able to compute MARs according to some definition


# In[68]:


# this uses the same dataset as used on Grace for the planck18 set
# function to compute the mass accretion rate..

cosmo = cosmology.setCosmology('planck18')


def MAR(mah, zeds, lbtimes, zf=0., zi=0.5):
    # delta log(Mvir) / delta log(a)
    # find the index corresponding to z=0.5

    zf_ind = np.where(zeds >= zf)[0][0]
    zi_ind = np.where(zeds >= zi)[0][0]
    zf = zeds[zf_ind]
    zi = zeds[zi_ind]
    t0 = cosmo.age(0)
    tf = t0 - lbtimes[zf_ind]
    ti = t0 - lbtimes[zi_ind]

    # need to compute the concentrations at zf, zi, so we can convert to M200m from Mvir
    mvirs_zf = mah[:, zf_ind]
    mvirs_zi = mah[:, zi_ind]

    concs_zf = np.zeros(len(mah))
    concs_zi = np.zeros(len(mah))

    delta_log_a = -1.*np.log10(1. + zf) - (-1.*np.log10(1. + zi))

    for i in range(0, len(mah)):
        t04_ind_zf = np.where(mah[i, :] > 0.04 * mvirs_zf[i])[0][-1]
        t04_ind_zi = np.where(mah[i, :] > 0.04 * mvirs_zi[i])[0][-1]
        t04_zf = t0 - lbtimes[t04_ind_zf]
        t04_zi = t0 - lbtimes[t04_ind_zi]
        concs_zf[i] = zhao_vdb_conc(tf, t04_zf)
        concs_zi[i] = zhao_vdb_conc(ti, t04_zi)

    # compute the M200ms for each mass at zf, zi
    m200m_zf, _, _ = mass_defs.changeMassDefinition(
        mvirs_zf, c=concs_zf, z=zf, mdef_in='vir', mdef_out='200m')
    m200m_zi, _, _ = mass_defs.changeMassDefinition(
        mvirs_zi, c=concs_zi, z=zi, mdef_in='vir', mdef_out='200m')

    mar = (np.log10(m200m_zf) - np.log10(m200m_zi)) / delta_log_a
    return mar


mars = MAR(mah, zeds, lbtimes)


# In[70]:


# loop over and grab the redshift
# look at redshift where mass falls below psi_res=10^-4

z_psires = np.zeros(len(mah))

for i in range(0, len(mah)):
    z_psires[i] = zeds[np.argwhere(mah[i, :] > 5e-20)[-1][0]]

print(np.min(z_psires), np.max(z_psires))

plot()
plt.hist(z_psires)
plt.ylabel(r'$N$')
plt.xlabel(r'$z(M(z)/M_0 = \psi_\mathrm{res})$')


# In[71]:


# A check for sensitivity to our timesteps used in MAH
# Here, we interpolate the MAH to get more precise Mvir(zf) and Mvir(zi)
# However, we still just use the t04 from the uninterpolated in order to get the
# concentrations needed to convert Mvir to M200m

cosmo = cosmology.setCosmology('planck18')


def MAR_interp(mah, zeds, lbtimes, zf=0., zi=0.5):
    # delta log(Mvir) / delta log(a)

    # loop over the haloes, interpolate against mass
    tf = cosmo.age(zf)
    ti = cosmo.age(zi)

    mvirs_zf = np.zeros(len(mah))
    mvirs_zi = np.zeros(len(mah))
    concs_zf = np.zeros(len(mah))
    concs_zi = np.zeros(len(mah))

    for i in range(0, len(mah)):
        mass_interp = interp(zeds, mah[i, :], k=1)
        mvirs_zf[i] = mass_interp(zf)
        mvirs_zi[i] = mass_interp(zi)

    # for the purpose of computing concentrations:

    zf_ind = np.where(zeds >= zf)[0][0]
    zi_ind = np.where(zeds >= zi)[0][0]
    zf = zeds[zf_ind]
    zi = zeds[zi_ind]
    t0 = cosmo.age(0)
    tf = t0 - lbtimes[zf_ind]
    ti = t0 - lbtimes[zi_ind]

    delta_log_a = -1.*np.log10(1. + zf) - (-1.*np.log10(1. + zi))

    for i in range(0, len(mah)):
        t04_ind_zf = np.where(mah[i, :] > 0.04 * mvirs_zf[i])[0][-1]
        t04_ind_zi = np.where(mah[i, :] > 0.04 * mvirs_zi[i])[0][-1]
        t04_zf = t0 - lbtimes[t04_ind_zf]
        t04_zi = t0 - lbtimes[t04_ind_zi]
        concs_zf[i] = zhao_vdb_conc(tf, t04_zf)
        concs_zi[i] = zhao_vdb_conc(ti, t04_zi)

    # compute the M200ms for each mass at zf, zi
    m200m_zf, _, _ = mass_defs.changeMassDefinition(
        mvirs_zf, c=concs_zf, z=zf, mdef_in='vir', mdef_out='200m')
    m200m_zi, _, _ = mass_defs.changeMassDefinition(
        mvirs_zi, c=concs_zi, z=zi, mdef_in='vir', mdef_out='200m')

    mar = (np.log10(m200m_zf) - np.log10(m200m_zi)) / delta_log_a
    return mar


mars_interp = MAR_interp(mah, zeds, lbtimes)


# In[73]:


# compute redshift consistent with dynamical time defined in Diemer et al.
# no mass dependence on this

cosmo = cosmology.setCosmology('planck18')

m200m = 10**14  # M_200m WLOG in Msun/h
zobs = 0.0
r200m = mass_so.M_to_R(m200m, zobs, '200m')  # in kpc/h
tdyn_diemer = 2. * (r200m**3 / (G*m200m))**(1./2.) *     km_per_kpc / (cosmology.getCurrent().H0 / 100.) / s_per_Gyr
# This is the proper time corresponding to where we want to look back
prop_time = cosmo.age(0) - tdyn_diemer
z_dyn = cosmo.age(prop_time, inverse=True)  # redshift tdyn ago
print(z_dyn)


# In[74]:


# Comparison of interpolator case vs. discrete, showing insignificant difference

planckdata = np.load(obs_data_dir / 'planck18_data.npz')['data']
mars = MAR(mah, zeds, lbtimes, zf=0., zi=z_dyn)
mars_interp = MAR_interp(mah, zeds, lbtimes, zf=0., zi=z_dyn)

msk = planckdata[0, :, 9] >= 1e14
plot()
plt.hist(mars[msk], bins='auto', density='normed', histtype='step', color='k')
plt.hist(mars_interp[msk], bins='auto',
         density='normed', histtype='step', color='r')
plt.xlabel(r'$\Gamma$')
plt.ylabel(r'pdf')

# the distribution itself is not changed at all basically


# In[75]:


# compare MARs to YSZ residuals

planckdata = np.load(obs_data_dir / 'planck18_data.npz')['data']
mars = MAR(mah, zeds, lbtimes, zf=0., zi=z_dyn)

msk = planckdata[0, :, 9] >= 1e14

fig = plt.figure(figsize=(7, 5.8))
ax = fig.add_subplot(111, label="1")
ax.tick_params(axis='both', which='minor',
               colors='black', width=1.0, length=3.0)
ax2 = fig.add_subplot(111, label="2", frame_on=False)
ax2.tick_params(axis='both', which='minor',
                colors='black', width=1.0, length=3.0)

ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax2.xaxis.set_minor_locator(MultipleLocator(0.025))
ax2.yaxis.set_minor_locator(MultipleLocator(0.25))

ax.hist(mars[msk], bins='auto', density='normed', histtype='step', color='k')
ax.set_ylabel(r'$\Gamma$ Probability Density')
ax.set_xlabel(r'$\Gamma$')
ax.text(2.5, 0.38, r'$z=0$', fontsize=18)
ax.text(
    2.5, 0.42, r'$\log(M_\mathrm{200m}/[h^{-1}M_\odot]) \geq 14$', fontsize=18)

aperture = 9  # R200m

slope, norma, pc_scatter, pc_rbscatter, mape, rbscatter = compute_fit(
    planckdata[0, msk, aperture], planckdata[3, msk, aperture], zero_point=1.)
preds = 10**norma * planckdata[0, msk, aperture]**slope
resids = np.log(planckdata[3, msk, aperture] / preds)
ax2.hist(resids, bins='auto', density='normed', histtype='step', color='r')
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel(
    r'$\mathcal{R} \equiv \ln(Y_\mathrm{SZ,200m,true}) - \ln(Y_\mathrm{SZ,200m,fit})$', color='r')
ax2.set_ylabel(r'$\mathcal{R}$ Probability Density', color='r')
ax2.xaxis.set_label_position('top')
ax2.yaxis.set_label_position('right')

#plt.savefig(fig_dir / 'resids_and_mars_pdfs.pdf', bbox_inches='tight')


# In[77]:


# Diemer peak height-scaling
# let's see how good it looks when we divide out the median

def diemer_medGamma(M200m, z):
    A = 1.2222 + 0.3515*z
    B = -0.2864 + 0.0778*z - 0.0562*z**2 + 0.0041*z**3
    nu200m = peaks.peakHeight(M200m, z)
    return A*nu200m + B*nu200m**(3./2.)
    

from matplotlib import cm
import matplotlib as mpl

fig, ax = plt.subplots(nrows=1,ncols=4,figsize=(15,5), sharey=True, sharex=True, gridspec_kw={'wspace':0.02})
for i in range(0,4):
    ax[i].yaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
    ax[i].yaxis.set_minor_locator(MultipleLocator(0.5))
    ax[i].xaxis.set_minor_locator(MultipleLocator(0.05))

zzs = [0.0, 1.0, 2.0, 3.0]
all_resids = []
all_mars = []
rad_ind = 9 #R200m
for i,zobs in enumerate(zzs):
    planckdata = np.load(obs_data_dir / ('redshifts/z%03d_data.npz' % int(100*zobs)))['data']
    mah, zeds, lbtimes, mvirs = multimah_multiM(zobs, cosmo, 9999)
    mtest = 10**15
    r200m = mass_so.M_to_R(mtest, zobs, '200m') # in kpc/h
    tdyn_diemer = 2. * (r200m**3 / (G*mtest))**(1./2.) * km_per_kpc / (cosmology.getCurrent().H0 / 100.) / s_per_Gyr
    prop_time = cosmo.age(zobs) - tdyn_diemer # This is the proper time corresponding to where we want to look back
    zdyn_from_zobs = cosmo.age(prop_time, inverse=True)
    
    mars = MAR(mah, zeds, lbtimes, zf = zobs, zi=zdyn_from_zobs)
    
    msk = planckdata[0,:,9]>=1e14
    mah = mah[msk]
    mvirs = mvirs[msk]
    mars = mars[msk]
    
    coeffs = np.polyfit(np.log10(planckdata[0,msk,rad_ind]), np.log10(planckdata[3,msk,rad_ind]), deg=1) # Y_SZ - M reln
    preds = 10**(coeffs[0]*np.log10(planckdata[0,msk,rad_ind]) + coeffs[1]) # Y_SZ predictions
    resids = np.log(planckdata[3,msk,rad_ind] / preds) # instead of delta_Y / Y, it would be log resid?
    print(coeffs)
    cvirs = conc_interps[zobs](mvirs)
    m200ms = mass_defs.changeMassDefinition(mvirs, cvirs, zobs, 'vir', '200m')[0]
    mars = mars - diemer_medGamma(m200ms, zobs)
    ax[i].scatter(resids, mars, c=np.log10(m200ms), rasterized=True, cmap=plt.get_cmap('magma_r'), s=3)
    all_mars.append(mars)
    all_resids.append(resids)
    coeffs_mars = np.polyfit(resids, mars, deg=1)
    print(coeffs_mars)
    print(np.corrcoef(resids, mars))
    print(spearmanr(resids, mars))
    
    rhos = np.corrcoef(resids, mars)[1,0]
    rs = spearmanr(resids, mars)[0]
    
    ax[i].text(-0.45, -2.5, r'$r_s = %.2f$' % rs, fontsize=18)
    ax[i].text(-0.45, -3.5, r'$\rho = %.2f$' % rhos, fontsize=18)

    ax[i].plot(resids, (coeffs_mars[0]*resids + coeffs_mars[1]), color='k', label=r'$%.1f$' % zobs)
    ax[i].set_title(r'$z=%.0f$' % zobs)

for i in range(0,4):
    ax[i].set_xticks([-0.4, -0.2, 0., 0.2])

all_mars = np.concatenate(all_mars)
all_resids = np.concatenate(all_resids)
coeffs_mars = np.polyfit(all_resids, all_mars, deg=1)
print(coeffs_mars)
print(np.corrcoef(all_resids, all_mars)) # 0.4
print(spearmanr(all_resids, all_mars))

cmap = plt.get_cmap('magma_r')
norm = mpl.colors.Normalize(vmin=np.min(np.log10(m200ms)), vmax=np.max(np.log10(m200ms)))
cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])
cbar = mpl.colorbar.ColorbarBase(ax=cbar_ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cbar.ax.set_ylabel(r'$\log(M_\mathrm{200m}/[h^{-1}M_\odot])$')

cortots = spearmanr(all_mars, all_resids).correlation
cortotp = np.corrcoef(all_mars, all_resids)[0,1]
ax[0].text(-0.47, 8.25, r'$\log(M_\mathrm{200m}/[h^{-1}M_\odot]) \geq 14$', fontsize=18)

fig.text(0.52, 0.01, r'$\mathcal{R} \equiv \ln(Y_\mathrm{SZ,200m,true}) - \ln(Y_\mathrm{SZ,200m,fit})$', ha='center', fontsize=18)
fig.text(0.08, 0.5, r'$\Gamma - \Gamma^*$', va='center', rotation='vertical', fontsize=18)


#plt.savefig(fig_dir / 'resids_vs_mar.pdf', bbox_inches='tight', dpi=300)


# In[78]:


# Diagnostic plot to look at covariance as requested by Gus
# Omitted from Paper 1, but we'll look at this in more detail in the next paper, where
# we plan to include a more realistic model of the gas density

import pandas as pd
import scipy.stats as stats


def plot_covar():

    zobs = 0.0
    rad_ind = 9
    print('Using radius definition of', radii_definitions[rad_ind])

    cosmo = cosmology.setCosmology('planck18')
    planckdata = np.load(obs_data_dir/'redshifts/z000_data.npz')['data']
    mah, zeds, lbtimes, mvirs = multimah_multiM(zobs, cosmo, 9999)
    mtest = 10**15
    r200m = mass_so.M_to_R(mtest, zobs, '200m')  # in kpc/h
    tdyn_diemer = 2. * (r200m**3 / (G*mtest))**(1./2.) *         km_per_kpc / (cosmology.getCurrent().H0 / 100.) / s_per_Gyr
    # This is the proper time corresponding to where we want to look back
    prop_time = cosmo.age(zobs) - tdyn_diemer
    zdyn_from_zobs = cosmo.age(prop_time, inverse=True)
    mars = MAR(mah, zeds, lbtimes, zf=zobs, zi=zdyn_from_zobs)

    msk = planckdata[0, :, 9] >= 1e14
    mah = mah[msk]
    mvirs = mvirs[msk]
    mars = mars[msk]

    m200ms = planckdata[0, msk, 9]

    cvirs = conc_interps[zobs](mvirs)
    mars = mars - diemer_medGamma(m200ms, zobs)

    # now we need to regress over each of the other properties to get the residuals
    resids = []
    for j in range(1, 4):
        slope, norm, _, _, _, _ = compute_fit(
            planckdata[0, msk, rad_ind], planckdata[j, msk, rad_ind], zero_point=1.)
        preds = 10**(slope*np.log10(planckdata[0, msk, rad_ind]) + norm)
        resids.append(np.log(planckdata[j, msk, rad_ind] / preds))

    # now we have Tmgas, Mgas, YSZ, and then append mars
    resids.append(mars)
    resids = np.array(resids).T
    resids[:, [0, 1]] = resids[:, [1, 0]]
    resids = pd.DataFrame(data=resids, columns=[r'$\mathcal{R}(M_\mathrm{gas})$', r'$\mathcal{R}(T_\mathrm{mg})$',
                                                r'$\mathcal{R}(Y_\mathrm{SZ}$)', r'$\Gamma - \Gamma^*$'])

    def corrfunc(x, y, **kws):
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)

    g = sns.pairplot(resids, kind='reg', markers='.', plot_kws={'marker': '.', 'scatter_kws': {'s': 2, 'color': 'gray', 'rasterized': True}, 'line_kws': {
                     'color': 'k', 'linewidth': 1}}, diag_kws={'color': 'k', 'bins': 30, 'histtype': 'step'})
    g.map_lower(corrfunc)
    #plt.savefig(fig_dir / 'covar.pdf', bbox_inches='tight')


plot_covar()


# ## Variance and mean MAR vs. M, z

# In[160]:


from useful_functions.utils import autobin
from scipy.interpolate import InterpolatedUnivariateSpline

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(
    7, 11.6), sharex=True, gridspec_kw={'wspace': 0.25, 'hspace': 0.075})
for i in range(0, 2):
    ax[i].yaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].tick_params(axis='both', which='minor',
                      colors='black', width=1.0, length=2.0)
    ax[i].xaxis.set_minor_locator(MultipleLocator(0.1))
    ax[i].xaxis.set_major_locator(MultipleLocator(0.5))

cosmo = cosmology.setCosmology('planck18')

zeds = [0., 1.0, 2.0, 3.0]
cols = sns.cubehelix_palette(len(zeds))[::-1]

# loop over redshifts
for i, z in enumerate(zeds):
    # load in the data
    data = np.load(
        obs_data_dir / ('mar_vs_m200m/mars_z%03d_data.npz' % int(100*z)))

    # bin by mass
    m200ms = data['m200ms']
    mar200mdyns = data['mars']

    mass_edges, mass_cents_1, avgs = autobin(
        np.min(m200ms), np.max(m200ms), 5, m200ms, mar200mdyns, typ='average')
    mass_edges, mass_cents_2, stds = autobin(
        np.min(m200ms), np.max(m200ms), 4, m200ms, mar200mdyns, typ='1684')

    # smooth the curve
    avg_interp = InterpolatedUnivariateSpline(np.log10(mass_cents_1), avgs)
    std_interp = InterpolatedUnivariateSpline(
        np.log10(mass_cents_2), stds, k=2)
    log10mv = np.log10(np.logspace(
        np.min(np.log10(mass_cents_1)), np.max(np.log10(mass_cents_1)), 15))

    # plot std and mean
    ax[0].plot(log10mv, avg_interp(log10mv),
               color=cols[i], label=r'$%.0f$' % z)
    ax[1].plot(log10mv, std_interp(log10mv), color=cols[i])

ax[1].set_xlabel(r'$\log_{10}(M_\mathrm{200m}/[h^{-1}M_\odot])$')
ax[0].set_ylabel(r'Mean $\Gamma$')
ax[1].set_ylabel(r'Dispersion in $\Gamma$')
ax[0].legend(frameon=False, title=r'$z=$', title_fontsize=18)

plt.savefig(fig_dir / 'gamma_mean_std.pdf', bbox_inches='tight')

