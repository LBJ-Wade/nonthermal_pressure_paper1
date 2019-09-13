#!/usr/bin/env python
# coding: utf-8

# In[262]:


import numpy as np
import matplotlib.pyplot as plt
import colossus
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_so, profile_nfw
from plotter import plot, loglogplot
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import subprocess
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# will need to test sensitivity to:
# 1. c(M,z) relationship: Zhao+09, Duffy+08, Diemer+19


# In[181]:


# global variables
# free parameters of Shi+14 model; can be changed later
beta_def = 1.0
eta_def  = 0.7

G = colossus.utils.constants.G
cm_per_km = 1e5
km_per_kpc = colossus.utils.constants.KPC / cm_per_km # KPC was in cm
s_per_Gyr = colossus.utils.constants.GYR
yr_per_Gyr = 1E9

# can reset cosmology on-the-fly, will start with WMAP 5
cosmo = cosmology.setCosmology('WMAP5')
cosmo_astro = FlatLambdaCDM(H0=cosmology.getCurrent().H0 * u.km / u.s / u.Mpc, Tcmb0=cosmology.getCurrent().Tcmb0 * u.K, Om0=cosmology.getCurrent().Om0, Neff=cosmology.getCurrent().Neff, Ob0=cosmology.getCurrent().Ob0)
print(cosmology.getCurrent())


# In[182]:


# mass-concentration relationships
# Original Shi paper used duffy08 and vdB+14 used the c(M,z) of Zhao+09
# need to make sure that we use consistent definitions of virial radius, thus concentration, everywhere
# takes the z=0 mass in units of Msun/h
nm = 50
masses = np.logspace(12,15,nm)
cvirs = concentration.concentration(masses, 'vir', 0.0, model = 'diemer19')
c200cs = concentration.concentration(masses, '200c', 0.0, model = 'diemer19')
c200ms = concentration.concentration(masses, '200m', 0.0, model = 'diemer19')
plot(semilogx=True)
plt.plot(masses, cvirs, label=r'$c_\mathrm{vir}$')
plt.plot(masses, c200cs, label=r'$c_\mathrm{200c}$')
plt.plot(masses, c200ms, label=r'$c_\mathrm{200m}$')
plt.title(r'Concentrations using Diemer+19 Model')
plt.legend()
plt.xlabel(r'$M_\odot / h$')
plt.ylabel(r'$c$')

# TODO: vdB+14 uses Zhao et al. 2009, not available in Colossus, so may need to code that up
# this uses the lookback time instead of redshift, so need to get z from t
# NOTE: vdB+14 paper and code use slightly different formulas
# We probably want to use one of the all-cosmology c(M,z) since we will use different definitions of the virial radius


# In[8]:


# computing t_d from t_dyn
# the masses are in Msun/h
# the lengths for haloes are in kpc/h
def NFWf(x):
    return np.log(1. + x) - x/(1. + x)

# accepts radius in physical kpc/h
# might change this...
def NFWM(r, M, z, conc_model='diemer19', mass_def='vir'):
    # compute the concentration
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    return M * NFWf(c*r/R) / NFWf(c)

# need to fix unit conversions here...
def t_d(r, M, z, conc_model='diemer19', mass_def='vir', beta=beta_def):
    Menc = NFWM(r, M, z, conc_model, mass_def)
    t_dyn = 2. * np.pi * (r**3 / (G*Menc))**(1./2.) * km_per_kpc / (cosmology.getCurrent().H0 / 100.)
    return beta * t_dyn


# In[9]:


# look at t_d vs. r for a 10^14 Msun/h halo to verify the results

mass = 1e14 #Msun/h
Rvir = mass_so.M_to_R(mass, 0.0, 'vir')
nr = 30
rads = np.logspace(np.log10(0.01*Rvir),np.log10(Rvir), nr)

dt = mass_so.dynamicalTime(0.0, 'vir', definition='orbit')
print(dt / 2)

loglogplot()
plt.plot(rads/Rvir, t_d(rads, mass, 0.0, mass_def='vir')/s_per_Gyr, label=r'Our code')
plt.plot([1.], [dt], '*', label=r'Colossus Orbit $t_\mathrm{dyn}$')
plt.legend()
plt.xlabel(r'$r / r_\mathrm{vir}$')
plt.ylabel(r'$t_\mathrm{d}$ [Gyr]')
# it agrees, so our calculation seems to be correct
# it disagrees by ~factor of 2~ with Shi and Komatsu, which is explained in their erratum
# hence, our result is indeed correct
# need to check later results to see if they actually use beta*t_dyn / 2 or beta*t_dyn for further stuff


# In[10]:


# Komatsu and Seljak Model
# Here, c_nfw is defined using the virialization condition of Lacey & Cole (1993) and Nakamura & Suto (1997)
# should make sure that it is pretty similar to Bryan and Norman

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

# this now agrees with Komatsu and Seljak eqn 19 for theta
# the confusion was that rho0 / P0 is 3*eta0^-1 and then units removed by scaling by R/GM
def theta(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    # the rho0/P0 is actually 3eta^-1(0) * R/(GM) from Komatsu and Seljak
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    return 1. + ((Gamma(c) - 1.) / Gamma(c))*rho0_by_P0*(NFWPhi(0, M, z, conc_model='diemer19', mass_def='vir')-NFWPhi(r, M, z, conc_model='diemer19', mass_def='vir'))

# arbitrary units for now while we figure out what to do with the normalization
# likely won't need this
def rho_gas(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    return theta(r, M, z, conc_model, mass_def)**(1.0 / (Gamma(c) - 1.0))

# in km/s
def sig2_tot_obsolete(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    return (1.0 / rho0_by_P0) * theta(r, M, z, conc_model, mass_def)

# the complete sig2_tot that only makes one of each relevant function call:
def sig2_tot(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model) # tabulated probably
    R = mass_so.M_to_R(M, z, mass_def) # tabulated probably
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gamma(c) - 1.) / Gamma(c)) * 3. *eta0(c)**-1 * (phi0 - phir)
    return (1.0 / rho0_by_P0) * theta

# concentration increases with time
# radius also increases with time (hence why c increases)


# In[11]:


concs = np.linspace(4,15,10)
plot()
plt.plot(concs, eta0(concs))
plt.xlabel(r'$c_\mathrm{vir}$'); plt.ylabel(r'$\eta(0)$ from Komatsu + Seljak')
plot()
plt.plot(concs, ((Gamma(concs) - 1.) / Gamma(concs)))
plt.xlabel(r'$c_\mathrm{vir}$'); plt.ylabel(r'$\frac{\Gamma}{\Gamma - 1}$')


# In[12]:


mass = 10**14.5 #Msun/h
Rvir = mass_so.M_to_R(mass, 0.0, 'vir')
nr = 30
rads = np.logspace(np.log10(0.01*Rvir),np.log10(Rvir), nr)

plot(semilogx=True)
plt.plot(rads/Rvir, (sig2_tot(rads, mass, 0.0, mass_def='vir'))**(1./2.))
plt.xlabel(r'$r / r_\mathrm{vir}$')
plt.ylabel(r'$\sigma_\mathrm{tot}$ [km/s]')

# what happens to Komatsu-Seljak model when you have c < 6.5?

# we now have recovered the result of Fig 4 from Komatsu and Seljak


# In[255]:


print(cosmo)


# In[270]:


# function that returns an average MAH given the input cosmology, mass, and z_obs

exec_name = 'mandc.x'

def zhao_mah(mass, z_obs, cosmo):
    instring = 'nthpre\n%.3f %.3f\n1\n%.3f\n%.3f\n%.3f\n%.4f %1.3f\n1\n%1.1f\n%2.1f' % (cosmo.Om0, cosmo.Ode0, cosmo.H0/100., cosmo.sigma8, cosmo.ns, cosmo.Ob0, cosmo.Tcmb0, z_obs, np.log10(mass))
    command = 'ls -l' #'./%s < %s' % (exec_name, instring)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)
zhao_mah(10**15, 0.0, cosmo)    


# In[ ]:


# assuming that 


# In[ ]:


# up next: compute dsigma_tot^2 / dt using Shi+14 eqn 19
# then: compute t_growth, sigma_nth (figure out how to deal with changing radial bins)
# then: MAH
# then: solve for evolution of sigma_nth


# In[ ]:


# want to reproduce Shi/Komatsu Figs 2,3,4 (and then something similar to 1 from vdB+14)
# we've so far got z=0 case for Fig 2 and top pane of 4, need to figure out t_growth and then sigma_nth


# In[ ]:


# general pipeline will look like the following:
# for cosm in cosmologies:
#      for mass in halo_masses:
#                # set initial conditions
#           for t in times:
#                get the t_d values for each radius and evolve all radii forward one timestep


# In[24]:


# loading Zhao+09 sample mass trajectory to see if we can reproduce Fig 1
zhao_dat = np.loadtxt('sample_mah_1e15',skiprows=1)
vdb_dat  = np.loadtxt('sample_mah_1e15_vdb')

M0 = 1e15

loglogplot()
plt.plot(1+zhao_dat[:,0],zhao_dat[:,1])
plt.plot(1+vdb_dat[:,1],10**vdb_dat[:,3]*M0)
plt.xlim(1,10)
plt.ylim(10**11,10**15)
loglogplot()
plt.plot(zhao_dat[:,-1] / (cosmology.getCurrent().H0 / 100.) / 10**9,zhao_dat[:,1])
plt.plot(zhao_dat[0,-1] / (cosmology.getCurrent().H0 / 100.) / 10**9 - vdb_dat[:,2],10**vdb_dat[:,3]*M0)
plt.xlim(0.5,15.)
plt.ylim(10**11,10**15)

# this looks bang-on, so let's see how similar it looks to vdB median MAH
# vdB final columns gives c and dM/dt (column 7 and 8 starting at 1)


# In[220]:


# now that we have a sample MAH vs. time, we want to write a code that takes an interpolated MAH
# which can just be a univariate interpolator M(t), which we can take derivate of to get dM/dt
# or if we use vdB, we can simply interpolate the result

# goal is to exactly reproduce the Fig 4 lower panel for the 1e15 halo at z=0 before we move on to use vdB and then
# to write the integrator

# then, using this, we compute dsigma^2/dt

# we'll do everything using time
cosmo.age(1.5) # Gyr

# compute dsigma^2/dt at z=0 using current dM/dt, can use vdB's
t0 = cosmo.age(0)
delta_t = 0.01
print(cosmo.age(t0+delta_t, inverse=True))
# need the new redshift at this updated time...

# take the two redshifts, dt is in Gyr, so need to convert dMdt to per Gyr
# once we're happy with this, can just give it fixed dt/dz/dM
def dsig2_dt(r, z1, z2, dt, M, dM, dMdt, conc_model='diemer19', mass_def='vir'):
    part_t = (sig2_tot(r, M, z2, conc_model, mass_def) - sig2_tot(r, M, z1, conc_model, mass_def)) / dt
    part_M = (sig2_tot(r, M+dM, z1, conc_model, mass_def) - sig2_tot(r, M, z1, conc_model, mass_def)) / dM
    return part_t + part_M * dMdt

def sig2_diff(r, z1, z2, dt, M, dM, dMdt, conc_model='diemer19', mass_def='vir'):
    return sig2_tot(r, M+dM, z1, conc_model, mass_def) - sig2_tot(r, M, z1, conc_model, mass_def)

def sig2_diff_t(r, z1, z2, dt, M, dM, dMdt, conc_model='diemer19', mass_def='vir'):
    return sig2_tot(r, M, z2, conc_model, mass_def) - sig2_tot(r, M, z1, conc_model, mass_def)


# In[219]:


# compute at the present day
mass = 10**15 #Msun/h
Rvir = mass_so.M_to_R(mass, 0.0, 'vir')
nr = 30
rads = np.logspace(np.log10(0.01*Rvir),np.log10(Rvir), nr)

z1 = 0.001
t1 = cosmo.age(z1)
print(t1)
dt = 0.02 # Gyr
z2 = cosmo.age(t1+dt, inverse=True)
dM = 10**5 # let's see if sensitive to this...
print(z2, z1)

# z=0 dM/dt from vdB data
dMdt = vdb_dat[0,-1] * yr_per_Gyr

ds2dt = dsig2_dt(rads, z1, z2, dt, mass, dM, dMdt)
s2diff = sig2_diff(rads, z1, z2, dt, mass, dM, dMdt)
s2difft = sig2_diff_t(rads, z1, z2, dt, mass, dM, dMdt)

plot(semilogx=True)
plt.plot(rads/Rvir, ds2dt)
plt.xlabel(r'$r / r_\mathrm{vir}$')
plt.ylabel(r'$\mathrm{d}\sigma_\mathrm{tot}^2 /\mathrm{d}t$ [(km/s)$^2$/Gyr]')

plot(semilogx=True)
plt.plot(rads/Rvir, s2diff / dM)
plt.xlabel(r'$r / r_\mathrm{vir}$')

plot(semilogx=True)
plt.plot(rads/Rvir, s2difft / dt)
plt.xlabel(r'$r / r_\mathrm{vir}$')

# need to figure out how to deal with the negatives...

tgrowth = sig2_tot(rads, mass, z1, mass_def='vir') / np.abs(ds2dt)
loglogplot()
plt.plot(rads/Rvir, tgrowth)
plt.xlabel(r'$r / r_\mathrm{vir}$')

# this looks very similar to the Shi+2014 paper, but seems to be sensitive to the concentration model
# one last thing to try: see how things change if we use the concentrations from the Zhao+09 model
# however, it explicitly says that they use the Duffy model, so that should look the same

# should be getting around ~17 Gyr for 10^15 Msun, so let's see if we can figure out the cause of this


# In[ ]:


# framework for integrating one halo forward to get sig_nth^2 at z=0
# set t_0 = cosmo.age(z_i) where z_i = 6
# set t_f = cosmo.age(z_0) where z_0 = 0
# set n_steps, then tvals = np.linspace(t_0, t_f, n_steps+1) # see if logspace works better...
# set zvals = cosmo.age(tvals)
# delta_t = tvals[1:] - tvals[:-1]
# for i in range(0, n_steps):
#     compute dsigma2_dt

z_i = 

t_0 = cosmo.age(z_i)


# In[171]:


masses = 10**vdb_dat[:,3] * mass
zeds = vdb_dat[:,1]
msk = zeds <= 6
vdb_concs = vdb_dat[:,6]
concs = np.zeros(len(vdb_dat))
for i in range(0,len(concs)):
    concs[i] = concentration.concentration(masses[i], 'vir', zeds[i], model = 'diemer19')

plot(semilogx=True)
plt.plot(masses[msk], concs[msk], label='Colossus')
plt.plot(masses[msk], vdb_concs[msk], label='vdB')
plt.legend()


# In[93]:


# can try interpolating dM/dt
# then, if we know the amount of mass gained vs. time, we can update time and update mass together
# this result SHOULD agree with the previous method, so I need to figure that out still

# interpolated in redshift?
dmdt_interp = interp(vdb_dat[:,1], vdb_dat[:,-1], k=3)
zeds = np.linspace(0.,10., 1000)
loglogplot()
plt.plot(1+zeds, dmdt_interp(zeds), '.-') # this seems to behave oddly...
plt.plot(1+vdb_dat[:,1], vdb_dat[:,-1], '.-')


# In[165]:


# now, we can try to compute the actual total change in sigma^2 and see if that works
mass = 10**15 #Msun/h
Rvir = mass_so.M_to_R(mass, 0.0, 'vir')
nr = 30
rads = np.logspace(np.log10(0.01*Rvir),np.log10(Rvir), nr)

z1 = 0.0001
t1 = cosmo.age(z1)
print(t1)
dt = -0.05 # Gyr
z2 = cosmo.age(t1+dt, inverse=True)

dmdt_interp = interp(vdb_dat[:,1], vdb_dat[:,-1] * yr_per_Gyr, k=3)

print(np.log10(-1.*dmdt_interp(z1)*dt)) # change by 10^8 msun...

sig2_1 = sig2_tot(rads, mass, z1, conc_model='diemer19', mass_def='vir')
sig2_2 = sig2_tot(rads, mass + dmdt_interp(z1)*dt, z2, conc_model='diemer19', mass_def='vir')
dsig2_dt = (sig2_1 - sig2_2) / dt

plot(semilogx=True)
plt.plot(rads/Rvir, dsig2_dt)
plt.xlabel(r'$r / r_\mathrm{vir}$')
plt.ylabel(r'$\mathrm{d}\sigma_\mathrm{tot}^2 /\mathrm{d}t$ [(km/s)$^2$/Gyr]')


# In[106]:


# so for both approaches, we have the issue that it is sensitive to the dt... need to figure that out
# should be able to plot, at fixed r, and then change time to see what is going on...
# let's hold r and mass constant and then change redshift linearly in t, then plot sig^2 vs t
# can do similarly at fixed z and then vary the mass alone

#varying time
mass = 1E15 # Msun/h
Rvir = mass_so.M_to_R(mass, 0.0, 'vir')
# let's look at 0.1 Rvir for an example
rad = 0.1 * Rvir

z1 = -0.005 #0.0001
t1 = cosmo.age(z1)
z2 = 0.0001 #1.5
t2 = cosmo.age(z2)

# the function seems well-behaved...

nt = 100
tvals = np.linspace(t2, t1, nt)
sig2_vals = np.zeros(nt)
for i in range(0,nt):
    sig2_vals[i] = sig2_tot(rad, mass, cosmo.age(tvals[i], inverse=True), conc_model='diemer19', mass_def='vir')
    
loglogplot()
plt.plot(tvals, sig2_vals) # as time goes on, it clearly decreases

# this is well-behaved, interestingly, so we need to figure out why my estimations of partial sig2 partial t is sensitive
# to partial t

# it seems that the difference between sig^2 approaches a constant as I decrease delta_t, so the derivative goes up
# by the corresponding factor associated with delta_t^-1, so this needs to be figured out


# In[188]:


mass = 10**15 #Msun/h
Rvir = mass_so.M_to_R(mass, 0.0, 'vir')
nr = 30
rads = np.logspace(np.log10(0.01*Rvir),np.log10(Rvir), nr)

z1 = 0.0001
t1 = cosmo.age(z1)
print(t1)
dt = 0.1 # Gyr
print(t1, t1+dt)
#z2 = cosmo.age(t1+dt, inverse=True) # this is the reason why... the redshift is not actually updating
print(z1,z2)

# z=0 dM/dt from vdB data
dMdt = vdb_dat[0,-1] * yr_per_Gyr

def sig2_diff_t(r, z1, z2, dt, M, conc_model='diemer19', mass_def='vir'):
    return sig2_tot(r, M, z2, conc_model, mass_def) - sig2_tot(r, M, z1, conc_model, mass_def)

dt1 = 0.02 # we've gone beyond and are only using the interpolation table...
dt2 = 0.04
print(z_at_value(cosmo_astro.age, (t1-dt1) * u.Gyr), z_at_value(cosmo_astro.age, (t1-dt2) * u.Gyr))
s2difft_1 = sig2_diff_t(rads, z1, z_at_value(cosmo_astro.age, (t1-dt1) * u.Gyr), dt1, mass)
s2difft_2 = sig2_diff_t(rads, z1, z_at_value(cosmo_astro.age, (t1-dt2) * u.Gyr), dt2, mass)

# this seems to work okay


plot(semilogx=True)
plt.plot(rads/Rvir, s2difft_1 / dt1)
plt.plot(rads/Rvir, s2difft_2 / dt2)
plt.xlabel(r'$r / r_\mathrm{vir}$')

plot(semilogx=True)
plt.plot(rads/Rvir, (s2difft_1 / dt1 - s2difft_2 / dt2) / (s2difft_1 / dt1)) # sub percent-level agreement


# In[191]:


mass = 10**15 #Msun/h
Rvir = mass_so.M_to_R(mass, 0.0, 'vir')
nr = 30
rads = np.logspace(np.log10(0.01*Rvir),np.log10(Rvir), nr)

z1 = 0.0001
t1 = cosmo.age(z1)
print(t1)
dt = 0.1 # Gyr
print(t1, t1+dt)
#z2 = cosmo.age(t1+dt, inverse=True) # this is the reason why... the redshift is not actually updating
print(z1,z2)

# z=0 dM/dt from vdB data
dMdt = vdb_dat[0,-1] * yr_per_Gyr

def sig2_diff_t(r, z1, z2, dt, M, conc_model='diemer19', mass_def='vir'):
    return sig2_tot(r, M, z2, conc_model, mass_def) - sig2_tot(r, M, z1, conc_model, mass_def)

dt1 = 0.02 # we've gone beyond and are only using the interpolation table...
dt2 = 0.04

# so timesteps of 0.05 Gyr should be fine...

z_z1 = cosmo.age(t1+dt1, inverse=True)
z_z2 = cosmo.age(t1+dt2, inverse=True)
print(z_z1, z_z2, 'this')

print(z_at_value(cosmo_astro.age, (t1-dt1) * u.Gyr), z_at_value(cosmo_astro.age, (t1-dt2) * u.Gyr))
s2difft_1 = sig2_diff_t(rads, z1, z_z1, dt1, mass)
s2difft_2 = sig2_diff_t(rads, z1, z_z2, dt2, mass)

# this seems to work okay


plot(semilogx=True)
plt.plot(rads/Rvir, s2difft_1 / dt1)
plt.plot(rads/Rvir, s2difft_2 / dt2)
plt.xlabel(r'$r / r_\mathrm{vir}$')

plot(semilogx=True)
plt.plot(rads/Rvir, (s2difft_1 / dt1 - s2difft_2 / dt2) / (s2difft_1 / dt1)) # sub percent-level agreement
# agreement within 1.2% here using the colossus stuff...
# don't want so much error in the numerical derivatives here before we've even gotten to the actual computation


# In[ ]:


# now we need convergence in the mass direction, and then we can put it all together
# also can go back and see if astropy is really necessary
# don't really need astropy and the mass dM doesn't seem to have a problem either


# In[ ]:


# let's see how things work when we use the Zhao+09 concentrations instead


# In[253]:


# looking at just Zhao concentrations from file
zhao_dat = np.loadtxt('sample_mah_1e14',skiprows=1)

def sig2_tot(r, M, c, R):
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gamma(c) - 1.) / Gamma(c)) * 3. *eta0(c)**-1 * (phi0 - phir)
    return (1.0 / rho0_by_P0) * theta


# needs to take R_1 and R_2, c_1 and c_2
def dsig2_dt(r, dt, M, dM, dMdt, c_1, R_1, c_2, R_2):
    part_t = (sig2_tot(r, M, c_2, R_2) - sig2_tot(r, M, c_1, R_1)) / dt
    part_M = (sig2_tot(r, M+dM, c_1, R_1) - sig2_tot(r, M, c_1, R_1)) / dM
    print(part_t, part_M*dMdt)
    return part_t + part_M * dMdt

z1 = zhao_dat[1, 0]
z2 = zhao_dat[0, 0]
dt = (zhao_dat[0,-1] - zhao_dat[1,-1]) / yr_per_Gyr / 0.7 #roughly h

mass = zhao_dat[1,1]
R_1 = zhao_dat[1,4] * 1e3
R_2 = zhao_dat[0,4] * 1e3
#Rvir = mass_so.M_to_R(mass, z1, 'vir')
Rvir = R_1
nr = 30
rads = np.logspace(np.log10(0.01*Rvir),np.log10(Rvir), nr)

dM = zhao_dat[0,1] - mass

dMdt = dM/dt # it should be in Msun/h / Gyr

c_1 = concentration.concentration(mass, 'vir', z1, model = 'duffy08') #zhao_dat[1,2]
c_2 = concentration.concentration(mass+dM, 'vir', z2, model = 'duffy08') #zhao_dat[0,2]



print(c_1, np.log10(dM), np.log10(mass), dt, np.log10(dMdt))
print(z1)

print(Rvir, R_1, R_2)

ds2dt = dsig2_dt(rads, dt, mass, dM, dMdt, c_1, R_1, c_2, R_2)

plot(semilogx=True)
plt.plot(rads/Rvir, ds2dt)
plt.xlabel(r'$r / r_\mathrm{vir}$')
plt.ylabel(r'$\mathrm{d}\sigma_\mathrm{tot}^2 /\mathrm{d}t$ [(km/s)$^2$/Gyr]')

tgrowth = sig2_tot(rads, mass, c_1, R_1) / np.abs(ds2dt)
loglogplot() #plot(semilogx=True)
plt.plot(rads/Rvir, tgrowth)
plt.xlabel(r'$r / r_\mathrm{vir}$')
plt.ylabel(r'$t_\mathrm{growth}$ [Gyr]')
plt.ylim(10**1, 3*10**2)
# this result is now entirely within the realm of reasonable values, although it is a bit more flat than it could be

# conclusion is that the result is pretty sensitive to the concentrations, I think
# let's see how this changes if we use the duffy concentrations instead
# then, we can also try the 10^14 at z=0 to see if things still look correct

# this seems to reproduce the Fig 3 of Shi+2014 pretty well, but the t_growth grows a bit too slowly in the outer
# regions for the 10^14 Msun halo; however, it looks bang-on for the 10^15 Msun halo

# in conclusion, if we use this method, we can claim that we are very closely able to reproduce the Figs 1-4 of Shi+14

