c**********************************************************************
c                    COSMOLOGY SUBROUTINES
c**********************************************************************
c
c  xH(z)           - compute hubble constant at redshift z (in km/s/Mpc)
c  omega(z)        - compute universal matter density at reshift z
c  dVdwdz(z)       - differentiable volume element at redshift z
c  r_comoving(z)   - comoving distance at redshift z
c  Delta_c(z)      - compute the critical density (i.e. 1.686 for SCDM)
c  Delta_crit(z)   - compute the virial mass fraction (i.e. 178 for SCDM)
c  time(z)         - time at redshift z, in terms of 1/H_0 with time(infty)=0
c  lookbacktime(z) - lookbacktime at redshift z in Gyrs with t(0) = 0
c  growth_rate(z)  - the linear growth factor at redshift z
c  tranfer_WDM(k)  - transfer function for WDM (wrt CDM)
c  power_spec(k)   - the CDM power spectrum (with arbitrary normalization)
c  variance(M)     - the variance of the CDM power spectrum on scale of mass M 
c
c  So far the following models are supported:
c         Omega_0 + Omega_Lambda = 1.0
c         Omega_0 < 1 and Omega_Lambda = 0.0
c
c  Frank van den Bosch                             Dec. 1999
c**********************************************************************

      REAL FUNCTION xH(z)
c----------------------------------------------------------------------
c
c Calculate hubble constant (in physical units; km/s/Mpc) at redshift z.
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL   z,z1,fac

c---

      z1 = 1.0 + z
    
      fac = omega_lambda + (1.0 - omega_lambda - omega_0) * z1**2 + 
     &      omega_0 * z1**3

      xH = xH_0 * SQRT(fac)

      END

c**********************************************************************

      REAL FUNCTION omega(z)
c----------------------------------------------------------------------
c
c Calculate the density parameter omega at redshift z. Note that omega 
c is only the mater contribution.
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z,xH
  
      EXTERNAL xH

c---

      omega = omega_0 * (1.0 + z)**3 / (xH(z)/xH_0)**2.0

      END

c**********************************************************************

      REAL FUNCTION dVdwdz(z)
c----------------------------------------------------------------------
c
c Calculate differentiable, comoving volume element at redshift z.
c
c UNITS: (Mpc/h)^3
c  NOTE: Only valid for flat cosmology!!!!!!
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL  z,r

      REAL     r_comoving,xH
      EXTERNAL r_comoving,xH

c---

      r = r_comoving(z)
      dVdwdz = 3000.0 * (r**2 / (xH(z)/xH_0))

      END 

c**********************************************************************

      REAL FUNCTION r_comoving(z)
c----------------------------------------------------------------------
c
c Comoving distance in Mpc/h at redshift z.
c
c----------------------------------------------------------------------

      IMPLICIT NONE

      REAL     z,ztemp,SS1

      REAL     toint8,midpnt
      EXTERNAL toint8,midpnt

c---

      ztemp = z
      CALL QROMO(toint8,0.0,ztemp,SS1,midpnt)
      r_comoving = 3000.0 * SS1

      END

c**********************************************************************

      REAL FUNCTION Delta_collapse(z)
c----------------------------------------------------------------------
c
c The overdensity for collapse at redshift z, called W in the
c EPS formalism of Lacey & Cole (1993). 
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z

      REAL     Delta_c,growth_rate
      EXTERNAL Delta_c,growth_rate

c---

      Delta_collapse = Delta_c(z) / growth_rate(z)
c      Delta_collapse = 1.686 / growth_rate(z)

      END

c**********************************************************************

      REAL FUNCTION Delta_c(z)
c----------------------------------------------------------------------
c
c Calculate the critical overdensity.
c We use the approximation in NFW97
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z,dc0,omz

      REAL     omega
      EXTERNAL omega

c---

      omz = omega(z)
      dc0 = 0.0
 
      IF (ABS(1.0-omega_0-omega_lambda).LT.1.0E-4) THEN
        dc0 = 0.15 * (12.0 * pi)**(2.0/3.0) * omz**(0.0055)
      END IF

      IF (omega_0.LT.1.0.AND.omega_lambda.EQ.0.0) THEN
        dc0 = 0.15 * (12.0 * pi)**(2.0/3.0) * omz**(0.0185)
      END IF
    
      IF (dc0.EQ.0.0) THEN
        WRITE(*,*)' Delta_c not defined for this cosmology'
        STOP
      ELSE
        Delta_c = dc0
      END IF

      END

c**********************************************************************

      REAL FUNCTION Delta_crit(z)
c----------------------------------------------------------------------
c
c Calculate the virial density in terms of critical density of the 
c Universe. We use the fitting formulae of Bryan & Norman, 1998, ApJ,
c 495, 80. These are accurate to better than 1% for 0.1 < omega_0 < 1.0
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL   z,x,omega
 
      EXTERNAL omega
c---

      x = omega(z) - 1.0

      IF (ABS(1.0-omega_0-omega_lambda).LT.1.0E-4) THEN
        Delta_crit = 18.0 * pi**2 + 82.0*x - 39.0 * x**2
        GOTO 3
      END IF
      
      IF (omega_0.LT.1.0.AND.omega_lambda.EQ.0.0) THEN
        Delta_crit = 18.0 * pi**2 + 60.0*x - 32.0 * x**2
        GOTO 3
      END IF

      WRITE(*,*)' Delta_crit not defined for this cosmology'
      STOP

 3    CONTINUE

      END

c**********************************************************************

      REAL FUNCTION time(z)
c----------------------------------------------------------------------
c
c Calculate time at redshift z in units of (1/H0)
c
c  This is a slightly modified version from a subroutine that appears 
c  in `subroutines.f' from the code of Navarro to calculate the 
c  concentration parameter c for NFW-halos.
c
c I have checked this procedure for SCDM, OCDM, and LCDM: works fine.
c
c Frank C. van den Bosch                                Feb. 1999
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL    z,tz,z1,a,atmp,omega1,const0,param0
      REAL    xx,xxz,xxmax,sstep

c---

c---Omega_0 = 1

      IF (omega_0.EQ.1.0) THEN
        tz=(2.0/3.0)*(1.0+z)**(-1.5)     
        GOTO 2
      END IF

c---Omega_0 <1, Omega_lambda = 0

c---this does not work using gfortran because the DO loop has to be
c   over integer values. Rather than rewriting this stupid integration
c   routine, which is pathetic programming anyways, I decided to simply
c   not allow for this cosmology

      IF (omega_0.LT.1.0.AND.omega_lambda.EQ.0) THEN
        WRITE(*,*)' Time does not support this cosmology'
        STOP 
c        sstep=1.0E-4
c        z1=1.0+z
c        a=1.0/z1
c        xxmax=1.0E5
c        DO xx=sstep,xxmax,sstep
c         atmp=0.5*omega_0*(COSH(xx)-1.0)/(1.0-omega_0)
c         IF (atmp.GT.a) THEN
c           xxz=xx
c           GOTO 1
c         END IF
c        END DO
c 1      CONTINUE
c        IF (atmp.LT.a) STOP 'in time :t_o: Increase xxmax..'
c        tz=0.5*omega_0*(sinh(xxz)-xxz)/(1.0-omega_0)**1.5
c        GOTO 2
      END IF

c---Omega_0 + Omega_lambda=1

      IF (ABS(1.0-omega_0-omega_lambda).LT.1.0E-4) THEN
        z1 = 1.0 + z
        param0 = cstlbt1 * z1**(-1.5)
        tz=cstlbt2 * ALOG(param0 + SQRT(1.0+param0*param0))
        GOTO 2
      END IF

      WRITE(*,*)' Time not defined for this cosmology'
      STOP

 2    CONTINUE
      time = tz

      END

c**********************************************************************

      REAL FUNCTION lookbacktime(z)
c--------------------------------------------------------------------
c
c Computes lookbacktime in Gyrs at redshift z.
c   Here lookbacktime is defined such that
c       lookbacktime = 0  at z=0
c       lookbacktime = t0 at z=infty
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z

      REAL     time
      EXTERNAL time

c---

      IF (z.EQ.0.0) THEN
        lookbacktime = 0.0
      ELSE
        lookbacktime = t0 - time(z) * xH_0_recip
      END IF

      END

c**********************************************************************

      REAL FUNCTION find_redshift(z)
c--------------------------------------------------------------------
c
c The root of this function gives the redshift associated with 
c time tbuf (with tbuf in Gyr).
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z

      REAL     tbuf
      COMMON /timepar/ tbuf

      REAL     time
      EXTERNAL time

c---
    
      find_redshift = (time(z) * xH_0_recip) - tbuf

      END

c**********************************************************************

      REAL FUNCTION z_at_t(z_try)
c--------------------------------------------------------------------
c
c  The root of this function gives the redshift at which the
c  lookback time is `time'
c
c--------------------------------------------------------------------
        
      INCLUDE 'paramfile.h'

      REAL    z_try

      REAL    tatz
      COMMON/zatt/ tatz

      REAL      lookbacktime
      EXTERNAL  lookbacktime

c---

      z_at_t = tatz - lookbacktime(z_try)

      END

c**********************************************************************

      REAL FUNCTION growth_rate(z)
c--------------------------------------------------------------------
c
c The linear growth factor at redshift z (see NFW97)
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z,ddtemp
      REAL*8   ww,w_0,y_0,y
  
      REAL*8   f1,f2,f3
      EXTERNAL f1,f2,f3

c---

      IF (omega_0.EQ.1.0.AND.omega_lambda.EQ.0.0) THEN
        ddtemp = 1.0/(1.0 + z)
        GOTO 3
      END IF

      IF (omega_0.LT.1.0.AND.omega_lambda.EQ.0.0) THEN
        w_0 = 1.0d0/DBLE(omega_0) - 1.0d0
        ww = w_0 / (1.0d0 + DBLE(z))
        ddtemp = SNGL(f1(ww)/f1(w_0))
        GOTO 3
      END IF

      IF (ABS(1.0-omega_0-omega_lambda).LT.1.0E-4) THEN
        w_0 = 1.0d0/DBLE(omega_0) - 1.0d0
        y_0 = (2.0d0 * w_0)**(1.0d0/3.0d0)
        y = y_0/(1.0d0 + DBLE(z))  
        ddtemp = SNGL((f2(y) * f3(y))/(f2(y_0) * f3(y_0)))
        GOTO 3
      END IF

 3    CONTINUE
      growth_rate = ddtemp

      END

c*********************************************************************

      REAL FUNCTION transferWDM(xk)
c--------------------------------------------------------------------
c
c The WDM transfer function T_WDM(k) = SQRT(P_WDM(k)/P_CDM(k))
c See Sommer-Larsen & Dolgov, 2001, ApJ, 551, 608
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL    xk,Rf,fac

      REAL     XEXP
      EXTERNAL XEXP

c---

      Rf = 0.065 * (omega_0-omega_b)**(-1.0/3.0) * 
     &     (Mf_WDM/1.0E+11)**(1.0/3.0)

      fac = (xk*Rf) + (xk*Rf)**2.0

      transferWDM = XEXP(-fac/2.0)

      END

c*********************************************************************

      REAL FUNCTION power_spec(xk)
c--------------------------------------------------------------------
c
c The CDM power spectrum with arbitrary normalization.
c The transfer function is taken from BBKS (Bardeen et el 1986) or
c from Efstathiou, Bond & White (1992) or from Eisenstein & Hu.
c The initial power-spectrum directly after inflation is a power-law
c with index `nspec'. For nspec=1 this yields the standard 
c Harrison-Zel'dovich spectrum. The normalization is set by xk0
c which defines the wavelength at which the amplitude of the initial
c fluctuation spectrum is unity (this is arbitrary). The actual
c normalisation of the power-spectrum is set by sigma8.
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL    e
      PARAMETER (e=2.7182818)

      REAL    xk,k,q,t1,t2,tk,xk0,T_WDM
      REAL    silk,stilde,fff,C1,C2,t11,t12,t3,tb1,tb2,T_c,T_b

      REAL     XEXP,transferWDM
      EXTERNAL XEXP,transferWDM

c---

c---set normalization of initial power spectrum

      xk0 = 1.0/3000.0

c---initialize tk to check that computation was succesful

      tk = -1.0

c---the BBKS fitting function

      IF (BBKS) THEN
        q = xk/Gamma_cosmo
        t1 = ALOG(1.0+2.34*q) / (2.34*q)
        t2 = 1.0 + (3.89*q) + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
        tk = t1 * t2**(-0.25) 
      END IF

c---the EBW fitting function

      IF (EBW) THEN
        q = xk/Gamma_cosmo
        tk = 1.0 + ((6.4*q) + (3.0*q)**(3./2.) + (1.7*q)**2)**(1.13)
        tk = tk**(-1.0/1.13)
      END IF

c---the Eisenstein & Hu fitting function

      IF (EISHU) THEN
        k = xk * xhubble
        q = k/(13.41*keq)

        silk = (k/ksilk)**1.4
        silk = XEXP(-silk)
        stilde = sEH / ((1.0 + (bnode/(k*sEH))**3)**0.333)

        fff = 1.0 / (1.0 + (k*sEH/5.4)**4)
        C1 = 14.2 + 386.0/(1.0+69.9*q**(1.08))
        C2 = 14.2/alpha_c + 386.0/(1.0+69.9*q**(1.08))

        t11 = ALOG(e + 1.8 * beta_c * q)
        t12 = ALOG(e + 1.8 * q)

        t1 = t11 / (t11 + C1*q**2)
        t2 = t11 / (t11 + C2*q**2)
        t3 = t12 / (t12 + C1*q**2)

        tb1 = t3 / (1.0 + (k*sEH/5.2)**2)
        tb2 = (alpha_b/(1.0 + (beta_b/(k*sEH))**3)) * silk

c---for small xk*stilde I approximate sin(x)/x=1

        T_c = fff * t1 + (1.0-fff) * t2
        IF ((k*stilde).LT.1.0E-25) THEN
          T_b = (tb1 + tb2)
        ELSE
          T_b = (tb1 + tb2) * SIN(k*stilde)/(k*stilde)
        END IF

        tk = (omega_b/omega_0) * T_b + ((omega_0-omega_b)/omega_0) * T_c
      END IF

c---check that a transfer function has been defined

      IF (tk.LT.0.0) STOP

c---in the case of WDM, filter this.

      IF (Mf_WDM.EQ.0.0) THEN
        T_WDM = 1.0
      ELSE
        T_WDM = transferWDM(xk)
      END IF

      power_spec = tk**2 * (xk/xk0)**nspec * (T_WDM)**2.0

      END

c*********************************************************************

      REAL FUNCTION variance(M)
c--------------------------------------------------------------------
c
c This function yields the mass variance s(M) for a CDM power spectrum.
c We use the BBKS transfer function, and there is a choise of
c three different filters (Top-Hat, Gaussian, and Sharp-k).
c 
c [M] = h^{-1} Msun
c 
c NOTE: This rms mass variance is normalized by sigma8.
c       For that you need to initialize it by calling this
c       routine with sigma8_norm set to sigma8 and the mass
c       set to M8 = 5.9543E+14 * omega_0. Call the resulting
c       value sigma8_norm. Subsequent calls than yield
c       the `sigma8-normalized' values.
c 
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL  M,Mbuf,logM

      REAL     var_numerical,var_spline
      EXTERNAL var_numerical,var_spline

c---

      Mbuf = M
      logM = ALOG10(M)

      IF (ivar.EQ.1) THEN
 
        variance = var_numerical(Mbuf)

      ELSE

        IF (logM.LT.Mminvar.OR.logM.GT.Mmaxvar) THEN
          variance = var_numerical(Mbuf)
        ELSE
          variance = var_spline(Mbuf)
        END IF

      END IF

      END

c*********************************************************************

      REAL FUNCTION var_numerical(M)
c--------------------------------------------------------------------
c
c Mass variance s(M), computed numerically by using the appropriate
c integral equation. Based on Top-Hat Filter
c 
c [M] = h^{-1} Msun
c 
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER ierr1,ierr2,ierr3
      REAL    M,SS1,SS2,SS3,SS1err,SS2err,SS3err

      REAL  Rf
      COMMON /filtering/ Rf

      REAL     toint1
      EXTERNAL toint1
 
c---

      Rf = ((3.0*M)/(4.0*pi*rho_aver_com))**(1.0/3.0)

      IF (Rf.LT.2.0) THEN
        CALL qags(toint1,0.0,0.5,1.0E-5,1.0E-5,SS1,SS1err,Neval,
     &            ierr1,Nlimit,Nlenw,last,iwork,work)
        CALL qags(toint1,0.5,1.0/Rf,1.0E-5,1.0E-5,SS2,SS2err,Neval,
     &            ierr2,Nlimit,Nlenw,last,iwork,work)
        CALL qagi(toint1,1.0/Rf,1,1.0E-5,1.0E-5,SS3,SS3err,Neval,
     &            ierr3,Nlimit,Nlenw,last,iwork,work)
      ELSE
        CALL qags(toint1,0.0,1.0/Rf,1.0E-5,1.0E-5,SS1,SS1err,Neval,
     &            ierr1,Nlimit,Nlenw,last,iwork,work)
        CALL qags(toint1,1.0/Rf,0.5,1.0E-5,1.0E-5,SS2,SS2err,Neval,
     &            ierr2,Nlimit,Nlenw,last,iwork,work)
        CALL qagi(toint1,0.5,1,1.0E-5,1.0E-5,SS3,SS3err,Neval,
     &            ierr3,Nlimit,Nlenw,last,iwork,work)
      END IF

      IF (ierr1.NE.0.AND.ierr1.NE.2) THEN
        WRITE(*,*)' WARNING variance 1: ',ALOG10(M),SS1,ALOG10(SS1err),
     &                                    ierr1
      END IF
      IF (ierr2.NE.0.AND.ierr2.NE.2) THEN
        WRITE(*,*)' WARNING variance 2: ',ALOG10(M),SS2,ALOG10(SS2err),
     &                                    ierr2
      END IF
      IF (ierr1.NE.0.AND.ierr1.NE.2) THEN
        WRITE(*,*)' WARNING variance 3: ',ALOG10(M),SS3,ALOG10(SS3err),
     &                                    ierr3
      END IF

      var_numerical = (sigma8/sigma8_norm) * 
     &                   SQRT((SS1 + SS2 + SS3)/(2.0*pi*pi))

      END

c*********************************************************************

      REAL FUNCTION var_spline(M)
c--------------------------------------------------------------------
c
c Mass variance s(M), computed using spline interpolation
c 
c [M] = h^{-1} Msun
c 
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL  M,logM,SS1
      
c---

      logM = ALOG10(M)
      CALL splint(vectorM,vectorS,vectorS2,Nsigma,logM,SS1)
      var_spline = SS1

      END

c**********************************************************************

      REAL FUNCTION dlnSdlnM(xM)
c----------------------------------------------------------------------
c
c  The logarithmic derivative dlnS/dlnM(M) where S(M) = variance(M)**2
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL      eps
      PARAMETER (eps=0.15)

      REAL     xM,M1,M2,sig1,sig2

      REAL     variance
      EXTERNAL variance

c---

      M1 = (1.0-eps) * xM
      M2 = (1.0+eps) * xM

      sig1 = (variance(M1))**2
      sig2 = (variance(M2))**2

      dlnSdlnM = (ALOG(sig2)-ALOG(sig1)) / (ALOG(M2)-ALOG(M1))

      END

c*********************************************************************

      REAL FUNCTION mf_EPS(M)
c--------------------------------------------------------------------
c
c   Halo mass function of EPS (standard).
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER  iz
      REAL     M,nu,dodl

      REAL     variance,Delta_c,dlnSdlnM,XEXP
      EXTERNAL variance,Delta_c,dlnSdlnM,XEXP

c---

      nu = Delta_c(0.0)/variance(M)
      dodl = (1.0/sqrt2pi) * nu * XEXP(-0.5 * nu**2)
      mf_EPS = rho_aver_com/(M**2) * dodl * ABS(dlnSdlnM(M))

      END

c**********************************************************************

      REAL*8 FUNCTION f1(x)
c--------------------------------------------------------------------
c
c Auxialiary function used in computation of growth rate
c
c--------------------------------------------------------------------

      IMPLICIT NONE

      REAL*8    x,fac1,fac2

c---

      fac1 = DSQRT(1.0d0+x) - DSQRT(x)
      fac2 = 3.0d0*DSQRT(1.0d0+x)/(x**1.5d0)

      f1 = 1.0d0 + (3.0d0/x) + fac2 * DLOG(fac1)

      END

c**********************************************************************

      REAL*8 FUNCTION f2(x)
c--------------------------------------------------------------------
c
c Auxialiary function used in computation of growth rate
c
c--------------------------------------------------------------------

      IMPLICIT NONE
       
      REAL*8    x

c---

      f2 = DSQRT(x**3.0d0 + 2.0d0)/(x**1.5d0)

      END

c**********************************************************************

      REAL*8 FUNCTION f3(x)
c--------------------------------------------------------------------
c
c Auxialiary function used in computation of growth rate
c
c--------------------------------------------------------------------

      IMPLICIT NONE

      REAL     ff3,SS
      REAL*8   x

      EXTERNAL ff3

c---

      CALL QROMB(ff3,0.0,SNGL(x),SS)

      f3 = DBLE(SS)

      END 

c**********************************************************************
   
      REAL FUNCTION ff3(x)
c--------------------------------------------------------------------
c
c Auxialiary function used in computation of growth rate
c
c--------------------------------------------------------------------

      IMPLICIT NONE

      REAL     x

c---

      ff3 = (x/(x**3.0 + 2.0))**1.5

      END 

c*********************************************************************

      REAL FUNCTION toint1(xk)
c--------------------------------------------------------------------
c
c Integrating this function yields the mass variance (TH filter only)
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     xk,wf,x

      REAL  Rf
      COMMON /filtering/ Rf

      REAL     power_spec,XEXP
      EXTERNAL power_spec,XEXP

c---

c---compute filter at wavelength xk

      x = xk * Rf
      wf = 3.0 * (SIN(x) - x * COS(x)) / x**3

c---and compute integrand

      toint1 = power_spec(xk) * wf**2 * xk**2

      END


c**********************************************************************

      REAL FUNCTION toint8(z)
c---------------------------------------------------------------------- 
c
c  The integral of this function from zero to z gives the comoving
c  distance for an object at redshift z.
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z

      REAL     xH
      EXTERNAL xH

c---

      toint8 = xH_0/xH(z)
      
      END

c*********************************************************************
