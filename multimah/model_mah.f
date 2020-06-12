c**********************************************************************
c          Subroutines to compute average and median MAHs 
c**********************************************************************

      SUBROUTINE get_model_MAH(M0,z0)
c---------------------------------------------------------------------------
c
c  Compute the mass accretion history using the models of Zhao et al. (2009)
c  and Fakhouri et al. (2010)
c  Results are stored in a matrix:
c     model_mah(i,1) = MEDIAN mah according to Zhao et al.
c     model_mah(i,2) =  MEAN  mah according to Zhao et al.
c     model_mah(i,3) =  MEAN  mah according to Fakhouri et al.
c
c  WARNING. This only works if z0 = 0!!!!
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER  i,iz
      REAL     M0,z0,M,S,W,Wnext,z,Mfak,dt,dMdt
      REAL     acc,dlogS,xlogM,fff

      REAL     Wbuf0,wpar0,ppar0
      COMMON /zhaobuf/ Wbuf0,wpar0,ppar0

      REAL     variance,Delta_collapse,lookbacktime
      REAL     acc_rate_Fakhouri,acc_rate_Zhao,w_zhao,XEXP
      EXTERNAL variance,Delta_collapse,lookbacktime
      EXTERNAL acc_rate_Fakhouri,acc_rate_Zhao,w_zhao,XEXP

c---

      IF (z0.NE.zsample(1)) THEN
        WRITE(*,*)' WARNING: MAHs not valid for this redshift!!!'
        STOP
      END IF

c---

      M = M0
      S = variance(M)
      W = Delta_collapse(0.0)

      wpar0 = w_zhao(W,M)
      ppar0 = (wpar0/2.0) / ((1.0 + (wpar0/4.0))**6)
      Wbuf0 = W

      DO i=1,3
        model_mah(1,i) = 0.0
      END DO

      Mfak = M0
      DO iz = 2,Nzsample
        z = zsample(iz)
        dt = lookbacktime(z) - lookbacktime(zsample(iz-1))
        dMdt = acc_rate_Fakhouri(Mfak,z)
        Mfak = Mfak - (dMdt * dt * 1.0E+9)
        Wnext = Delta_collapse(z)
        acc = acc_rate_Zhao(W,M)
        dlogS = acc * ALOG10(Wnext/W)
        S = S * 10.0**dlogS
        IF (S.GT.vectorSS(Nsigma)) THEN
          model_mah(iz,1) = -20.0
          model_mah(iz,2) = -20.0
        ELSE
          CALL splint(vectorSS,vectorMM,vectorM2,Nsigma,S,xlogM)
          M = 10.0**xlogM
          fff = XEXP(((lnten * (0.12 - 0.15*ALOG10(M/M0)))**2)/2.0)
          model_mah(iz,1) = ALOG10(M/M0)
          model_mah(iz,2) = model_mah(iz,1) + ALOG10(fff)
        END IF
        model_mah(iz,3) = ALOG10(Mfak/M0)
        W = Wnext
      END DO

      RETURN
      END

c**********************************************************************

      REAL FUNCTION acc_rate_Fakhouri(xM,z)
c---------------------------------------------------------------------------
c
c The average mass accretion rate according to Fakhouri et al (2010)
c NOTE: the equation in Fakhouri+10 is for h=073, we need to convert
c       that to h=1 for our definition of halo mass.
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL      xhused
      PARAMETER (xhused = 0.73)

      REAL      xM,z,fac
      
c---

      fac = 46.1 * ((xM/xhused)/1.0E+12)**(1.1) * (1.0+1.11*z)
      fac = fac * SQRT(omega_0 * (1.0+z)**3 + omega_lambda)
      acc_rate_Fakhouri = fac * xhused

      END

c**********************************************************************

      REAL FUNCTION acc_rate_Zhao(Wbuf,Mbuf)
c---------------------------------------------------------------------------
c
c  The median mass accretion rate of Zhao et al. (2009).
c  NOTE: we set p=0, which implies the modification of Yang et al. (2011), 
c  as defined by their eq. (22). This implies that the MAH accounts for 
c  ALL subhaloes accreted onto the main progenitor, independent
c  of whether they are WITHIN the virial radius at z=0
c
c---------------------------------------------------------------------------

      IMPLICIT NONE

      REAL     Wbuf,Mbuf,pc,xp,xw

      REAL     Wbuf0,wpar0,ppar0
      COMMON /zhaobuf/ Wbuf0,wpar0,ppar0

      REAL     w_zhao
      EXTERNAL w_zhao

c---


c      pc = (ALOG10(Wbuf) - ALOG10(Wbuf0)) / (0.272/wpar0)
c      xp = ppar0 * MAX(0.,1.0-pc)

      xp = 0.0
 
      xw = w_zhao(Wbuf,Mbuf)

      acc_rate_Zhao = (xw-xp)/5.85

      END

c**********************************************************************

      REAL FUNCTION w_zhao(Wbuf,Mbuf)

      IMPLICIT NONE

      REAL     Wbuf,Mbuf,Sbuf

      REAL     variance,dlnSdlnM
      EXTERNAL variance,dlnSdlnM

c---

      Sbuf = variance(Mbuf)      
      w_zhao = (Wbuf/Sbuf) * 10.**(-(dlnSdlnM(Mbuf)/2.0))

      END

c**********************************************************************


