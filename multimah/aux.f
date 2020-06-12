c**********************************************************************
c          SOME USEFUL AUXILIARY FUNCTIONS & SUBROUTINES
c
c   - subroutine that reads time
c   - subroutine for linear interpolation
c   - ARCTANH
c   - TANH
c   - XEXP
c   - lblnk
c
c  Frank van den Bosch                                    Dec. 1999
c**********************************************************************

      SUBROUTINE get_time()
c----------------------------------------------------------------------
c
c  This subroutine computes time since start of execution, and
c  since last call to this subroutine. It uses two intrinsic functions:
c    - etime returns time (in sec) elapsed since start of execution
c    - dtime returns time (in sec) elapsed since last call to dtime
c  The argument ttime(2) on return yields the user time, ttime(1),
c  and the system time, ttime(2). The times (e1 and e2) returned by
c  the functions themselves are the sums of user and system time.
c
c See the following html for information
c   http://docs.oracle.com/cd/E19957-01/805-4942/6j4m3r8t4/index.html
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL    ttime1(2),ttime2(2)

c---
  
      e_time = etime(ttime1)
      d_time = dtime(ttime2)

      et_h = INT(e_time/3600.0)
      et_m = INT(MOD(e_time,3600.0)/60.0)
      et_s = MOD(MOD(e_time,3600.0),60.0)

      dt_h = INT(d_time/3600.0)
      dt_m = INT(MOD(d_time,3600.0)/60.0)
      dt_s = MOD(MOD(d_time,3600.0),60.0)

      RETURN
      END      

c**********************************************************************

      SUBROUTINE write_time(ichoice)
c----------------------------------------------------------------------
c
c  IF (ichoice=1) write total elapsed time 
c  IF (ichoice=2) write elapsed time since last call
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER ichoice

c---
  
      CALL get_time

      IF (ichoice.EQ.1) THEN
        WRITE(*,73)et_h,et_m,et_s
      ELSE
        WRITE(*,74)dt_h,dt_m,dt_s
      END IF

c---

 73   FORMAT('Total elapsed time: ',I2,'h',I2,'m',F5.2,'s')
 74   FORMAT('Time elapsed since last call: ',I2,'h',I2,'m',F5.2,'s')

      RETURN
      END      

c**********************************************************************

      SUBROUTINE linintpol(xa,ya,N,x,y)

      IMPLICIT NONE

      INTEGER N,j
      REAL    xa(N),ya(N),x,y,x1,x2,y1,y2

c---

      CALL locate(xa,N,x,j)
      x1 = xa(j)
      x2 = xa(j+1)
      y1 = ya(j)
      y2 = ya(j+1)

      y = y1 + ((x-x1)/(x2-x1)) * (y2-y1)

      RETURN
      END      

c**********************************************************************

      REAL FUNCTION arctanh(x)
c--------------------------------------------------------------------
c
c Auxialiary function to compute arctanh(x)
c
c--------------------------------------------------------------------          
        
      REAL    x

c---

      arctanh = 0.5 * ALOG((1.0+x)/(1.0-x))

      END

c**********************************************************************

      REAL FUNCTION TANH(x)
c--------------------------------------------------------------------
c
c Auxialiary function to compute TANH(x)
c
c--------------------------------------------------------------------

      REAL     x

      REAL     XEXP
      EXTERNAL XEXP

c---

      TANH = (XEXP(x) - XEXP(-x)) / (XEXP(x) + XEXP(-x)) 

      END

c**********************************************************************

      REAL FUNCTION XEXP(x)
c--------------------------------------------------------------------
c
c Auxialiary function to compute EXP(x)
c
c--------------------------------------------------------------------

      REAL    x

c---

      IF (x.LT.-40.0) THEN
        XEXP = 0.0
      ELSE
        XEXP = EXP(x)
      END IF

      END

c**********************************************************************

      INTEGER FUNCTION lblnk(char)
c--------------------------------------------------------------------
c
c Function gives NUMBER of characters in a character variable `char'
c
c--------------------------------------------------------------------

      IMPLICIT NONE

      character char*(*)

c---

      lblnk=index(char,' ')-1

      RETURN
      END

c**********************************************************************

      SUBROUTINE Terminate(message)
c--------------------------------------------------------------------
c
c  Output error message and terminate program
c
c--------------------------------------------------------------------

      IMPLICIT NONE

      character  message*(*)

c---

      WRITE(*,'(A)')message

      STOP

      RETURN
      END

c**********************************************************************

      SUBROUTINE update_histograms(i,M,io)
c----------------------------------------------------------------------
c
c Update the histograms that store progenitor mass functions at a
c select number of redshifts
c
c----------------------------------------------------------------------
  
      INCLUDE 'paramfile.h'

      INTEGER   i,j,io
      REAL      M

c---

      IF (io.EQ.0) THEN

        j = INT(ABS(ALOG10(M/Mhalo)) / binsize) + 1

        IF (i.EQ.iz1) histo(j,1) = histo(j,1) + 1.0
        IF (i.EQ.iz2) histo(j,2) = histo(j,2) + 1.0
        IF (i.EQ.iz3) histo(j,3) = histo(j,3) + 1.0
        IF (i.EQ.iz4) histo(j,4) = histo(j,4) + 1.0
        IF (i.EQ.iz5) histo(j,5) = histo(j,5) + 1.0
        IF (i.EQ.iz6) histo(j,6) = histo(j,6) + 1.0

      END IF

      RETURN
      END

c**********************************************************************

      SUBROUTINE conf_levels(yy,n,s02,s16,s50,s84,s98)
c---------------------------------------------------------------------------
c
c  Given an array yy(1:n), sort and determine the 
c  5, 16, 50, 84 and 95 percent confidence levels.
c
c---------------------------------------------------------------------------

      INTEGER n,n02,n16,n50,n84,n98
      REAL    yy(n),xn,s02,s16,s50,s84,s98

c---

      CALL sort(n,yy)

      xn = FLOAT(n)       
      n02 = NINT(0.0228 * xn)
      n16 = NINT(0.1587 * xn)
      n50 = NINT(0.5 * xn)
      n84 = NINT(0.8413 * xn)
      n98 = NINT(0.9772 * xn)

      s02 = yy(n02)
      s16 = yy(n16)
      s50 = yy(n50)
      s84 = yy(n84)
      s98 = yy(n98)

      RETURN
      END

c**********************************************************************

      SUBROUTINE store_Jfunction
c----------------------------------------------------------------------
c
c  pre-compute the function J(ures)  [Eq. (A7) in Parkinson+08]
c  on a grid. In compute on logarithmic grid from 1.0E-6 to 1.0E+6.
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER i,j
      REAL    xx,urs,SS,SSerr

      REAL     Jfuncint
      EXTERNAL Jfuncint

c---

      DO i=1,NJfunc
        xx = -6.0 + FLOAT(i-1)/FLOAT(NJfunc-1) * 12.0
        urs = 10.0**xx
 
        CALL qags(Jfuncint,0.0,urs,1.0E-5,1.0E-5,SS,SSerr,Neval,
     &     ierr,Nlimit,Nlenw,last,iwork,work)

        IF (ierr.NE.0.AND.ierr.NE.2) THEN
          WRITE(*,*)' WARNING Jfunction: ',urs,SS,SSerr,ierr
        END IF

        xJ(i) = xx
        yJ(i) = ALOG10(SS)

      END DO

      RETURN
      END

c**********************************************************************

      REAL FUNCTION Jfuncint(u)

      INCLUDE 'paramfile.h'

      REAL   u

c---

      Jfuncint = (1.0 + (1.0/u**2))**(gamma1/2.0)

      END

c**********************************************************************

      REAL FUNCTION fEPS_SC(deltaS,deltaW)
c---------------------------------------------------------------------------
c
c  The EPS function  f(deltaS,deltaW)    for Spherical Collapse
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     deltaS,deltaW,Anorm

      REAL     XEXP
      EXTERNAL XEXP

c---

      Anorm = (1.0/sqrt2pi) * (deltaW/(deltaS**1.5)) 
      fEPS_SC = Anorm * XEXP(-((deltaW**2)/(2.0*deltaS)))

      END 

c**********************************************************************

      REAL FUNCTION fEPS_EC(deltaS,deltaW)
c---------------------------------------------------------------------------
c
c  The EPS function  f(deltaS,deltaW)   for Ellipsoidal Collapse
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     deltaS,deltaW
      REAL     Anorm,A0,A1,A2,nu0,Stilde,ddW,add

      REAL     XEXP
      EXTERNAL XEXP

c---

      Anorm = (1.0/sqrt2pi) * (deltaW/(deltaS**1.5)) 

      Stilde = deltaS/Shost
      nu0 = (Whost**2)/Shost
     
      A0 = 0.866 * (1.0 - 0.133*nu0**(-0.615))
      A1 = 0.308 * nu0**(-0.115) 
      A2 = 0.0373* nu0**(-0.115) 
        
      ddW = A0 * deltaW + A1 * SQRT(deltaS * Stilde)
      add = A2 * Stilde**1.5 * XEXP(-((A1**2)/2.0)*Stilde) *
     &        (1.0 + (A1/gamma32)*SQRT(Stilde)) 
      fEPS_EC = A0 * Anorm * (XEXP(-((ddW**2)/(2.0*deltaS))) + add)

      END 

c**********************************************************************






