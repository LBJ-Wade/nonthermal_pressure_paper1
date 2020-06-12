c**********************************************************************
c                SUBROUTINES WITH INITIALIZATIONS
c
c
c  Frank van den Bosch                             March 2001
c**********************************************************************

      SUBROUTINE read_global_param
c---------------------------------------------------------------------------
c
c  Subroutine to read in global parameters
c
c---------------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      INTEGER      ipow

c---

c---read iseed for randum number generator

      WRITE(*,*)' Give iseed : '
      READ(*,*)iseed
      iseed = -ABS(iseed)
      WRITE(*,*)' iseed = ',iseed
      WRITE(*,*)' '

c---read parameters of cosmological parameters

      WRITE(*,*)' Give Omega_0 : '
      READ(*,*)omega_0
      WRITE(*,*)'Omega_m,0 = ',omega_0
      WRITE(*,*)' '

      WRITE(*,*)' Give Omega_Lambda : '
      READ(*,*)omega_lambda
      WRITE(*,*)'Omega_Lambda = ',omega_lambda
      WRITE(*,*)' '

      WRITE(*,*)' Give h (=H_0/100) : '
      READ(*,*)xhubble
      WRITE(*,*)'xhubble = ',xhubble
      WRITE(*,*)' '

      WRITE(*,*)' Give sigma_8 : '
      READ(*,*)sigma8
      WRITE(*,*)'sigma_8 = ',sigma8
      WRITE(*,*)' '

      WRITE(*,*)' Give nspec (Harrisson-Zeldovich = 1.0)'
      READ(*,*)nspec
      WRITE(*,*)'nspec = ',nspec
      WRITE(*,*)' '

      WRITE(*,*)' Give Omega_b h^2 : '
      READ(*,*)omega_b_h2
      WRITE(*,*)'Omega_b_h2 = ',omega_b_h2
      WRITE(*,*)' '
   
c---decide which power spectrum fitting function to use
c     BBKS = Bardeen, Bond, Kaiser & Szalay, 1986, ApJ, 304, 15
c     EBW  = Efstathiou, Bond & White, 1992, MNRAS, 258, 1
c     EH   = Eisenstein & Hu, 1998, ApJ, 496, 605

      WRITE(*,*)' T(K): BBKS (1), EBW (2), EH (3)?'
      READ(*,*)ipow
      WRITE(*,*)' choice = ',ipow
      WRITE(*,*)' '

      IF (ipow.NE.1.and.ipow.NE.2.and.ipow.NE.3) THEN
        CALL Terminate('Invalid ipow')
      END IF
 
      BBKS  = .FALSE.
      EBW   = .FALSE.
      EISHU = .FALSE.
      IF (ipow.EQ.1) BBKS  = .TRUE.
      IF (ipow.EQ.2) EBW   = .TRUE.
      IF (ipow.EQ.3) EISHU = .TRUE.

c---allow for the possibility of WDM by specifying the filter mass

      WRITE(*,*)' Give WDM filter mass (0.0 = CDM)'
      READ(*,*)Mf_WDM
      WRITE(*,*)'Mf_WDM = ',Mf_WDM
      WRITE(*,*)' '

c---indicate whether you want to compute the mass variance numerically
c   (slow) or using the fitting function (fast)

      WRITE(*,*)' Give method of computing S(M) (1=num, 2=fit)'
      READ(*,*)ivar
      WRITE(*,*)' method = ',ivar
      WRITE(*,*)' '

      IF (ivar.NE.1.AND.ivar.NE.2) THEN
        CALL Terminate('Invalid ivar')
      END IF

      RETURN
      END

c**********************************************************************

      SUBROUTINE read_tree_param
c---------------------------------------------------------------------------
c
c  Subroutine to read in parameters specific to the merger tree(s)
c
c---------------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      INTEGER  icollapse,ianswer
      REAL     xlgMhalo

c---

c---read present day halo mass

      WRITE(*,*)' Give present day mass of halo (h^{-1} Msun)'
      READ(*,*)xlgMhalo
      WRITE(*,*)' Halo mass = ',xlgMhalo
      WRITE(*,*)' '
      Mhalo = 10.0**xlgMhalo

c---give ratio of mass at z=0 and the resolution limit

      WRITE(*,*)' Give mass resolution, Mmin/M0 (0=compute) '
      READ(*,*)flimit
      WRITE(*,*)' Mass resolution Mmin/M0 = ',flimit
      WRITE(*,*)' '

c---read redshift that corresponds to `present'

      WRITE(*,*)' Give redshift'
      READ(*,*)znow
      WRITE(*,*)'znow = ',znow
      WRITE(*,*)' '

      WRITE(*,*)' Give maximum redshift for resampling'
      READ(*,*)zmaxsample
      WRITE(*,*)'zmax = ',zmaxsample
      WRITE(*,*)' '

      WRITE(*,*)' Give timestep in fraction of free-fall time'
      READ(*,*)frac_tff
      WRITE(*,*)'frac_tff = ',frac_tff
      WRITE(*,*)' '

c---read parameters for mass loss rate (as defined in vdB05)

      WRITE(*,*)' Give slope alpha '
      READ(*,*)slope
      WRITE(*,*)'slope = ',slope
      WRITE(*,*)' '

      WRITE(*,*)' Give tau '
      READ(*,*)tau
      WRITE(*,*)'tau = ',tau
      WRITE(*,*)' '

c---give mass ratio msub/macc at which subhalo `dissolves'

      WRITE(*,*)' Give mass ratio for disruption (Msub/Macc) '
      READ(*,*)fdissolve
      WRITE(*,*)' Disruption mass ratio Msun/Macc = ',fdissolve
      WRITE(*,*)' '

c---read PCH08 parameters
 
      WRITE(*,*)' Give PCH08 parameter G0 [fiducial = 0.57] '
      READ(*,*)G0
      WRITE(*,*)'    G0 = ',G0
      WRITE(*,*)' '

      WRITE(*,*)' Give PCH08 parameter gamma1 [fiducial = 0.38] '
      READ(*,*)gamma1
      WRITE(*,*)'gamma1 = ',gamma1
      WRITE(*,*)' '

      WRITE(*,*)' Give PCH08 parameter gamma2 [fiducial =-0.01] '
      READ(*,*)gamma2
      WRITE(*,*)'gamma2 = ',gamma2
      WRITE(*,*)' '

      WRITE(*,*)' Give dWmax [fiducial = 0.1] '
      READ(*,*)dWmax
      WRITE(*,*)'dWmax = ',dWmax
      WRITE(*,*)' '

c---do you want to vary Mhalo or keep it fixed?

      WRITE(*,*)' Keep Mhalo fixed? (0=NO, 1=YES) '
      READ(*,*)ianswer
      WRITE(*,*)' option = ',ianswer
      WRITE(*,*)' '

      IF (ianswer.NE.0.AND.ianswer.NE.1) THEN
        CALL Terminate('Invalid Mhalo-answer')
      END IF
 
      IF (ianswer.EQ.0) THEN
        varyingMhalo=.TRUE.
      ELSE
        varyingMhalo=.FALSE.
      END IF

c---do you want to vary Mmin or keep it fixed?

      WRITE(*,*)' Keep Mmin fixed? (0=NO, 1=YES) '
      READ(*,*)ianswer
      WRITE(*,*)' option = ',ianswer
      WRITE(*,*)' '

      IF (ianswer.NE.0.AND.ianswer.NE.1) THEN
        CALL Terminate('Invalid Mmin-answer')
      END IF
 
      IF (ianswer.EQ.0) THEN
        varyingMmin =.TRUE.
      ELSE
        varyingMmin =.FALSE.
      END IF
      
c---do you want to evolve the subhaloes or not?
 
      WRITE(*,*)' Evolve subhaloes? (0=NO, 1=YES) '
      READ(*,*)ianswer
      WRITE(*,*)' option = ',ianswer
      WRITE(*,*)' '

      IF (ianswer.NE.0.AND.ianswer.NE.1) THEN
        CALL Terminate('Invalid evolve-option')
      END IF
 
      IF (ianswer.EQ.0) THEN
        evolve =.FALSE.
      ELSE
        evolve =.TRUE.
      END IF

c---do you want to compute Vmax for all haloes?
 
      WRITE(*,*)' Compute Vmax? (0=NO, 1=YES) '
      READ(*,*)ianswer
      WRITE(*,*)' option = ',ianswer
      WRITE(*,*)' '

      IF (ianswer.NE.0.AND.ianswer.NE.1) THEN
        CALL Terminate('Invalid vmax-option')
      END IF
 
      IF (ianswer.EQ.0) THEN
        compVmax=.FALSE.
      ELSE
        compVmax=.TRUE.
      END IF

c---what mode do you want to do?
c     1 = Compute Statistics
c     2 = Write Merger Trees
c     3 = Do Both

      WRITE(*,*)' Give mode (1=stat, 2=write, 3=both)'
      READ(*,*)imode
      WRITE(*,*)'imode = ',imode
      WRITE(*,*)' '

      IF (imode.NE.1.AND.imode.NE.2.AND.imode.NE.3) THEN
        CALL Terminate('Invalid mode')
      END IF
      
c---give directory to which output should be written

      WRITE(*,*)' Give model directory '
      READ(*,'(A)')moddir
      WRITE(*,'(A)')moddir
      WRITE(*,*)' '
          
      RETURN
      END

c**********************************************************************

      SUBROUTINE init_cosmo    
c---------------------------------------------------------------------------
c
c  Subroutine to initialize cosmology related  stuff
c
c---------------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

c---parameter that sets photon temperature (needed for EH transfer function)

      REAL   theta
      PARAMETER (theta = 1.0093)

      REAL     sTH,sGS,sSK,m_WDM,rfilter
      REAL     f,bb1,bb2,zeq,zd,y,Fy,Gy,Req,Rd,a1,a2,b1,b2

      REAL     time,Delta_crit,XEXP
      EXTERNAL time,Delta_crit,XEXP
     
c---

c---Calculate H0 in km/s/Mpc and its reciprocal in Gyr. From the latter
c   we then calculate the age of the Universe, t0, in Gyr

      xH_0 = 100.0 * xhubble
      xH_0_recip = 1.0 / (xH_0 * 1.023E-3)            
  
c---calculate free-fall time at z=0 for an overdensity of 200
 
      tff0 = (pi/SQRT(800.0 * omega_0)) * xH_0_recip

c---calculate parameters used to speed up calculation of (lookback)time

      cstlbt1 = SQRT((1.0-omega_0)/omega_0)
      cstlbt2 = 2.0 / (3.0*SQRT(1.0-omega_0))

      t0 = time(0.0) * xH_0_recip

c---set critical density at z=0 and comoving mass density [h^2 Msun Mpc^{-3}]

      rho_crit_0 = 3.0E+4 / (8.0 * pi * gee)
      rho_aver_com = rho_crit_0 * omega_0

c---calculate critical density for collapse at z=0

      deltacrit0 = Delta_crit(0.0)

c---calculate baryon density and baryonic mass fraction

      omega_b = omega_b_h2/(xhubble**2.0)
      f_bar = omega_b/omega_0

c---define Gamma: two options here; i) go with the standard definition
c   (Gamma = Omega_0 h), which is for negligible baryon mass, or use
c   the baryonic correction from Sugiyama (1995).

c---with baryonic correction

c      Gamma_cosmo = omega_0 * xhubble * 
c     &              XEXP(-omega_b - SQRT(2.0*xhubble) * f_bar)

c---without baryonic correction

      Gamma_cosmo = omega_0 * xhubble

c---if we are considering WDM, compute the mass of WDM particles

      IF (Mf_WDM.NE.0.0) THEN
        m_WDM = 2.4 * xhubble**(1.25) * SQRT(omega_0-omega_b) *
     &          (Mf_WDM/1.0E+11)**(-0.25)
        rfilter = 0.065 * (omega_0-omega_b)**(-1.0/3.0) * 
     &          (Mf_WDM/1.0E+11)**(1.0/3.0)
        WRITE(*,*)'  Mf_WDM = ',Mf_WDM
        WRITE(*,*)'   m_WDM = ',m_WDM,' keV'
        WRITE(*,*)'     R_f = ',rfilter,' Mpc/h'
        WRITE(*,*)' '
      END IF

c---define a number of parameters needed to compute the Eisenstein & Hu
c   power specrum

      f = omega_0 * xhubble**2

      bb1 = 0.313 * f**(-0.419) * (1.0 + 0.607*f**0.674)
      bb2 = 0.238 * f**(0.223)
 
      bnode = 8.41*f**0.435

      keq = 7.46E-2 * f / theta**2.0
      ksilk = 1.6 * (omega_b_h2)**0.52 * f**0.73 * 
     &       (1.0 + (10.4*f)**(-0.95))

      zeq = 2.5E+4 * f / theta**4.0
      zd = 1291.0 * ((f**0.251)/(1.0 + 0.659*f**0.828)) *
     &               (1.0 + bb1 * omega_b_h2**bb2)

      y = ((1.0+zeq)/(1.0+zd))
      Fy = ALOG((SQRT(1.0+y) + 1.0) / (SQRT(1.0+y) - 1.0))
      Gy = y * (-6.0 * SQRT(1.0+y) + (2.0+3.0*y) * Fy)

      Req = 31.5 * omega_b_h2 * (1000.0/zeq) / theta**4.0
      Rd  = 31.5 * omega_b_h2 * (1000.0/zd) / theta**4.0

      sEH = (2.0/(3.0*keq)) * SQRT(6.0/Req) *
     &      ALOG((SQRT(1.0+Rd) + SQRT(Rd+Req))/(1.0+SQRT(Req)))
        
      a1 = (46.9*f)**(0.670) * (1.0 + (32.1*f)**(-0.532))
      a2 = (12.0*f)**(0.424) * (1.0 + (45.0*f)**(-0.582))
      b1 = 0.944 / (1.0+(458.0*f)**(-0.708))
      b2 = (0.395*f)**(-0.0266)

      alpha_c = a1**(-f_bar) * a2**(-(f_bar**3))
      beta_c = 1.0 + b1*(((omega_0-omega_b)/omega_0)**b2 - 1.0)
      beta_c = 1.0/beta_c

      alpha_b = 2.07 * keq * sEH * (1.0+Rd)**(-0.75) * Gy
      beta_b = 0.5 + f_bar + 
     &   (3.0-2.0*f_bar) * SQRT((17.2*f)**2 + 1.0)

      RETURN
      END

c**********************************************************************

      SUBROUTINE init_variance
c---------------------------------------------------------------------------
c
c  Subroutine to initialize mass variance
c
c---------------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      INTEGER  i,ivtemp
      REAL     xM,z,yp1,ypn,xp1,xpn

      REAL     variance,Delta_collapse
      EXTERNAL variance,Delta_collapse

c---

c---compute un-normalized rms mass variance inside spherical
c   shell with radius Rf = 8 h^{-1} Msun
c   This is used to normalize the power-spectrum to sigma8.

      WRITE(*,*)'           >>> Computing Mass Variance <<<'
      WRITE(*,*)' '

      ivtemp = ivar
      ivar = 1

      sigma8_norm = sigma8
      sigma8_norm = variance(5.9543E+14 * omega_0)
      c8 = 1.0

      ivar = ivtemp

c---from now, `variance' is normalized to sigma8
c---set up the mass variance on a grid. The grid is a one-D vector,
c   for which we compute the mass variance numerically. The grid
c   consistes of Nsigma points with 5 <= log(M) <= 18.0

      ivtemp = ivar
      ivar = 1

      DO i=1,Nsigma
        vectorM(i) = Mminvar + 
     &       FLOAT(i-1)/FLOAT(Nsigma-1) * (Mmaxvar-Mminvar)
        vectorZ(i) = FLOAT(i-1)/FLOAT(Nsigma-1) * 100.0
        xM = 10.0**vectorM(i)
        z = vectorZ(i)
        vectorS(i) = variance(xM)
        vectorD(i) = Delta_collapse(z)
      END DO      

c---compute the derivatives at the two ends of one-D grids

      yp1 = (vectorS(2) - vectorS(1)) / 
     &      (vectorM(2) - vectorM(1))
      ypn = (vectorS(Nsigma) - vectorS(Nsigma-1)) / 
     &      (vectorM(Nsigma) - vectorM(Nsigma-1))

      xp1 = (vectorZ(2) - vectorZ(1)) /
     &      (vectorD(2) - vectorD(1))
      xpn = (vectorZ(Nsigma) - vectorZ(Nsigma-1)) / 
     &      (vectorD(Nsigma) - vectorD(Nsigma-1))

c---and compute the spline coefficients, to be used for spline interpolation
c   note that we compute the spline coefficients both ways!

      DO i=1,Nsigma
        vectorSS(i) = vectorS(Nsigma+1-i)
        vectorMM(i) = vectorM(Nsigma+1-i)
      END DO

      CALL spline(vectorM,vectorS,Nsigma,yp1,ypn,vectorS2)
      CALL spline(vectorSS,vectorMM,Nsigma,2.0E+30,2.0E+30,vectorM2)
      CALL spline(vectorD,vectorZ,Nsigma,xp1,xpn,vectorZ2)
      ivar = ivtemp

      RETURN
      END

c**********************************************************************

      SUBROUTINE init_param
c---------------------------------------------------------------------------
c
c  Subroutine to initialize parameters and variables
c
c---------------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      INTEGER  i,j,k,ihalo
      REAL     xlgz

c---

c---initialize histograms

      DO i=1,Nhismax
        DO k=1,Ntree
          histo_cum(k,i,0,1) = 0.0
          histo_cum(k,i,0,2) = 0.0
          histo_cum(k,i,0,3) = 0.0
          histo_cum(k,i,0,4) = 0.0
          histo_cum(k,i,0,5) = 0.0
          histo_cum(k,i,0,6) = 0.0
        END DO
        DO j=1,5
          histo(i,j) = 0.0
          zacc_sub(i,j,1) = 0.0
          zacc_sub(i,j,2) = 0.0
          histo_sub(i,j,1) = 0.0
          histo_sub(i,j,2) = 0.0
          histo_sub(i,j,3) = 0.0
          histo_sub(i,j,4) = 0.0
          histo_sub(i,j,5) = 0.0
          histo_sub(i,j,6) = 0.0
          histo_sub(i,j,7) = 0.0
          DO k=1,Ntree
            histo_cum(k,i,j,1) = 0.0
            histo_cum(k,i,j,2) = 0.0
            histo_cum(k,i,j,3) = 0.0
            histo_cum(k,i,j,4) = 0.0
            histo_cum(k,i,j,5) = 0.0
            histo_cum(k,i,j,6) = 0.0
          END DO
        END DO
      END DO

c---initialize tree properties

      DO ihalo=1,Ntree
        DO i=1,4
          DO j=0,2,1          
            itree_prop(ihalo,i,j)= 0
            tree_prop(ihalo,i,j) = 0.0
          END DO
        END DO
      END DO

c---initialize mahs, to be computed on grid linear in log(1+z)

      DO j=1,Nzmax
        DO i=1,Ntree
          mah(i,j,1) = -20.0
          mah(i,j,2) = -20.0
        END DO
       END DO	  

c---initialize halo concentrations of host haloes

      DO i=1,Ntree
        conc(i) = 0.0  
        zform(i,1) = 0.0
        zform(i,2) = 0.0
        zform(i,3) = 0.0
      END DO
      
c---store J(mu) function (used for PCH+08 algorithm) in array

      CALL store_Jfunction

      RETURN
      END

c**********************************************************************

      SUBROUTINE flush_tree()
c---------------------------------------------------------------------------
c
c  initialize the tree structure
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER i,j

c---

      MaxResolution = 0.0

      DO i=0,Nmax,1
        Mprog(i) = 0.0
        Wprog(i) = 0.0
        zprog(i) = 0.0
        itree(i,1) = 0
        itree(i,2) = 0
        itree(i,3) = 0
      END DO

      DO i=1,Nmaxtraj
        itraj(i,1) = 0
        itraj(i,2) = 0
        itraj(i,3) = 0
      END DO

      RETURN
      END

c**********************************************************************

