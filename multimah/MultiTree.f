      PROGRAM MultiTree
c**********************************************************************
c  + Version 1.5: Feb 28, 2014
c----------------------------------------------------------------------
c  Program to construct halo merger trees using binary method
c  of Parkinson, Cole & Helly (2008, MNRAS, 383, 557)
c
c  A detailed description of how to compile and run the program
c  is provided in the README file in this directory.
c  The document Versions.doc describes the modifications made
c  in the different version.
c----------------------------------------------------------------------
c  Author: Frank van den Bosch                        Yale University  
c**********************************************************************

      INCLUDE 'paramfile.h'

      INTEGER  ihalo,iwrite
      REAL     M0,z0,W0

      REAL     Delta_collapse,Delta_crit,variance,xH
      EXTERNAL Delta_collapse,Delta_crit,variance,xH

c**********************************************************************

c---check that binsize is not too small

       Nhis = INT(5.0/binsize) + 1
       IF (Nhis.GT.Nhismax) CALL Terminate('invalid binsize')
       
c---initialize elapsed time

      CALL get_time

c---read global and cosmological parameters

      CALL read_global_param

c---read parameters specific to the merger trees

      CALL read_tree_param

c---initialize cosmological parameters to be used throughout

      CALL init_cosmo

c---initialize halo masses for which merger trees are to be constructed

      CALL setup_halo_masses
   
c---set up redshift grid for registering merger trees

      CALL setup_redshift_sampling
cc      CALL convert_MAHs
cc      STOP
c---initialize and store mass variance

      CALL init_variance   

c---initialize and define parameters & variables to be used

      CALL init_param

c---compute average/median MAHs according to some model(s)
c   (only if Mhalo is kept fixed)

      IF (.NOT.varyingMhalo.AND.imode.NE.2) THEN
        CALL get_model_MAH(Mhalo,znow)
      END IF

c---write output file with the threshold mass Mmin(z)

      CALL write_threshold_mass
            
c---construct merger tree

      WRITE(*,*)'           >>> Computing Merger Trees <<< '
      WRITE(*,*)' '
      WRITE(*,*)'                   Ntree = ',Ntree
      WRITE(*,*)' '
      WRITE(*,*)'ihalo  Mhalo      Nprog     Ntraj     Nstep  Norder Max
     &Res  TimeElapsed'
      WRITE(*,*)'-------------------------------------------------------
     &----------------'
      WRITE(*,82)'  MAX VALUES: ',Nmax,Nmaxtraj               
      WRITE(*,*)'-------------------------------------------------------
     &----------------'

      iwrite = 0      
      CALL get_time
      DO ihalo=1,Ntree

        Mhalo = Mh(ihalo)
 
        M0 = Mhalo
        z0 = znow
        W0 = Delta_collapse(z0)
        V0 = 159.43 * (M0/1.0E+12)**(1.0/3.0) * 
     &       (xH(z0)/xH_0)**(1.0/3.0) * 
     &       (Delta_crit(z0)/178.0)**(1.0/6.0) 

        CALL flush_tree()
        CALL build_merger_tree(M0,W0,z0)
        CALL analyze_merger_tree(ihalo,Ntot,Ntraj)
        CALL get_time

        WRITE(*,81)ihalo,ALOG10(Mhalo),Ntot,Ntraj,Nstep,Norder,
     &              MaxResolution,dt_h,dt_m,dt_s
        
        iwrite = iwrite + 1
        IF (.NOT.varyingMhalo.AND.imode.NE.2) THEN
          IF (iwrite.EQ.100.AND.ihalo.LT.Ntree) THEN
            CALL output_histograms(M0,W0,ihalo)
            CALL output_MAHs(W0,ihalo)
            CALL output_SHMFs(ihalo)
            iwrite = 0
          END IF
        END IF

        IF (imode.NE.1) CALL output_tree(ihalo,Ntraj,Nzsample)
         
      END DO

c---and write out results once more for all Ntree merger trees

      IF (.NOT.varyingMhalo.AND.imode.NE.2) THEN
        CALL output_histograms(M0,W0,Ntree)
        CALL output_MAHs(W0,Ntree)
        CALL output_SHMFs(Ntree)
        CALL output_subhalos()
      END IF

      WRITE(*,*)' '
      WRITE(*,*)'      >>> DONE: program completed succesfully <<<'
      WRITE(*,*)
      CALL write_time(1)

c---FORMATS

 81   FORMAT(1X,I5,2X,F5.2,2X,I9,2X,I8,2X,I8,2X,I6,1X,F6.2,1X,
     &I2,'h',I2,'m',F5.2,'s')
 82   FORMAT(A14,I10,I10,I10)

      STOP
      END

c**********************************************************************

      SUBROUTINE build_merger_tree(M0,W0,z0)
c---------------------------------------------------------------------------
c
c  Subroutine to build merger tree using one of ZFM methods
c  The tree information is stored in three matrices
c
c  Mprog(i) = M of progenitor   [Msun/h]      
c  Wprog(i) = W of progenitor
c  zprog(i) = z of progenitor
c
c  itree(i,1) = iparent       [=ID of parent of progenitor]
c  itree(i,2) = itrajectory   [=ID of trajectory of progenitor]
c  itree(i,3) = iorder        [=order of progenitor; 0 = MMP)
c
c  itraj(it,1) = ID of progenitor at leave-point of trajectory
c  itraj(it,2) = ID of progenitor along this trajetory AT ACCRETION
c  itraj(it,3) = order of subhalo at z=0 (order=0 --> MMP trajectory)
c
c  Ntot = total number of progenitors in tree
c  Nn   = total number of progenitors in tree at this timestep
c  Np   = total number of progenitors in tree of this parent
c  Ntraj= total number of trajectories (=`leaves' of tree)
c
c---------------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      INTEGER   iparent,Np,Nn,N1,N2,No,itot,it,iz
      REAL      W0,M0,M1,M2,z0,z,znew,Wnew

      REAL      variance
      EXTERNAL  variance

c---

c---initialize the base of the merger tree

      Mprog(0) = M0
      Wprog(0) = W0
      zprog(0) = z0

c---if Mmin is kept fixed, then compute its variance

      IF (.NOT.varyingMmin) THEN
        Mflimit = flimit * M0
        Sflimit = variance(Mflimit)**2
      END IF

c---initialize counters

      N1  = 0
      N2  = 0
      itot= 0 
      it  = 0
      iz  = 0
      No  = 0

c---start construction of tree

      DO WHILE (.TRUE.)                              ! timestep loop

        iz = iz + 1
         
        Nn = 0
        DO iparent = N1,N2                           ! loop over parents

          Mhost = Mprog(iparent)
          Whost = Wprog(iparent)
          zhost = zprog(iparent)

c---compute the minimum progenitor mass
        
          CALL get_minimum_mass(zhost)
          MaxResolution = MIN(MaxResolution,ALOG10(Mmin/Mhost))
   
c---compute timestep dW, and new redshift, znew

          CALL compute_timestep

          Wnew = Whost + dW
          CALL splint(vectorD,vectorZ,vectorZ2,Nsigma,Wnew,znew)

c---compute progenitor masses

          CALL get_progenitors(M1,M2,Np)
          Nn = Nn + Np

c---if zero progenitors, we are at the base of a trajectory

          IF (Np.LT.1) THEN
            it = it + 1
            itraj(it,1) = iparent
            itraj(it,3) = itree(iparent,3)
          END IF

c---if one or more progenitors, update mass etc
          
          IF (Np.GE.1) THEN
            itot = itot + 1
            Mprog(itot) = M1
            Wprog(itot) = Wnew
            zprog(itot) = znew
            itree(itot,1) = iparent
            itree(itot,3)  = itree(iparent,3)

            IF (Np.EQ.2) THEN
              itot = itot + 1
              Mprog(itot) = M2
              Wprog(itot) = Wnew
              zprog(itot) = znew
              itree(itot,1) = iparent
              itree(itot,3) = itree(iparent,3) + 1
              No = MAX(No,itree(itot,3))
            END IF
          END IF

c---make sure number of progenitors and trajectories are within bounds

          IF (it.GT.Nmaxtraj) CALL Terminate('Too many trajectories')
          IF (itot.GT.Nmax)   CALL Terminate('Too many progenitors')

        END DO                                   !!! proceed to next parent
 
        IF (Nn.LT.1) GOTO 30
        
        N1 = N2 + 1
        N2 = N1 + Nn - 1

      END DO                                     !!! proceed to next timestep 
      
 30   CONTINUE

c---store numbers of tree in global parameters, and write to screen

      Ntot  = itot
      Ntraj = it
      Nstep = iz
      Norder= No

      RETURN
      END

c**********************************************************************

      SUBROUTINE compute_timestep
c---------------------------------------------------------------------------
c
c  Compute the timestep dW
c 
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     eps1,eps2
      PARAMETER (eps1 = 0.1)
      PARAMETER (eps2 = 0.1)

      REAL     dW1,dW2

      REAL     variance,dlnSdlnM
      EXTERNAL variance,dlnSdlnM

c---

      s2 = variance(Mhost)
      sh = variance(Mhost/2.0)
      ah =-dlnSdlnM(Mhost/2.0) / 2.0

      Shalf = sh**2
      Shost = s2**2

      prefac = sqrt2divpi * G0 * (Whost/s2)**gamma2 

      ures = s2 / SQRT(Smin - Shost)
      qres = Mmin/Mhost
 
      Vres = Smin / (Smin - Shost)**1.5
      Vh   = Shalf/ (Shalf- Shost)**1.5

      IF (gamma1.GE.0.0) THEN
        muP = ah
      ELSE
        muP = -ALOG(Smin/Shalf) / (2.0 * ALOG(2.0*qres))
      END IF
      beP = ALOG(Vres/Vh)/ALOG(2.0*qres)
      etP = beP - 1.0 - (gamma1 * muP)
      Bfac = 2.0**beP * Vh

c---compute Nupper divided by deltaW

      IF (qres.GE.(0.5-epsq)) THEN
        NupperdivdW = 0.0
      ELSE
        Sqf1 = prefac * Bfac * ah * 2.0**(-muP*gamma1) *
     &        (sh/s2)**gamma1
        IF (ABS(etP).GT.epseta) THEN
          Sqf2 = (0.5**etP - qres**etP)/etP
        ELSE
          Sqf2 =-ALOG(2.0 * qres)
        END IF
        NupperdivdW = Sqf1 * Sqf2
      END IF

c---compute the timestep deltaW

      dW1 = eps1 * 1.414213562 * SQRT(Shalf - Shost)
      IF (qres.GE.(0.5-epsq)) THEN
        dW2 = 1.0E+20
      ELSE
        dW2 = eps2 / NupperdivdW
      END IF
      dW = MIN(dW1,dW2,dWmax)

      RETURN
      END

c**********************************************************************

      SUBROUTINE get_progenitors(M1,M2,Np)
c---------------------------------------------------------------------------
c
c  Select progenitors for a host halo of variance S0 using 
c  binary algorithm of Parkinson+08 
c 
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER  Np
      REAL     M1,M2,Mbuf,xlgJ,qq,sq,sq2,aq,Vq,Rq,FF
      REAL     r1,r2,r3,f1,f2,f3

      REAL     ran3,variance,dlnSdlnM
      EXTERNAL ran3,variance,dlnSdlnM

c---

c---initialize

      Np = 0
      M1 = 0.0
      M2 = 0.0
      Rq = 0.0

      r1 = ran3(iseed)
      r2 = ran3(iseed)
      r3 = ran3(iseed)

c---compute F

      IF (qres.GE.(0.5-epsq).AND.SQRT(Smin-Shost).LE.sqrt2divpi*dW) THEN
        FF = 1.0
      ELSE
        CALL linintpol(xJ,yJ,NJfunc,ALOG10(ures),xlgJ)
        FF = (prefac/s2) * 10.0**xlgJ * dW
      END IF

      IF (FF.GT.1.0) CALL Terminate('FF too large')

c---determine progenitor masses

      IF (r1.GT.(NupperdivdW * dW)) THEN
        M1 = Mhost * (1.0 - FF)
      ELSE
        IF (ABS(etP).GT.epseta) THEN
          qq = (qres**etP + Sqf2*etP*r2)**(1.0/etP)
        ELSE
          qq = qres * (2.0*qres)**(-r2)
        END IF
        sq = variance(qq*Mhost)
        sq2= sq**2
        aq =-dlnSdlnM(qq*Mhost) / 2.0
        Vq = sq2 / (sq2 - Shost)**1.5
        Rq = (aq/ah) * Vq/(Bfac * qq**beP) * 
     &      ((2.0*qq)**muP * sq/sh)**gamma1
        IF (r3.LT.Rq) THEN
          M2 = qq * Mhost
          M1 = Mhost * (1.0 - FF - qq)
        ELSE
          M1 = Mhost * (1.0 - FF)
        END IF
      END IF

c---If any of the progenitor masses is less than the resolution mass
c   we consider that progenitor part of the smooth accretion

      IF (M1.GT.Mmin) Np = Np + 1
      IF (M2.GT.Mmin) Np = Np + 1

c---make sure that M1 >= M2, which is required for the tree structure

      IF (M2.GT.M1) THEN
        Mbuf = M1
        M1 = M2
        M2 = Mbuf
      END IF

c---Rq has to be smaller than unity; we check this allowing for
c   a computational precission of 1.0E-3

      IF (Rq.GT.1.001) THEN
        f1 = (aq/ah) 
        f2 = Vq/(Bfac * qq**beP) 
        f3 = ((2.0*qq)**muP * sq/sh)**gamma1         
        WRITE(*,*)' WARNING: Rq too large; ',Rq,Mhost
ccc        WRITE(*,*)f1,f2,f3,Mhost/2.0,Mmin,ah,aq
ccc        WRITE(*,*)Vq,Bfac,qq,beP,etP,sq2,Shost
      END IF

      RETURN
      END
      
c**********************************************************************

      SUBROUTINE get_minimum_mass(z)
c----------------------------------------------------------------------
c
c  Compute the minimum progenitor mass. If we allow it to vary, then
c  we set it to the reionization filtering mass (transiting from 
c  a circular velocity of 10 km/s prior to reionization to 30 km/s
c  after reionization. zrei is the reionization redshift, and afac
c  controls the sharpness of the transition.
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z,Vc_i,Vc_f,zrei,afac,ffac,Vc3

      REAL     variance,xH
      EXTERNAL variance,xH

c---

      Vc_i = 10.0
      Vc_f = 30.0
      zrei = 6.0
      afac = 10.0

      IF (varyingMmin) THEN
c        ffac = (1.0+TANH(afac*(z-zrei)))/2.0
c        Vc3 = Vc_f**3 + ffac * (Vc_i**3 - Vc_f**3)
        IF (z.LT.zrei) THEN
          Vc3 = Vc_i - (z/zrei - 1.0) * (Vc_f - Vc_i)
        ELSE
          Vc3 = Vc_i
        END IF
        Mmin = Vc3**3 / (10. * gee * xH(z))
        Smin = variance(Mmin)**2
      ELSE
        Mmin = Mflimit
        Smin = Sflimit
      END IF

      RETURN
      END 

c**********************************************************************

      SUBROUTINE setup_halo_masses
c----------------------------------------------------------------------
c
c  Store halo masses for which merger trees are to be computed
c  
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER   i
      REAL      xlgM
      CHARACTER infile*60

c---

      infile = 'halomasses.dat' 

      IF (varyingMhalo) THEN
        OPEN(10,file=infile,status='OLD')
        DO i=1,Ntree
          READ(10,*)xlgM
          Mh(i) = 10.0**xlgM
        END DO
        CLOSE(10)
      ELSE
        DO i=1,Ntree
          Mh(i) = Mhalo
        END DO
      END IF

      RETURN
      END

c**********************************************************************
c
c      SUBROUTINE convert_MAHs
cc----------------------------------------------------------------------
cc
cc  Setup redshifts sampling for registering merger tree. 
cc  We do this using fractions of the local dynamical time.
cc
cc----------------------------------------------------------------------
c
c      INCLUDE 'paramfile.h'
c
c      INTEGER  i,iID
c      REAL     a,xM,xV,c,z6,t6
c
c      REAL     lookbacktime
c      EXTERNAL lookbacktime
c      
cc---
c
c      OPEN(11,file="a_sampling.dat",status='OLD')
c      OPEN(12,file="tz_sampling.dat",status='UNKNOWN')
c      DO i=1,181
c        READ(11,*)iID,a,xM,xV,c
c        z6 = (1.0/a) - 1.0
c        t6 = lookbacktime(z6)
c        WRITE(12,122)i,a,z6,t6
c      END DO
c      CLOSE(11)
c       
c 122  FORMAT(I4,2X,3(F8.4,2X))       
c
c      RETURN
c      END
c       
c**********************************************************************

      SUBROUTINE setup_redshift_sampling
c----------------------------------------------------------------------
c
c  Setup redshifts sampling for registering merger tree. 
c  We do this using fractions of the local dynamical time.
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER  iz
      REAL     t,z,tff,tstep
      REAL     xz1,xz2,xz3,xz4,xz5,xz6
      REAL     xy1,xy2,xy3,xy4,xy5,xy6
      REAL     dz1,dz2,dz3,dz4,dz5,dz6
      CHARACTER outfil1*60

      REAL    tatz
      COMMON/zatt/ tatz

      INTEGER  lblnk
      REAL     lookbacktime,z_at_t,zriddr
      EXTERNAL lookbacktime,z_at_t,zriddr,lblnk

c---

c---initialize

      DO iz = 1,Nzmax
        zsample(iz) = -1.0
      END DO

c---sample every fraction `fsample' of the free-fall time at an
c   overdensity of 200 (at that time)

      z = znow
      t = lookbacktime(z)
      iz = 1
      DO WHILE (z.LE.zmaxsample)
        zsample(iz) = z
        tsample(iz) = lookbacktime(z)
        tff = tff0 * (1.0+z)**(-1.5)
        tstep = MIN(0.1,frac_tff * tff)
        t = t + (frac_tff * tff)
ccc        t = t + tstep
        tatz = t
        z = zriddr(z_at_t,0.0,1.2*zmaxsample,1.0E-4)
        iz = iz + 1        
      END DO
      Nzsample = iz-1

      IF (Nzsample.GT.Nzmax) CALL Terminate('Nzsample too large')

      WRITE(*,*)' Redshift Sampling: Nz = ',Nzsample
      WRITE(*,*)' '
     
c---write to outputfile if required

ccc      IF (imode.NE.1) THEN
        outfil1='/TimeSteps.dat'
        OPEN(43,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN')
        DO iz=1,Nzsample
          WRITE(43,433)iz,zsample(iz),tsample(iz)
        END DO
        CLOSE(43)
ccc      END IF

c---specify the redshifts of interest (only if doing statistics)

      IF (imode.NE.2) THEN
        xz1 = 0.1
        xz2 = 0.5
        xz3 = 1.0
        xz4 = 3.0
        xz5 = 6.0
        xz6 =10.0

        xy1 = 99.99
        xy2 = 99.99
        xy3 = 99.99
        xy4 = 99.99
        xy5 = 99.99
        xy6 = 99.99

        DO iz = 1,Nzsample
          dz1 = ABS(zsample(iz) - xz1)
          dz2 = ABS(zsample(iz) - xz2)
          dz3 = ABS(zsample(iz) - xz3)
          dz4 = ABS(zsample(iz) - xz4)
          dz5 = ABS(zsample(iz) - xz5)
          dz6 = ABS(zsample(iz) - xz6)

          IF (dz1.LT.xy1) THEN
            xy1 = dz1
            iz1 = iz
          END IF
          IF (dz2.LT.xy2) THEN
            xy2 = dz2
            iz2 = iz
          END IF
          IF (dz3.LT.xy3) THEN
            xy3 = dz3
            iz3 = iz
          END IF
          IF (dz4.LT.xy4) THEN
            xy4 = dz4
            iz4 = iz
          END IF
          IF (dz5.LT.xy5) THEN
            xy5 = dz5
            iz5 = iz
          END IF
          IF (dz6.LT.xy6) THEN
            xy6 = dz6
            iz6 = iz
          END IF
        END DO
  
        WRITE(*,*)' Output redshifts to be used:'
        WRITE(*,*)zsample(iz1),iz1
        WRITE(*,*)zsample(iz2),iz2
        WRITE(*,*)zsample(iz3),iz3
        WRITE(*,*)zsample(iz4),iz4
        WRITE(*,*)zsample(iz5),iz5
        WRITE(*,*)zsample(iz6),iz6
        WRITE(*,*)' '

      END IF

 433  FORMAT(I5,2X,F9.4,2X,F9.6)

      RETURN
      END

c**********************************************************************

