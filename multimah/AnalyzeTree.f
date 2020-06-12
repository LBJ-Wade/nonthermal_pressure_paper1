c**********************************************************************
c       SUBROUTINES THAT ANALYZE A GIVEN MERGER TREE
c**********************************************************************

      SUBROUTINE analyze_merger_tree(k,Np,Nt)
c----------------------------------------------------------------------
c
c  Analyze merger tree k, which consists of a total of Np progenitors
c  distributed over Nt trajectories.
c
c  We start by computing (and storing) some useful information for
c  each of the Nt trajectories:
c    itraj(it,2) = ID of progenitor when it is host halo for last time
c                   i.e., before it becomes a subhalo
c  Finally, we also determine, for each progenitor, on which trajectory
c  it resides. This is always the trajectory with the lowest possible 
c  order. This information is stored in itree(j,2). 
c
c  Once these properties are computed and stored, we perform a
c  number of analyses.
c
c-----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER   j,k,it,ktree,jnew,korder
      INTEGER   Np,Nt,Nprogenitors,Ntrajectories
      REAL      z,zmax

c---

      Ntrajectories = Nt
      Nprogenitors = Np
      ktree = k

c---store properties of trajectories

      zmax = 0.0
      DO it=1,Ntrajectories

        j = itraj(it,1)
        zmax = MAX(zmax,zprog(j))
        itree(j,2) = it

        korder = 0
        DO WHILE (korder.EQ.0.AND.j.GT.0) 
          jnew = itree(j,1)
          IF (itree(jnew,3).LT.itree(j,3)) THEN
            itraj(it,2) = j
            korder = korder + 1
          ELSE
            itree(jnew,2)=it
          END IF
          j = jnew
        END DO
      END DO

c---Resample merger tree on a pre-set redshift grid

      CALL resample_merger_tree(Ntrajectories)
      
c---compute statistics of unevolved SubHalo Mass Functions

      IF (imode.NE.2) CALL analyze_unevolved_SHMF(ktree,Ntrajectories)

c---Evolve the SubHalo Mass Functions using simple stripping model

      IF (evolve) CALL evolve_SHMF(ktree,Ntrajectories)
      
c---compute Vmax if required

      IF (compVmax) CALL compute_Vmax(ktree,Ntrajectories)

c---and write some statistics to file (if required)

      IF (imode.NE.2) CALL stat_Vmax(ktree,Ntrajectories)

c---compute Mass Assembly Histories and halo assembly times

      IF (imode.NE.2) CALL analyze_MAH(ktree,Ntrajectories)
      IF (ktree.LE.10000) CALL writeMAH(ktree)

      RETURN
      END

c**********************************************************************

      SUBROUTINE resample_merger_tree(Ntrajectories)
c----------------------------------------------------------------------
c
c  Resample the merger tree onto the predetermined redshift array
c  zsample(1:Nzsample)
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER    iz,iz0,it,j,jnew,korder,Ntrajectories,itr_host
      REAL       M,z

      REAL*8     magicnumber
      EXTERNAL   magicnumber

c---

      DO iz=1,Nzsample
        DO it = 1,Ntrajectories
          mergertree(it,iz)=0.0d0
        END DO
      END DO

c---

      DO it = 1,Ntrajectories

        j = itraj(it,1)
        z = zprog(j)
        CALL locate(zsample,Nzsample,z,iz)

        iz0 = MIN(iz+1,Nzsample)
        korder = 0
        mergertree(it,iz0) = magicnumber(korder,it,Mprog(j),Vbuf)

        korder = 0
        DO WHILE (j.GT.0.AND.iz.GT.0)

          jnew = itree(j,1)
          IF (itree(jnew,3).LT.itree(j,3)) korder = korder + 1
 
          IF (zprog(jnew).LE.zsample(iz)) THEN

            IF (korder.EQ.0) THEN
              M = Mprog(jnew)
              itr_host = itree(jnew,2)   
            ELSE
              M = Mprog(itraj(it,2))
              itr_host = itree(itree(itraj(it,2),1),2)
            END IF
   
            IF (imode.NE.2) CALL update_histograms(iz,M,korder)
             
 77         mergertree(it,iz) = magicnumber(korder,itr_host,M,Vbuf)

            iz = iz - 1
            IF (iz.GT.0) THEN
              IF (zprog(jnew).LE.zsample(iz)) THEN
                IF (imode.NE.2) CALL update_histograms(iz,M,korder)
                GOTO 77
              END IF
            END IF
          END IF

          j = jnew

        END DO
      END DO                    ! loop of trajectories

 23   FORMAT(I3,1X,I7,1X,F7.3,1X,F7.3,1X,I4,1X,I7,1X,F15.5)

      RETURN    
      END

c**********************************************************************

      SUBROUTINE compute_Vmax(k,Ntrajectories)
c----------------------------------------------------------------------
c
c  Convert the halo masses to maximum circular velocities
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER    Nbuf
      PARAMETER  (Nbuf=10000)
      
      INTEGER    iz,it,i,j,ileaf,Np,Ntrajectories,korder,itt,io,k
      INTEGER    kbuf(Nbuf)  
      REAL       M,z,Mbuf(Nbuf),zbuf(Nbuf),Mleaf,zleaf,Wnew,znew
      REAL       M0,M1,M2,Macc,z0,z4,c0,Vmax,Vacc,xlgM,xlgV,f0
      REAL       f1,f2,z5  
      REAL*8     magicnum

      REAL*8     magicnumber
      EXTERNAL   magicnumber

      REAL       get_Vmax,Delta_collapse,variance,time
      EXTERNAL   get_Vmax,Delta_collapse,variance,time
         
c---

c---for each trajectory, first construct mass accretion history all 
c   the way down to time when M_mmp < 0.04 M_leaf

      DO it=1,Ntrajectories

c---first store what is already available

         DO iz=1,Nzsample
           magicnum = mergertree(it,iz)
           IF (magicnum.GT.1.0d0) THEN
             CALL deconstruct(magicnum,korder,itt,xlgM,xlgV)     
             Mbuf(iz) = 10.0**xlgM
             zbuf(iz) = zsample(iz)
             kbuf(iz) = korder
           ELSE
             ileaf = iz - 1 
             Mleaf = Mbuf(ileaf)
             zleaf = zbuf(ileaf)  
             GOTO 55
           END IF  
         END DO
55      CONTINUE 

c---and now extent to higher z; note, this procedure works as long
c   as Mmin < Mleaf = Mhost/25. I verified that results are independent
c   of value of Mmin (ranging from Mhost/1000 to Mhost/30), but is much
c   faster for larger values (hence, default is Mhost/30)

        Mhost = Mleaf
        zhost = zleaf
        Whost = Delta_collapse(zhost)

        Mmin = Mhost/30.0
        Smin = variance(Mmin)**2

c---construct MAH from leaf down to point where M_mmp < 0.04 Mleaf 

        i = ileaf
        DO WHILE (Mhost.GT.0.04*Mleaf)
          i = i + 1
          CALL compute_timestep
          Wnew = Whost + dW
          CALL splint(vectorD,vectorZ,vectorZ2,Nsigma,Wnew,znew)
          CALL get_progenitors(M1,M2,Np)
          Mbuf(i) = M1
          zbuf(i) = znew
          kbuf(i) = 0
          Mhost = M1
          Whost = Wnew
        END DO 
        
c---fill rest of vectors with buffer information to allow for linintpol
        
        DO j=i+1,Nbuf
          Mbuf(j) = 0.99*Mbuf(j-1)
          zbuf(j) = 1.01*zbuf(j-1)
          kbuf(j) = 0
        END DO
          
c---now compute Vmax. In the case of subhaloes, we use an equation that 
c   was provided to me by Andrey Kravtsov and based on simulations.
c   (see e-mail from Arthur d.d. Dec 2, 2013)

        Vacc = 0.0
        DO iz = ileaf,1,-1
          M0 = Mbuf(iz)
          z0 = zbuf(iz)
          io = kbuf(iz)
          IF (iz.EQ.ileaf.AND.io.NE.0) CALL TERMINATE('ileaf error') 
          IF (io.EQ.0) THEN
            CALL linintpol(Mbuf,zbuf,Nbuf,0.5*M0,z5)            
            CALL linintpol(Mbuf,zbuf,Nbuf,0.04*M0,z4)            

c---Original Zhao model
ccc            f0 = time(z0) / (3.75 * time(z4))
ccc            c0 = 4.0 * (1.0 + f0**8.4)**(0.125)
c---Modified Zhao model
c            f0 = time(z0) / (3.40 * time(z4))
            f0 = time(z0) / (3.40 * time(z4))
            c0 = 4.0 * (1.0 + f0**6.5)**(0.125)
c---Giocoli+12 model
ccc            f1 = time(z0)/time(z4)
ccc            f2 = time(z0)/time(z5) 
ccc            c0 = 0.45 * (4.23 + f1**1.15 + f2**2.3)
            Vmax = get_Vmax(M0,z0,c0)
            Vacc = Vmax
            Macc = M0
            IF (iz.EQ.1) conc(k) = c0
          ELSE
            f0 = ((M0/Macc)**0.44) / ((1.0 + (M0/Macc))**0.60)
            Vmax = 2.0**0.6 * f0 * Vacc 
          END IF
          mergertree(it,iz) = magicnumber(io,it,M0,Vmax)
        END DO

        Vaccretion(it) = Vacc
        
      END DO                                          ! Ntrajectories loop

      RETURN    
      END

c**********************************************************************

      SUBROUTINE analyze_MAH(k,Nt)
c----------------------------------------------------------------------
c
c  Determine the Mass Assembly History (MAH) and some of the related
c  statistics, such as formation/assembly times.
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER k,it,iz,korder,itt,iten,ihalf,ifour,Nt,Ntrajectories
      REAL    xlgM,xlgV,z1,z2,M1,M2
      REAL*8  magicnum

c---
      
      Ntrajectories = Nt

c---register MAH of MMP in a 1D array      

      DO it = 1,Ntrajectories
        IF (itraj(it,3).EQ.0) THEN
           DO iz=1,Nzsample
             magicnum = mergertree(it,iz)
             IF (magicnum.GT.0.0) THEN
               CALL deconstruct(magicnum,korder,itt,xlgM,xlgV)     
               IF (korder.NE.0) CALL Terminate('Incorrect order of MAH')
             ELSE  
               xlgM = -20.0
               xlgV = -20.0
             END IF  
             mah(k,iz,1) = xlgM - ALOG10(Mhalo)              
             mah(k,iz,2) = xlgV - ALOG10(V0)
           END DO 
        END IF
      END DO

c---make certain that MAH at z=0 is correct.

      mah(k,1,1) = 0.0

c---compute redshifts at which Mmmp/M0 = 0.5 and 0.1

      iten = 0
      ihalf= 0
      ifour= 0
      DO iz=1,Nzsample
        IF (mah(k,iz,1).LT.-0.30103.AND.ihalf.EQ.0) THEN
          z1 = zsample(iz-1)
          z2 = zsample(iz)
          M1 = 10.0**mah(k,iz-1,1)
          M2 = 10.0**mah(k,iz,1)
          zform(k,1) = z1 + ((0.5-M1)/(M2-M1)) * (z2-z1)
          ihalf=1
        END IF
        IF (mah(k,iz,1).LT.-1.0.AND.iten.EQ.0) THEN
          z1 = zsample(iz-1)
          z2 = zsample(iz)
          M1 = 10.0**mah(k,iz-1,1)
          M2 = 10.0**mah(k,iz,1)
          zform(k,2) = z1 + ((0.1-M1)/(M2-M1)) * (z2-z1)
          iten=1
        END IF
        IF (mah(k,iz,1).LT.-1.39794.AND.ifour.EQ.0) THEN
          z1 = zsample(iz-1)
          z2 = zsample(iz)
          M1 = 10.0**mah(k,iz-1,1)
          M2 = 10.0**mah(k,iz,1)
          zform(k,3) = z1 + ((0.04-M1)/(M2-M1)) * (z2-z1)
          ifour=1
        END IF
      END DO

      RETURN
      END

c***************************************************************************

      SUBROUTINE analyze_unevolved_SHMF(k,Ntrajectories)
c---------------------------------------------------------------------------
c
c  Store information on unevolved SHMFs of order 1 to 5
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER i,k,it,j1,j2,jj,n,Ntrajectories
      REAL    x,macc,zacc

c---

      DO it=1,Ntrajectories
        macc = Mprog(itraj(it,2))
        zacc = zprog(itraj(it,2))
        j1 = INT(ABS(ALOG10(macc/Mhalo)) / binsize) + 1
        j2 = INT(ALOG10(1.0+zacc)/0.05) + 1
        n = itraj(it,3)
        IF (j1.LE.Nhis.AND.n.GE.1.AND.n.LE.5) THEN  
          histo_sub(j1,n,1) = histo_sub(j1,n,1) + 1.0
          zacc_sub(j2,n,1) = zacc_sub(j2,n,1) + 1.0
          DO jj = 1,Nhis
            x = -(binsize * FLOAT(jj-1)) - (binsize/2.)
            IF (ALOG10(macc/Mhalo).GT.x) THEN
              histo_cum(k,jj,n,1) = histo_cum(k,jj,n,1) + 1.0
              histo_cum(k,jj,0,1) = histo_cum(k,jj,0,1) + 1.0
            END IF
          END DO
        END IF
      END DO

      RETURN
      END

c***************************************************************************

      SUBROUTINE evolve_SHMF(k,Ntrajectories)
c---------------------------------------------------------------------------
c
c Evolve the subhalo masses due to tidal stripping we update mergertree(i,j) 
c with the evolved subhalo masses. 
c NOTE: we start with the lowest order subhalos, and step up in order. 
c       That way the subhalo is always evolved in the proper, evolved
c       parent mass. If a hosting subhalo is dissolved, its subhaloes 
c       are passed on to the (sub)halo that hosts the dissolved one.
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER   it,iz,i,j,j1,j2,j3,jj,k,n,iparent,Ntrajectories,iorder
      INTEGER   ix1(Nmaxtraj),ix2(Nmaxtraj),itt,korder,ibuf1,ibuf2
      INTEGER   ii,nn,iy
      REAL*8    magicnum
      REAL      msub,mparent,xx1(Nmaxtraj),xx2(Nmaxtraj)
      REAL      x,zacc,zaver,dt,xlgM,xlgV,macc,boost

      REAL      lookbacktime,fstrip,gasdev
      EXTERNAL  lookbacktime,fstrip,gasdev

      REAL*8     magicnumber
      EXTERNAL   magicnumber

c---


      DO iorder = 1,Norder        
        DO it = 1,Ntrajectories
          IF (itraj(it,3).EQ.iorder) THEN
            msub = 0.0
ccc            boost = 10.0**(0.29*gasdev(iseed))
            boost = 10.0**(0.17*gasdev(iseed))
            DO iz = Nzsample,1,-1
              magicnum = mergertree(it,iz)
              IF (magicnum.GT.1.0d0) THEN
                CALL deconstruct(magicnum,korder,itt,xlgM,xlgV)     
                IF (itt.EQ.it.OR.msub.EQ.0.0) THEN
                  msub = 10.0**xlgM
                  macc = msub
                ELSE
 48               magicnum = mergertree(itt,iz)
                  CALL deconstruct(magicnum,ibuf1,ibuf2,xlgM,xlgV)     
                  mparent = 10.0**xlgM
                  dt = tsample(iz+1) - tsample(iz)
                  zaver = (zsample(iz+1) + zsample(iz))/2.0
                  IF (mparent.LT.10.2) THEN
                    itt = ibuf2
                    GOTO 48
                  END IF  
                  msub = msub * (1.0 - boost + 
     &                    boost * fstrip(msub,mparent,dt,zaver)) 
                  IF (msub.LT.fdissolve*macc) msub = 10.1
                END IF
                mergertree(it,iz) = magicnumber(korder,itt,msub,Vbuf)
              END IF
            END DO
          END IF
        END DO
      END DO

c---write results to file

      DO it = 1,Ntrajectories
        magicnum = mergertree(it,1)
        CALL deconstruct(magicnum,n,itt,xlgM,xlgV)     
        
        IF (n.EQ.0) GOTO 34
        msub = 10.0**xlgM
        macc = Mprog(itraj(it,2))

c---subtract from subhalo masses the masses of it own subhalos
c   THIS MIMICS SUBFIND

cc        DO iy=1,Ntrajectories
cc          magicnum = mergertree(iy,1)
cc          CALL deconstruct(magicnum,nn,ii,xlgM,xlgV)     
cc          IF (ii.EQ.it.AND.nn.EQ.n+1) THEN
cc            msub = msub - 10.0**xlgM
cc          END IF
cc        END DO

c---update histogram of evolved subhalo masses

        j = INT(ABS(ALOG10(msub/Mhalo)) / binsize) + 1
        IF (j.LE.Nhis.AND.n.GE.1.AND.n.LE.5) THEN
          histo_sub(j,n,2) = histo_sub(j,n,2) + 1.0
          DO jj=1,Nhis
            x = -(binsize * FLOAT(jj-1)) - (binsize/2.)
            IF (ALOG10(msub/Mhalo).GT.x) THEN
              histo_cum(k,jj,n,2) = histo_cum(k,jj,n,2) + 1.0
              histo_cum(k,jj,n,3) = histo_cum(k,jj,n,3) + msub
              histo_cum(k,jj,0,2) = histo_cum(k,jj,0,2) + 1.0
              histo_cum(k,jj,0,3) = histo_cum(k,jj,0,3) + msub
            END IF
          END DO
        END IF

c---update histograms on msub(now)/msub(accretion); keep track of
c   two such histograms; one with all subhaloes, and one in which we
c   only count those that at z=znow are above the mass limit

        j = INT(ABS(ALOG10(msub/macc)) / binsize) + 1
        IF (j.LE.Nhis.AND.n.GE.1.AND.n.LE.5) THEN
          histo_sub(j,n,3) = histo_sub(j,n,3) + 1.0
          IF ((msub/Mhalo).GT.flimit) THEN
            histo_sub(j,n,4) = histo_sub(j,n,4) + 1.0
          END IF
        END IF

c---store some properties specific to this merger-tree

        IF ((msub/Mhalo).GT.1.0E-4) THEN
          itree_prop(k,1,0)=itree_prop(k,1,0) + 1
          tree_prop(k,1,0) = tree_prop(k,1,0) + (msub/Mhalo)
          IF (n.EQ.1.OR.n.EQ.2) THEN
            itree_prop(k,1,n)=itree_prop(k,1,n) + 1
            tree_prop(k,1,n) = tree_prop(k,1,n) + (msub/Mhalo)
          END IF
        END IF
        IF ((msub/Mhalo).GT.1.0E-3) THEN
          itree_prop(k,2,0)=itree_prop(k,2,0) + 1
          tree_prop(k,2,0) = tree_prop(k,2,0) + (msub/Mhalo)
          IF (n.EQ.1.OR.n.EQ.2) THEN
            itree_prop(k,2,n)=itree_prop(k,2,n) + 1
            tree_prop(k,2,n) = tree_prop(k,2,n) + (msub/Mhalo)
          END IF
        END IF
        IF ((msub/Mhalo).GT.1.0E-2) THEN
          itree_prop(k,3,0)=itree_prop(k,3,0) + 1
          tree_prop(k,3,0) = tree_prop(k,3,0) + (msub/Mhalo)
          IF (n.EQ.1.OR.n.EQ.2) THEN
            itree_prop(k,3,n)=itree_prop(k,3,n) + 1
            tree_prop(k,3,n) = tree_prop(k,3,n) + (msub/Mhalo)
          END IF
        END IF
        IF ((msub/Mhalo).GT.1.0E-1) THEN
          itree_prop(k,4,0)=itree_prop(k,4,0) + 1
          tree_prop(k,4,0) = tree_prop(k,4,0) + (msub/Mhalo)
          IF (n.EQ.1.OR.n.EQ.2) THEN
            itree_prop(k,4,n)=itree_prop(k,4,n) + 1
            tree_prop(k,4,n) = tree_prop(k,4,n) + (msub/Mhalo)
          END IF
        END IF

c---store progenitor masses (at accretion and at znow) as well as
c   the order. These will be sorted later (hence the negative sign)

        xx1(it) =-ALOG10(Mprog(itraj(it,2))/Mhalo)
        xx2(it) =-ALOG10(msub/Mhalo)
        ix1(it) = itraj(it,3)
        ix2(it) = itraj(it,3)

c---if msub/Mhalo > flimit than store accretion time

        IF ((msub/Mhalo).GT.flimit) THEN
          IF (n.GE.1.AND.n.LE.5) THEN  
            zacc = zprog(itraj(it,2))
            j2 = INT(ALOG10(1.0+zacc)/0.05) + 1
            zacc_sub(j2,n,2) = zacc_sub(j2,n,2) + 1.0
          END IF
        END IF

 34     CONTINUE

      END DO

c---store 20 most massive haloes (both at accretion and at znow)
c   as well as their orders

      CALL sort2(Ntrajectories,xx1,ix1)
      CALL sort2(Ntrajectories,xx2,ix2)

      DO n=1,20
        Msubhalos(k,n,1) =-xx1(n)
        isubhalos(k,n,1) = ix1(n)
        Msubhalos(k,n,2) =-xx2(n)
        isubhalos(k,n,2) = ix2(n)
      END DO

      RETURN
      END

c**********************************************************************

      SUBROUTINE stat_Vmax(k,Ntrajectories)
c----------------------------------------------------------------------
c
c  Compute statistics based on Vmax and Vacc
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER   it,itt,n,j,jj,k,Ntrajectories
      REAL*8    magicnum
      REAL      xv,Vparam,Vacc,Vmax,Vhost,xlgV,xlgM
      
c---

      it = 1
      Vhost = 0.0
      DO WHILE (Vhost.LE.1.0E-4)
        magicnum = mergertree(it,1)
        CALL deconstruct(magicnum,n,itt,xlgM,xlgV)             
        IF (n.EQ.0) Vhost = 10.0**xlgV
        it = it + 1
      END DO
      
c---store SHMF in terms of both V_{acc,sub}/V_{vir,host}
c   and V_{max,sub}/V_{vir,host}

      DO it = 1,Ntrajectories

        magicnum = mergertree(it,1)
        CALL deconstruct(magicnum,n,itt,xlgM,xlgV)             
        Vmax = 10.0**xlgV
        Vacc = Vaccretion(it)
        
        Vparam = ALOG10(Vacc/V0)
        j = INT(ABS(Vparam) / (binsize/3.)) + 1
        IF (j.LE.Nhis.AND.Vparam.LT.0.0.AND.n.GE.1.AND.n.LE.5) THEN
          histo_sub(j,n,5) = histo_sub(j,n,5) + 1.0
          DO jj=1,Nhis
            xv = -((binsize/3.) * FLOAT(jj-1)) - (binsize/6.)
            IF (Vparam.GT.xv) THEN 
              histo_cum(k,jj,n,4) = histo_cum(k,jj,n,4) + 1.0
              histo_cum(k,jj,0,4) = histo_cum(k,jj,0,4) + 1.0
            END IF
          END DO
        END IF

        Vparam = ALOG10(Vmax/V0)
        j = INT(ABS(Vparam) / (binsize/3.)) + 1
        IF (j.LE.Nhis.AND.Vparam.LT.0.0.AND.n.GE.1.AND.n.LE.5) THEN
          histo_sub(j,n,6) = histo_sub(j,n,6) + 1.0
          DO jj=1,Nhis
            xv = -((binsize/3.) * FLOAT(jj-1)) - (binsize/6.)
            IF (Vparam.GT.xv) THEN
              histo_cum(k,jj,n,5) = histo_cum(k,jj,n,5) + 1.0
              histo_cum(k,jj,0,5) = histo_cum(k,jj,0,5) + 1.0
            END IF
          END DO
        END IF

        Vparam = ALOG10(Vmax/Vhost)
        j = INT(ABS(Vparam) / (binsize/3.)) + 1
        IF (j.LE.Nhis.AND.Vparam.LT.0.0.AND.n.GE.1.AND.n.LE.5) THEN
          histo_sub(j,n,7) = histo_sub(j,n,7) + 1.0
          DO jj=1,Nhis
            xv = -((binsize/3.) * FLOAT(jj-1)) - (binsize/6.)
            IF (Vparam.GT.xv) THEN
              histo_cum(k,jj,n,6) = histo_cum(k,jj,n,6) + 1.0
              histo_cum(k,jj,0,6) = histo_cum(k,jj,0,6) + 1.0
            END IF
          END DO
        END IF
        
      END DO

      RETURN
      END
       
c**********************************************************************
    
      REAL*8 FUNCTION magicnumber(io,it,M,V)

      INCLUDE 'paramfile.h'

      INTEGER*8 imagicnumber
      INTEGER   io,it
      REAL*8    xM,xV,xQ
      REAL      M,V      

c---

c---the MAX(M,10) is done to prevent the digits behind the comma from
c   becoming zero, which prevents the `magicnumber' from doing its magic.

      xV = DLOG10(DMAX1(DBLE(V),1.0d0+deps))
      xM = DLOG10(DMAX1(DBLE(M),10.0d0+deps))
      xQ = DBLE(IDINT(xM * 1000.0d0)) + xV/10.0d0
      
      imagicnumber = (io+1) * trajsize + it
      magicnumber = DBLE(imagicnumber) + (xQ/1.0D+5)

      END

c**********************************************************************
 
      SUBROUTINE deconstruct(magicnum,io,it,xlgM,xlgV)

      INCLUDE 'paramfile.h'
      
      INTEGER*8 imagicnum
      INTEGER   io,it
      REAL*8    magicnum,xQ
      REAL      xlgM,xlgV
      
c---

      imagicnum = INT(magicnum)
      io = INT(magicnum/trajsize) - 1
      it = MOD(imagicnum,trajsize)
      
      xQ = (magicnum - DBLE(imagicnum)) * 1.0D+5
      xlgM = SNGL(IDNINT(xQ) / 1000.0d0)
      xlgV = SNGL((xQ - DBLE(IDNINT(xQ))) * 10.0d0)
      
      RETURN
      END

c**********************************************************************
     
      REAL FUNCTION fstrip(msub,mtot,delta_t,z)
 
      INCLUDE 'paramfile.h'

      REAL    z,msub,mtot,delta_t,tdyn,fac

      REAL     xH,Delta_crit,XEXP
      EXTERNAL xH,Delta_crit,XEXP

c---
  
      fac = SQRT(Delta_crit(z)/deltacrit0) * xH(z)/xH_0

      tdyn = tau / fac

      IF (slope.EQ.0.0) THEN
        fstrip = XEXP(-delta_t/tdyn)
      ELSE
        fstrip = (1.0 + slope * (msub/mtot)**slope * (delta_t/tdyn))
        fstrip = fstrip**(-1.0/slope)
      END IF

      END

c**********************************************************************
     
      REAL FUNCTION get_Vmax(M,z,c)
 
      INCLUDE 'paramfile.h'

      REAL    M,z,c,Vvir,fc
      
      REAL     xH,Delta_crit
      EXTERNAL xH,Delta_crit
      
c---

      Vvir = 159.43 * (M/1.0E+12)**(1.0/3.0) * (xH(z)/xH_0)**(1.0/3.0) * 
     &       (Delta_crit(z)/178.0)**(1.0/6.0) 
      fc = SQRT(c / (ALOG(1.0+c) - (c/(1.0+c))))
      
      get_Vmax = 0.465 * Vvir * fc
      
      END    
c**********************************************************************



 

