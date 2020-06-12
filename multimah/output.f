c**********************************************************************
c       SUBROUTINES THAT OUTPUT RESULTS TO SCREEN OR FILE
c**********************************************************************

      SUBROUTINE output_histograms(M0,W0,Nt)
c---------------------------------------------------------------------------
c
c  initialize the tree structure
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER i,j,Nt
      REAL     M0,W0,M,x,xx,S1,dS,fac
      REAL     xz1,xz2,xz3,xz4,xz5,xz6,dW1,dW2,dW3,dW4,dW5,dW6
      REAL     y1,y2,y3,y4,y5,y6,u1,u2,u3,u4,u5,u6,temp(50,6)
      CHARACTER outfile*30

      REAL     variance,fEPS_SC,fEPS_EC,dlnSdlnM
      EXTERNAL variance,fEPS_SC,fEPS_EC,dlnSdlnM

      INTEGER  lblnk
      REAL     delta_collapse
      EXTERNAL lblnk,delta_collapse

c---

      Shost = variance(M0)**2
      Whost = W0

      outfile='/histo.dat'
      OPEN(10,file=moddir(1:lblnk(moddir))//outfile(1:lblnk(outfile)),
     &     status='UNKNOWN')

c---define timesteps

      dW1 = delta_collapse(zsample(iz1)) - Whost
      dW2 = delta_collapse(zsample(iz2)) - Whost
      dW3 = delta_collapse(zsample(iz3)) - Whost
      dW4 = delta_collapse(zsample(iz4)) - Whost
      dW5 = delta_collapse(zsample(iz5)) - Whost
      dW6 = delta_collapse(zsample(iz6)) - Whost

c---normalize histograms & compute predictions
c   Histogram contains average number of progenitors of given mass
c   per host, per logarithmic bin in (Mp/M0)

      DO i=1,50
        x = -(binsize * FLOAT(i-1)) - (binsize/2.0)
        xx= 10.0**x
        DO j=1,6
          temp(i,j) = histo(i,j)/FLOAT(Nt)/binsize
        END DO
        M = xx * Mhalo
        S1 = (variance(M))**2
        dS =  S1 - Shost
        fac = (S1/xx) * ABS(dlnSdlnM(M)) * ALOG(10.0)
        y1 = ALOG10(MAX(1.0E-20,fac * fEPS_SC(dS,dW1)))
        y2 = ALOG10(MAX(1.0E-20,fac * fEPS_SC(dS,dW2)))
        y3 = ALOG10(MAX(1.0E-20,fac * fEPS_SC(dS,dW3)))
        y4 = ALOG10(MAX(1.0E-20,fac * fEPS_SC(dS,dW4)))
        y5 = ALOG10(MAX(1.0E-20,fac * fEPS_SC(dS,dW5)))
        y6 = ALOG10(MAX(1.0E-20,fac * fEPS_SC(dS,dW6)))
        u1 = ALOG10(MAX(1.0E-20,fac * fEPS_EC(dS,dW1)))
        u2 = ALOG10(MAX(1.0E-20,fac * fEPS_EC(dS,dW2)))
        u3 = ALOG10(MAX(1.0E-20,fac * fEPS_EC(dS,dW3)))
        u4 = ALOG10(MAX(1.0E-20,fac * fEPS_EC(dS,dW4)))
        u5 = ALOG10(MAX(1.0E-20,fac * fEPS_EC(dS,dW5)))
        u6 = ALOG10(MAX(1.0E-20,fac * fEPS_EC(dS,dW6)))
        WRITE(10,82)x,(temp(i,j),j=1,6),y1,y2,y3,y4,y5,y6,
     &              u1,u2,u3,u4,u5,u6 
      END DO

      WRITE(*,*)'>>>>>> HISTOGRAMS WRITTEN TO FILE <<<<<<'

      CLOSE(10)

c---

 82   FORMAT(F7.4,2X,18(E12.5,1X))

      RETURN 
      END

c**********************************************************************

      SUBROUTINE output_MAHs(W0,Nt)
c---------------------------------------------------------------------------
c
c  output the mass accretion histories and formation times
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'
 
      INTEGER  i,j,iz,Nt,n50,n16,n84
      REAL     W0,Wstep,z,t
      REAL     aver1,sigma1,aver3,sigma3,tmp1(Ntree)
      REAL     aver2,sigma2,aver4,sigma4,tmp2(Ntree)
      CHARACTER outfil1*30,outfil2*30,outfil3*30,outfil4*30
      CHARACTER outfil5*30
        
      INTEGER  lblnk
      EXTERNAL lblnk

c---

      outfil1='/zform.dat'
      outfil2='/averMAH.dat'
      outfil5='/averVAH.dat'
      outfil3='/tree_prop_massfrac.dat'
      outfil4='/tree_prop_Nprog.dat'

      OPEN(10,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN')
      OPEN(20,file=moddir(1:lblnk(moddir))//outfil2(1:lblnk(outfil2)),
     &     status='UNKNOWN')
      OPEN(21,file=moddir(1:lblnk(moddir))//outfil5(1:lblnk(outfil5)),
     &     status='UNKNOWN')
      OPEN(30,file=moddir(1:lblnk(moddir))//outfil3(1:lblnk(outfil3)),
     &     status='UNKNOWN')
      OPEN(40,file=moddir(1:lblnk(moddir))//outfil4(1:lblnk(outfil4)),
     &     status='UNKNOWN')

c---for each tree, output halo formation times, mass fractions, and number 
c   of progenitors (above different mass cuts and for different orders)

      DO i=1,Nt
        WRITE(10,71)i,Nt,Mhalo,zform(i,1),zform(i,2),zform(i,3),conc(i)
        WRITE(30,73)i,Nt,Mhalo,(tree_prop(i,j,0),j=1,4),
     &    (tree_prop(i,j,1),j=1,4),(tree_prop(i,j,2),j=1,4)
        WRITE(40,74)i,Nt,Mhalo,(itree_prop(i,j,0),j=1,4),
     &    (itree_prop(i,j,1),j=1,4),(itree_prop(i,j,2),j=1,4)
      END DO

c---output average MAHs

      n50 = NINT(0.50 * FLOAT(Nt))
      n16 = NINT(0.16 * FLOAT(Nt))
      n84 = NINT(0.84 * FLOAT(Nt))
      
      DO iz = 1,Nzsample

        z = zsample(iz)
        t = tsample(iz)

        aver1 = 0.0
        aver2 = 0.0
        aver3 = 0.0
        aver4 = 0.0
        DO i=1,Nt
          aver1 = aver1 + 10.0**mah(i,iz,1)
          aver2 = aver2 + 10.0**mah(i,iz,2)
          aver3 = aver3 + mah(i,iz,1)
          aver4 = aver4 + mah(i,iz,2)
        END DO
        aver1 = aver1/FLOAT(Nt)
        aver2 = aver2/FLOAT(Nt)
        aver3 = aver3/FLOAT(Nt)
        aver4 = aver4/FLOAT(Nt)

        sigma1 = 0.0
        sigma2 = 0.0
        sigma3 = 0.0
        sigma4 = 0.0
        DO i=1,Nt
          sigma1 = sigma1 + (10.0**mah(i,iz,1)-aver1)**2
          sigma2 = sigma2 + (10.0**mah(i,iz,2)-aver2)**2
          sigma3 = sigma3 + (mah(i,iz,1)-aver3)**2
          sigma4 = sigma4 + (mah(i,iz,2)-aver4)**2
        END DO
        sigma1 = SQRT(sigma1/FLOAT(Nt))
        sigma2 = SQRT(sigma2/FLOAT(Nt))
        sigma3 = SQRT(sigma3/FLOAT(Nt))
        sigma4 = SQRT(sigma4/FLOAT(Nt))

        DO i=1,Ntree
          IF (i.GT.Nt) THEN
            tmp1(i) = 20.0
            tmp2(i) = 20.0
          ELSE
            tmp1(i) = -mah(i,iz,1)
            tmp2(i) = -mah(i,iz,2)
          END IF
        END DO
        CALL sort(Ntree,tmp1)
        CALL sort(Ntree,tmp2)
 
        WRITE(20,72)iz,z,t,ALOG10(aver1),sigma1,aver3,sigma3, 
     &     -tmp1(n16),-tmp1(n50),-tmp1(n84),
     &      model_mah(iz,1),model_mah(iz,2),model_mah(iz,3),Nt        
        WRITE(21,75)iz,z,t,ALOG10(aver2),sigma2,aver4,sigma4, 
     &     -tmp2(n16),-tmp2(n50),-tmp2(n84)

      END DO

c---

      CLOSE(10)
      CLOSE(20)
      CLOSE(21)
      CLOSE(30)
      CLOSE(40)
      
c---

 71   FORMAT(I5,2X,I5,2X,E12.5,2X,F7.4,2X,F7.4,2X,F7.4,2X,F8.3)
 72   FORMAT(I5,2X,F9.4,2X,F7.4,2X,10(E12.5,2X),I5)
 73   FORMAT(I5,2X,I5,2X,E12.5,2X,12(E12.5,1X))
 74   FORMAT(I5,2X,I5,2X,E12.5,2X,12(I8,1X))
 75   FORMAT(I5,2X,F9.4,2X,F7.4,2X,7(E12.5,2X))

      RETURN
      END

c**********************************************************************

      SUBROUTINE output_SHMFs(Nt)
c---------------------------------------------------------------------------
c
c  output the mass accretion histories and formation times
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'
 
      INTEGER  i,j,ihalo,n,Nt
      REAL     x,xv,z,ave,adev,sdev,var,s02,s16,s50,s84,s98
      REAL     aveN,adevN,sdevN,varN,aveD,aveND
      REAL     tmp1(Nt),tmp2(Nt),tmp3(Nt),tmp4(Nt),tmp5(Nt)
      REAL     tmp6(Nt),tmp7(Nt),tmp8(Nt),tmp9(Nt),tmp10(Nt)
      REAL     tmp11(Nt),tmp12(Nt),tmp13(Nt),tmp14(Nt)
      REAL     tmp15(Nt),tmp16(Nt),tmp17(Nt),tmp18(Nt)
      REAL     tmp19(Nt),tmp20(Nt),tmp21(Nt)
      REAL     zacc_tmp1(50,6),zacc_tmp2(50,6)
      REAL     histo_tmp1(50,5),histo_tmp2(50,5)
      REAL     histo_tmp3(50,5),histo_tmp4(50,5)
      REAL     histo_tmp5(50,5),histo_tmp6(50,5)  
      REAL     histo_tmp7(50,5)
      CHARACTER outfil1*30,outfil2*30,outfil3*30
      CHARACTER outfil4*30,outfil5*30

      INTEGER  lblnk
      EXTERNAL lblnk

c---

c---open the various output file

      outfil1='/histo_SHMF_un.dat'
      outfil2='/histo_SHMF_ev.dat'
      outfil3='/histo_SHMF_va.dat'
      outfil4='/histo_SHMF_vv.dat'
      outfil5='/histo_SHMF_vm.dat'
      OPEN(20,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN')
      OPEN(21,file=moddir(1:lblnk(moddir))//outfil2(1:lblnk(outfil2)),
     &     status='UNKNOWN')
      OPEN(22,file=moddir(1:lblnk(moddir))//outfil3(1:lblnk(outfil3)),
     &     status='UNKNOWN')
      OPEN(23,file=moddir(1:lblnk(moddir))//outfil4(1:lblnk(outfil4)),
     &     status='UNKNOWN')
      OPEN(24,file=moddir(1:lblnk(moddir))//outfil5(1:lblnk(outfil5)),
     &     status='UNKNOWN')

      outfil1='/cumul_SHMF_un.dat'
      outfil2='/cumul_SHMF_ev.dat'
      outfil3='/cumul_SHMF_va.dat'
      outfil4='/cumul_SHMF_vv.dat'
      outfil5='/cumul_SHMF_vm.dat'
      OPEN(40,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN')
      OPEN(41,file=moddir(1:lblnk(moddir))//outfil2(1:lblnk(outfil2)),
     &     status='UNKNOWN')
      OPEN(42,file=moddir(1:lblnk(moddir))//outfil3(1:lblnk(outfil3)),
     &     status='UNKNOWN')
      OPEN(43,file=moddir(1:lblnk(moddir))//outfil4(1:lblnk(outfil4)),
     &     status='UNKNOWN')
      OPEN(44,file=moddir(1:lblnk(moddir))//outfil5(1:lblnk(outfil5)),
     &     status='UNKNOWN')

      outfil1='/cumul_massfrac_ev.dat'
      OPEN(50,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN')

      outfil1='/zacc_SHMFs1.dat'
      outfil2='/zacc_SHMFs2.dat'
      OPEN(30,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN')
      OPEN(31,file=moddir(1:lblnk(moddir))//outfil2(1:lblnk(outfil2)),
     &     status='UNKNOWN')

      outfil1='/massloss_all.dat'
      outfil2='/massloss_flimit.dat'
      OPEN(60,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN')
      OPEN(61,file=moddir(1:lblnk(moddir))//outfil2(1:lblnk(outfil2)),
     &     status='UNKNOWN')

c---write histograms to file

      DO j=1,50
        x = -( binsize      * FLOAT(j-1)) - (binsize/2.0)
        xv= -((binsize/3.0) * FLOAT(j-1)) - (binsize/6.0)
        z = 0.05 * FLOAT(j) - 0.025
        DO n=1,5
          histo_tmp1(j,n) = histo_sub(j,n,1)/FLOAT(Nt)/binsize
          histo_tmp2(j,n) = histo_sub(j,n,2)/FLOAT(Nt)/binsize
          histo_tmp3(j,n) = histo_sub(j,n,3)/FLOAT(Nt)/binsize
          histo_tmp4(j,n) = histo_sub(j,n,4)/FLOAT(Nt)/binsize
          histo_tmp5(j,n) = histo_sub(j,n,5)/FLOAT(Nt)/(binsize/3.0)
          histo_tmp6(j,n) = histo_sub(j,n,6)/FLOAT(Nt)/(binsize/3.0)
          histo_tmp7(j,n) = histo_sub(j,n,7)/FLOAT(Nt)/(binsize/3.0)
          zacc_tmp1(j,n) = zacc_sub(j,n,1)
          zacc_tmp2(j,n) = zacc_sub(j,n,2)
        END DO
        WRITE(20,61)x,(histo_tmp1(j,n),n=1,5),Nt
        WRITE(21,61)x,(histo_tmp2(j,n),n=1,5),Nt
        WRITE(22,61)xv,(histo_tmp5(j,n),n=1,5),Nt
        WRITE(23,61)xv,(histo_tmp6(j,n),n=1,5),Nt
        WRITE(24,61)xv,(histo_tmp7(j,n),n=1,5),Nt
        WRITE(30,61)z,(zacc_tmp1(j,n),n=1,5),Nt
        WRITE(31,61)z,(zacc_tmp2(j,n),n=1,5),Nt
        WRITE(60,61)x,(histo_tmp3(j,n),n=1,5),Nt
        WRITE(61,61)x,(histo_tmp4(j,n),n=1,5),Nt
      END DO

c---compute mean, stdev, and CLs of cumulative SHMFs. Also we compute
c   <N(N-1)>, which will allow us to test Poissonian nature of P(N|M)
c   We do this BOTH for CUMULATIVE N and for DIFFERENTIAL N

      DO j=0,5,1
        DO i=1,50

          x = -( binsize      * FLOAT(i-1)) - (binsize/2.0)
          xv= -((binsize/3.0) * FLOAT(i-1)) - (binsize/6.0)
           
          DO ihalo=1,Nt
            tmp1(ihalo) = histo_cum(ihalo,i,j,1)
            tmp2(ihalo) = histo_cum(ihalo,i,j,2)
            tmp10(ihalo)= histo_cum(ihalo,i,j,4)
            tmp14(ihalo)= histo_cum(ihalo,i,j,5) 
            tmp18(ihalo)= histo_cum(ihalo,i,j,6) 
            tmp3(ihalo) = tmp1(ihalo) * (tmp1(ihalo) -1.0)
            tmp4(ihalo) = tmp2(ihalo) * (tmp2(ihalo) -1.0)
            tmp11(ihalo)= tmp10(ihalo)* (tmp10(ihalo)-1.0)
            tmp15(ihalo)= tmp14(ihalo)* (tmp14(ihalo)-1.0)
            tmp19(ihalo)= tmp18(ihalo)* (tmp18(ihalo)-1.0)
            IF (i.EQ.50) THEN
             tmp5(ihalo) = histo_cum(ihalo,i,j,1)
             tmp6(ihalo) = histo_cum(ihalo,i,j,2)
             tmp12(ihalo)= histo_cum(ihalo,i,j,4)
             tmp16(ihalo)= histo_cum(ihalo,i,j,5)
             tmp20(ihalo)= histo_cum(ihalo,i,j,6)
            ELSE
             tmp5(ihalo) = histo_cum(ihalo,i,j,1) - 
     &                     histo_cum(ihalo,i+1,j,1)
             tmp6(ihalo) = histo_cum(ihalo,i,j,2) - 
     &                     histo_cum(ihalo,i+1,j,2)
             tmp12(ihalo)= histo_cum(ihalo,i,j,4) - 
     &                     histo_cum(ihalo,i+1,j,4)
             tmp16(ihalo)= histo_cum(ihalo,i,j,5) - 
     &                     histo_cum(ihalo,i+1,j,5)
             tmp20(ihalo)= histo_cum(ihalo,i,j,6) - 
     &                     histo_cum(ihalo,i+1,j,6)
            END IF
            tmp7(ihalo) = tmp5(ihalo) * (tmp5(ihalo) -1.0)
            tmp8(ihalo) = tmp6(ihalo) * (tmp6(ihalo) -1.0)
            tmp13(ihalo)= tmp12(ihalo)* (tmp12(ihalo)-1.0)
            tmp17(ihalo)= tmp16(ihalo)* (tmp16(ihalo)-1.0)
            tmp21(ihalo)= tmp20(ihalo)* (tmp20(ihalo)-1.0)
            tmp9(ihalo) = histo_cum(ihalo,i,j,3)/Mhalo
          END DO

          CALL moment(tmp1,Nt,ave,adev,sdev,var)
          CALL moment(tmp3,Nt,aveN,adevN,sdevN,varN)
          CALL moment(tmp5,Nt,aveD,adevN,sdevN,varN)
          CALL moment(tmp7,Nt,aveND,adevN,sdevN,varN)
          CALL conf_levels(tmp1,Nt,s02,s16,s50,s84,s98)
          WRITE(40,60)j,i,x,ave,sdev,s02,s16,s50,s84,s98,
     &     aveN,aveD,aveND

          CALL moment(tmp2,Nt,ave,adev,sdev,var)
          CALL moment(tmp4,Nt,aveN,adevN,sdevN,varN)
          CALL moment(tmp6,Nt,aveD,adevN,sdevN,varN)
          CALL moment(tmp8,Nt,aveND,adevN,sdevN,varN)
          CALL conf_levels(tmp2,Nt,s02,s16,s50,s84,s98)
          WRITE(41,60)j,i,x,ave,sdev,s02,s16,s50,s84,s98,
     &     aveN,aveD,aveND

          CALL moment(tmp10,Nt,ave,adev,sdev,var)
          CALL moment(tmp11,Nt,aveN,adevN,sdevN,varN)
          CALL moment(tmp12,Nt,aveD,adevN,sdevN,varN)
          CALL moment(tmp13,Nt,aveND,adevN,sdevN,varN)
          CALL conf_levels(tmp10,Nt,s02,s16,s50,s84,s98)
          WRITE(42,60)j,i,xv,ave,sdev,s02,s16,s50,s84,s98,
     &     aveN,aveD,aveND

          CALL moment(tmp14,Nt,ave,adev,sdev,var)
          CALL moment(tmp15,Nt,aveN,adevN,sdevN,varN)
          CALL moment(tmp16,Nt,aveD,adevN,sdevN,varN)
          CALL moment(tmp17,Nt,aveND,adevN,sdevN,varN)
          CALL conf_levels(tmp14,Nt,s02,s16,s50,s84,s98)
          WRITE(43,60)j,i,xv,ave,sdev,s02,s16,s50,s84,s98,
     &     aveN,aveD,aveND

          CALL moment(tmp18,Nt,ave,adev,sdev,var)
          CALL moment(tmp19,Nt,aveN,adevN,sdevN,varN)
          CALL moment(tmp20,Nt,aveD,adevN,sdevN,varN)
          CALL moment(tmp21,Nt,aveND,adevN,sdevN,varN)
          CALL conf_levels(tmp18,Nt,s02,s16,s50,s84,s98)
          WRITE(44,60)j,i,xv,ave,sdev,s02,s16,s50,s84,s98,
     &     aveN,aveD,aveND

          CALL moment(tmp9,Nt,ave,adev,sdev,var)
          CALL conf_levels(tmp9,Nt,s02,s16,s50,s84,s98)
          WRITE(50,60)j,i,x,ave,sdev,s02,s16,s50,s84,s98

        END DO
      END DO

      CLOSE(20)
      CLOSE(21)
      CLOSE(22)
      CLOSE(23)
      CLOSE(24)

      CLOSE(30)
      CLOSE(31)

      CLOSE(40)
      CLOSE(41)
      CLOSE(42)
      CLOSE(43)
      CLOSE(44)

      CLOSE(50)

      CLOSE(60)
      CLOSE(61)

c---

 60   FORMAT(I2,2X,I3,2X,F7.4,2X,10(E12.5,1X))
 61   FORMAT(F7.4,2X,5(E12.5,1X),I5)

      RETURN
      END 

c**********************************************************************

      SUBROUTINE output_subhalos()
c---------------------------------------------------------------------------
c
c  output the mass accretion histories and formation times
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER   k,n
      CHARACTER outfil1*30,outfil2*30,outfil3*30,outfil4*30

      INTEGER  lblnk
      EXTERNAL lblnk

c---

c---open the various output file

      outfil1='/subhalos_un.dat'
      outfil2='/subhalos_ev.dat'
      outfil3='/subhalos_order_un.dat'
      outfil4='/subhalos_order_ev.dat'

      OPEN(20,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN')
      OPEN(21,file=moddir(1:lblnk(moddir))//outfil2(1:lblnk(outfil2)),
     &     status='UNKNOWN')
      OPEN(30,file=moddir(1:lblnk(moddir))//outfil3(1:lblnk(outfil3)),
     &     status='UNKNOWN')
      OPEN(31,file=moddir(1:lblnk(moddir))//outfil4(1:lblnk(outfil4)),
     &     status='UNKNOWN')

      DO k=1,Ntree
        WRITE(20,41)k,(Msubhalos(k,n,1),n=1,20)
        WRITE(21,41)k,(Msubhalos(k,n,2),n=1,20)
        WRITE(30,42)k,(isubhalos(k,n,1),n=1,20)
        WRITE(31,42)k,(isubhalos(k,n,2),n=1,20)
      END DO
     
      CLOSE(20)
      CLOSE(21)
      CLOSE(30)
      CLOSE(31)

 41   FORMAT(I6,2X,20(E12.5,1X))
 42   FORMAT(I6,2X,20(I6,1X))

      RETURN
      END

c**********************************************************************

      SUBROUTINE output_tree(k,Nx,Ny)
c---------------------------------------------------------------------------
c
c  output the Mass merger trees in magic form
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER   i,j,k,k1,k2,k3,k4,Nx,Ny

      CHARACTER outfil1*60,id1,id2,id3,id4

      INTEGER   lblnk
      EXTERNAL  lblnk

c---

      k1 = INT(k/1000)
      k2 = INT(MOD(k,1000)/100)
      k3 = INT(MOD(k,100)/10)
      k4 = MOD(k,10)

      id1=CHAR(k1+ICHAR(null))
      id2=CHAR(k2+ICHAR(null))
      id3=CHAR(k3+ICHAR(null))
      id4=CHAR(k4+ICHAR(null))

      outfil1='/MergerTree'//id1//id2//id3//id4//'.dat'

      OPEN(44,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN',form='UNFORMATTED')

      WRITE(44)Nx,Ny
      WRITE(44)((mergertree(i,j),j=1,Ny),i=1,Nx)

      CLOSE(44)

      RETURN
      END

c**********************************************************************

      SUBROUTINE writeMAH(k)
c---------------------------------------------------------------------------
c
c  output the mass accretion histories and formation times
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER   iz,k,k1,k2,k3,k4

      CHARACTER outfil1*60,id1,id2,id3,id4

      INTEGER   lblnk
      EXTERNAL  lblnk

c---

      k1 = INT(k/1000)
      k2 = INT(MOD(k,1000)/100)
      k3 = INT(MOD(k,100)/10)
      k4 = MOD(k,10)

      id1=CHAR(k1+ICHAR(null))
      id2=CHAR(k2+ICHAR(null))
      id3=CHAR(k3+ICHAR(null))
      id4=CHAR(k4+ICHAR(null))

      outfil1='/MAH'//id1//id2//id3//id4//'.dat'

      OPEN(44,file=moddir(1:lblnk(moddir))//outfil1(1:lblnk(outfil1)),
     &     status='UNKNOWN')

      DO iz=1,Nzsample
        WRITE(44,*)iz,zsample(iz),tsample(iz),mah(k,iz,1),mah(k,iz,2)
      END DO
      
      CLOSE(44)

      RETURN
      END

c**********************************************************************

      SUBROUTINE write_threshold_mass
c---------------------------------------------------------------------------
c
c  output the mass accretion histories and formation times
c
c---------------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER  iz
      REAL     z
    
c---

      OPEN(10,file='threshold.dat',status='UNKNOWN')

      DO iz=1,200
        z = FLOAT(iz-1)/199.0 * 20.0
        CALL get_minimum_mass(z)
        WRITE(10,*)iz,z,ALOG10(Mmin)
      END DO

      CLOSE(10)
    
      RETURN 
      END 
      
c**********************************************************************


