#!/usr/bin/python3

import numpy as np
from sklearn.model_selection import train_test_split
import random
import time
from scipy import interpolate
import importlib
import logging
importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

# for acceleration
# Ref: http://numba.pydata.org
from numba import jit


                    
"""
Create_Pseudodata
"""
class Create_Pseudodata:
    def __init__(self, parameter, data, chi, LOW = 10 , HIGH = 30 , number = 100, index=0):
        self.length = len(data)
        self.data = data
        self.parameter = parameter
        self.chi = chi
        self.LOW = LOW
        self.HIGH = HIGH
        self.number = number
        
        self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        self.index = index
        
    def Create_Pseudodata(self):
        data = self.data
        parameter = self.parameter
        chi = self.chi
        LOW = self.LOW
        HIGH = self.HIGH
        number = self.number
        index = self.index
        
        def interpolate_chisq(P,F,S):
            return round(np.sum(((P-np.array(F))**2)/(np.array(S)**2)), 10)
        
        logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        """
        Load Experimental Data
        """
        logging.info("Experimental data are loading.")
        logging.info("=====START=====")
        ####################################################################################
        exp_data_path = "../Data/Exp_Data/"

        LiAMS, LiV = np.load(exp_data_path + "Li_AMS2.npy"),np.load(exp_data_path + "Li_Voyager.npy")
        Li_Eams, Li_Fams, Li_Sams = LiAMS[0], LiAMS[1], LiAMS[2]
        Li_Evy, Li_Fvy, Li_Svy = LiV[0], LiV[1], LiV[2]

        BeAMS, BeV = np.load(exp_data_path + "Be_AMS2.npy"), np.load(exp_data_path + "Be_Voyager.npy")
        Be_Eams, Be_Fams, Be_Sams = BeAMS[0], BeAMS[1], BeAMS[2]
        Be_Evy, Be_Fvy, Be_Svy = BeV[0], BeV[1], BeV[2]

        BAMS, BV, BA = np.load(exp_data_path + "B_AMS2.npy"), np.load(exp_data_path + "B_Voyager.npy"), np.load(exp_data_path + "B_ACE.npy")
        B_Eams, B_Fams, B_Sams = BAMS[0], BAMS[1], BAMS[2]
        B_Evy, B_Fvy, B_Svy = BV[0], BV[1], BV[2]
        B_Eace,B_Face,B_Sace = BA[0], BA[1], BA[2]

        CAMS, CV, CA = np.load(exp_data_path + "C_AMS2.npy"), np.load(exp_data_path + "C_Voyager.npy"), np.load(exp_data_path + "C_ACE.npy")
        C_Eams, C_Fams, C_Sams = CAMS[0], CAMS[1], CAMS[2]
        C_Evy, C_Fvy, C_Svy = CV[0], CV[1], CV[2]
        C_Eace,C_Face,C_Sace = CA[0], CA[1], CA[2]

        OAMS, OV, OA = np.load(exp_data_path + "O_AMS2.npy"), np.load(exp_data_path + "O_Voyager.npy"), np.load(exp_data_path + "O_ACE.npy")
        O_Eams, O_Fams, O_Sams = OAMS[0], OAMS[1], OAMS[2]
        O_Evy, O_Fvy, O_Svy = OV[0], OV[1], OV[2]
        O_Eace,O_Face,O_Sace = OA[0], OA[1], OA[2]
        logging.info("=====Finish=====")
        ####################################################################################

        '''
        Using whitening data to make pseudodata
        '''
        ####################################################################################
        total_data_divAp = np.zeros((data.shape[0], 84, 8))
        for i in range(data.shape[0]):
            total_data_divAp[i,:,0] = (data[i,:,1]/parameter[i,11])/(data[i,:,4]/data[i,52,4]) # (Li/N_Li)/(C/C_109.5)
            total_data_divAp[i,:,1] = (data[i,:,2]/parameter[i,12])/(data[i,:,4]/data[i,52,4]) # (Be/N_Be)/(C/C_109.5)
            total_data_divAp[i,:,2] = data[i,:,3]/(data[i,:,4]/data[i,52,4]) # B/(C/C_109.5)

            total_data_divAp[i,:,3] = (data[i,:,1]/parameter[i,11])/(data[i,:,5]/parameter[i,13]) # (Li/N_Li)/(O/N_O)
            total_data_divAp[i,:,4] = (data[i,:,2]/parameter[i,12])/(data[i,:,5]/parameter[i,13]) # (Be/N_Be)/(O/N_O)
            total_data_divAp[i,:,5] = data[i,:,3]/(data[i,:,5]/parameter[i,13]) # B/(O/N_O)

            total_data_divAp[i,:,6] = data[i,:,4] # C
            total_data_divAp[i,:,7] = (data[i,:,5]/parameter[i,13])/(data[i,:,4]/data[i,52,4])  # (O/N_O)/(C/C_109.5)
        ####################################################################################
        ticks_Search_The_Pack_1 = time.time()
        '''
        Search The Pack of Spectrum
        '''
        divAp_pack = np.zeros((2, 84, 8))
        for i in range(8):
            for j in range(84):
                divAp_pack[0,j,i] = max(total_data_divAp[:,j,i])
                divAp_pack[1,j,i] = min(total_data_divAp[:,j,i])
        
        ticks_Search_The_Pack_2 = time.time()
        logging.info("\033[3;33mTime for Search The Pack of Spectrum : {:.4f} min\033[0;m".format((ticks_Search_The_Pack_2-ticks_Search_The_Pack_1)/60.))
        
        
        '''
        Create Pseudodata
        '''
        ticks_Pseudodata_1 = time.time()
        
        spec_data_LiC = np.zeros((number,84))
        spec_data_BeC = np.zeros((number,84))
        spec_data_BC = np.zeros((number,84))
        spec_data_LiO = np.zeros((number,84))
        spec_data_BeO = np.zeros((number,84))
        spec_data_BO = np.zeros((number,84))
        spec_data_C = np.zeros((number,84))
        spec_data_OC = np.zeros((number,84))


        for i in range(number):
#             rand = np.random.randint(data.shape[0])
            rand = i

            data_delta = np.zeros((84, 8))
            delta = float(np.random.uniform(LOW, HIGH, size=None)) #15, 30
            for a in range(8):
                for b in range(84):
                    data_delta[b,a] = (max(total_data_divAp[:,b,a]) - min(total_data_divAp[:,b,a]))/delta

            j = 0
            while j != 84:
                '''
                pack
                '''
    #             spec_data_LiC[i,j] = np.random.uniform(divAp_pack[1,j,0], divAp_pack[0,j,0], size=None)
    #             spec_data_BeC[i,j] = np.random.uniform(divAp_pack[1,j,1], divAp_pack[0,j,1], size=None)
    #             spec_data_BC[i,j]  = np.random.uniform(divAp_pack[1,j,2], divAp_pack[0,j,2], size=None)
    #             spec_data_C[i,j]  = np.random.uniform(divAp_pack[1,j,6], divAp_pack[0,j,6], size=None)
    #             spec_data_OC[i,j]  = np.random.uniform(divAp_pack[1,j,7], divAp_pack[0,j,7], size=None)

                '''
                arround one line + Delta_data
                '''
                spec_data_LiC[i,j] = np.random.uniform(total_data_divAp[rand,j,0]-data_delta[j,0], total_data_divAp[rand,j,0]+data_delta[j,0], size=None)
                spec_data_BeC[i,j] = np.random.uniform(total_data_divAp[rand,j,1]-data_delta[j,1], total_data_divAp[rand,j,1]+data_delta[j,1], size=None)
                spec_data_BC[i,j]  = np.random.uniform(total_data_divAp[rand,j,2]-data_delta[j,2], total_data_divAp[rand,j,2]+data_delta[j,2], size=None)
                spec_data_C[i,j]  = np.random.uniform(total_data_divAp[rand,j,6]-data_delta[j,6], total_data_divAp[rand,j,6]+data_delta[j,6], size=None)
                spec_data_OC[i,j]  = np.random.uniform(total_data_divAp[rand,j,7]-data_delta[j,7], total_data_divAp[rand,j,7]+data_delta[j,7], size=None)

    #             spec_data_LiC[i,j] = np.random.uniform(total_data_divAp[rand,j,0]*0.95, total_data_divAp[rand,j,0]*1.05, size=None)
    #             spec_data_BeC[i,j] = np.random.uniform(total_data_divAp[rand,j,1]*0.95, total_data_divAp[rand,j,1]*1.05, size=None)
    #             spec_data_BC[i,j]  = np.random.uniform(total_data_divAp[rand,j,2]*0.95, total_data_divAp[rand,j,2]*1.05, size=None)
    #             spec_data_C[i,j]  = np.random.uniform(total_data_divAp[rand,j,6]*0.95, total_data_divAp[rand,j,6]*1.05, size=None)
    #             spec_data_OC[i,j]  = np.random.uniform(total_data_divAp[rand,j,7]*0.95, total_data_divAp[rand,j,7]*1.05, size=None)

                '''
                arround one line 
                '''
    #             spec_data_LiC[i,j] = np.random.uniform(total_data_divAp[rand,j,0], total_data_divAp[rand,j,0], size=None)
    #             spec_data_BeC[i,j] = np.random.uniform(total_data_divAp[rand,j,1], total_data_divAp[rand,j,1], size=None)
    #             spec_data_BC[i,j]  = np.random.uniform(total_data_divAp[rand,j,2], total_data_divAp[rand,j,2], size=None)
    #             spec_data_C[i,j]  = np.random.uniform(total_data_divAp[rand,j,6], total_data_divAp[rand,j,6], size=None)
    #             spec_data_OC[i,j]  = np.random.uniform(total_data_divAp[rand,j,7], total_data_divAp[rand,j,7], size=None)


    #             if spec_data_LiC[i,j] < divAp_pack[1,j,0] or spec_data_LiC[i,j] > divAp_pack[0,j,0]:
    #                 j -= 1
    #             elif spec_data_BeC[i,j] < divAp_pack[1,j,1] or spec_data_BeC[i,j] > divAp_pack[0,j,1]:
    #                 j -= 1
    #             elif spec_data_BC[i,j] < divAp_pack[1,j,2] or spec_data_BC[i,j] > divAp_pack[0,j,2]:
    #                 j -= 1
    #             elif spec_data_C[i,j] < divAp_pack[1,j,6] or spec_data_C[i,j] > divAp_pack[0,j,6]:
    #                 j -= 1
    #             elif spec_data_OC[i,j] < divAp_pack[1,j,7] or spec_data_OC[i,j] > divAp_pack[0,j,7]:
    #                 j -= 1

                j += 1


    #     Li_norm = np.random.uniform(0.8,1.2,100)
    #     Be_norm = np.random.uniform(0.8,1.2,100)
    #     O_norm = np.random.uniform(0.8,1.2,100)

        ticks_Pseudodata_2 = time.time()
        logging.info("\033[3;33mTime for Create Pseudodata : {:.4f} min\033[0;m".format((ticks_Pseudodata_2-ticks_Pseudodata_1)/60.))
        
        
        ticks_N_1 = time.time()

        Li_norm = np.random.uniform(min(parameter[:,11]),max(parameter[:,11]),100)
        Be_norm = np.random.uniform(min(parameter[:,12]),max(parameter[:,12]),100)
        O_norm = np.random.uniform(min(parameter[:,13]),max(parameter[:,13]),100)

        '''
        Recorver to (Spectrum)/(Normal Factor)
        '''
        spectrum_E = np.zeros((number,84))
        for i in range(number):
            spectrum_E[i,:] = data[0,:,0]
        spectrum_Li = spec_data_LiC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))
        spectrum_Be = spec_data_BeC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))
        spectrum_B = spec_data_BC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))
        spectrum_C = spec_data_C[:,:]
        spectrum_O = spec_data_OC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))


        '''
        Determine the Normal Factor for each Pseudodata
        '''
        norm_factor = np.zeros((number,14))
        for i in range(number):
            chisq_scan_Li = np.zeros((100))
            chisq_scan_Be = np.zeros((100))
            chisq_scan_O = np.zeros((100))
            for j in range(100):
                pred_interpolate_Li = interpolate.interp1d((spectrum_E[0,:]), (spectrum_Li[i,:]*Li_norm[j]), kind='cubic')
                pred_interpolate_Be = interpolate.interp1d((spectrum_E[0,:]), (spectrum_Be[i,:]*Be_norm[j]), kind='cubic')
                pred_interpolate_O = interpolate.interp1d((spectrum_E[0,:]), (spectrum_O[i,:]*O_norm[j]), kind='cubic')

                chisq_scan_Li[j] = (interpolate_chisq(pred_interpolate_Li(Li_Eams),Li_Fams,Li_Sams)+
                                    interpolate_chisq(pred_interpolate_Li(Li_Evy),Li_Fvy,Li_Svy)
                                   )

                chisq_scan_Be[j] = (interpolate_chisq(pred_interpolate_Be(Be_Eams),Be_Fams,Be_Sams)+
                                    interpolate_chisq(pred_interpolate_Be(Be_Evy),Be_Fvy,Be_Svy)
                                   )

                chisq_scan_O[j] = (interpolate_chisq(pred_interpolate_O(O_Eams),O_Fams,O_Sams)+
                                   interpolate_chisq(pred_interpolate_O(O_Evy),O_Fvy,O_Svy)+
                                   interpolate_chisq(pred_interpolate_O(O_Eace),O_Face,O_Sace)
                                  )

            norm_factor[i,11] = Li_norm[np.where(chisq_scan_Li == min(chisq_scan_Li))]
            norm_factor[i,12] = Be_norm[np.where(chisq_scan_Be == min(chisq_scan_Be))]
            norm_factor[i,13] = O_norm[np.where(chisq_scan_O == min(chisq_scan_O))]


        logging.info("Pseudo Normal Factor")
        logging.info(" {:^4} {:^6} {:^6} {:^6}".format("","Li","Be","O"))
        logging.info(" {:^4} {:^6} {:^6} {:^6}".format("#",len(norm_factor[:,11]),len(norm_factor[:,12]),len(norm_factor[:,13])))
        logging.info(" {:^4} {:^4.4f} {:^4.4f} {:^4.4f}".format("Max",max(norm_factor[:,11]),max(norm_factor[:,12]),max(norm_factor[:,13])))
        logging.info(" {:^4} {:^4.4f} {:^4.4f} {:^4.4f}".format("Min",min(norm_factor[:,11]),min(norm_factor[:,12]),min(norm_factor[:,13])))
        logging.info(" {:^4} {:^4.4f} {:^4.4f} {:^4.4f}".format("Ave.",np.average(norm_factor[:,11]),np.average(norm_factor[:,12]),np.average(norm_factor[:,13])))
        
        
        ticks_N_2 = time.time()
        logging.info("\033[3;33mTime for N_Li : {:.4f} min\033[0;m".format((ticks_N_2-ticks_N_1)/60.))
        

        ticks_Recorver_Spectrum_1 = time.time()
        
        '''
        Recorver to Spectrum
        '''
        spectrum_E = np.zeros((number,84))
        for i in range(number):
            spectrum_E[i,:] = data[0,:,0] 
        spectrum_Li = spec_data_LiC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))*np.reshape(norm_factor[:,11],(number,1))
        spectrum_Be = spec_data_BeC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))*np.reshape(norm_factor[:,12],(number,1))
        spectrum_B = spec_data_BC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))
        spectrum_C = spec_data_C[:,:]
        spectrum_O = spec_data_OC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))*np.reshape(norm_factor[:,13],(number,1))

        '''
        Select 
        '''
        pseudodata_tmp = []
        norm_chisq = np.zeros((number))
        normalfactor = []
        for i in range(number):
            norm_interpolate_Li = interpolate.interp1d((spectrum_E[0,:]), (spectrum_Li[i,:]), kind='cubic')
            norm_interpolate_Be = interpolate.interp1d((spectrum_E[0,:]), (spectrum_Be[i,:]), kind='cubic')
            pred_interpolate_B  = interpolate.interp1d((spectrum_E[0,:]), (spectrum_B[i,:]), kind='cubic')
            pred_interpolate_C  = interpolate.interp1d((spectrum_E[0,:]), (spectrum_C[i,:]), kind='cubic')
            norm_interpolate_O  = interpolate.interp1d((spectrum_E[0,:]), (spectrum_O[i,:]), kind='cubic')
            norm_chisq[i] = (
                                interpolate_chisq(P=norm_interpolate_Li(Li_Eams), F=Li_Fams, S=Li_Sams)+
                                interpolate_chisq(P=norm_interpolate_Li(Li_Evy), F=Li_Fvy, S=Li_Svy)+

                                interpolate_chisq(P=norm_interpolate_Be(Be_Eams), F=Be_Fams, S=Be_Sams)+
                                interpolate_chisq(P=norm_interpolate_Be(Be_Evy), F=Be_Fvy, S=Be_Svy)+

                                interpolate_chisq(P=pred_interpolate_B(B_Eams), F=B_Fams, S=B_Sams)+
                                interpolate_chisq(P=pred_interpolate_B(B_Evy), F=B_Fvy, S=B_Svy)+
                                interpolate_chisq(P=pred_interpolate_B(B_Eace), F=B_Face, S=B_Sace)+

                                interpolate_chisq(P=pred_interpolate_C(C_Eams), F=C_Fams, S=C_Sams)+
                                interpolate_chisq(P=pred_interpolate_C(C_Evy), F=C_Fvy, S=C_Svy)+
                                interpolate_chisq(P=pred_interpolate_C(C_Eace), F=C_Face, S=C_Sace)+

                                interpolate_chisq(P=norm_interpolate_O(O_Eams), F=O_Fams, S=O_Sams)+
                                interpolate_chisq(P=norm_interpolate_O(O_Evy), F=O_Fvy, S=O_Svy)+
                                interpolate_chisq(P=norm_interpolate_O(O_Eace), F=O_Face, S=O_Sace)
                             )
            
            ticks_Recorver_Spectrum_2 = time.time()
            logging.info("\033[3;33mTime for Recorver_Spectrum : {:.4f} min\033[0;m".format((ticks_Recorver_Spectrum_2-ticks_Recorver_Spectrum_1)/60.))
        
            
            
            if index == 0:
    #             if norm_chisq[i] < min(chi)+15.94: # 1sigma
    #             if norm_chisq[i] < min(chi)+24.03: # 2sigma
    #             if norm_chisq[i] < min(chi)+33.20: # 3sigma
    #               if norm_chisq[i] < min(chi)+43.82: # 4sigma
#                 if norm_chisq[i] < min(chi)+70:  # 6 sigma
                if True:
                    pseudodata_tmp.append([data[0,:,0],spectrum_Li[i,:],
                                           spectrum_Be[i,:],spectrum_B[i,:],
                                           spectrum_C[i,:],spectrum_O[i,:]]
                                         )
                    normalfactor.append(norm_factor[i,:])



            if index == 1:
                pseudodata_tmp.append([data[0,:,0],spectrum_Li[i,:],
                                           spectrum_Be[i,:],spectrum_B[i,:],
                                           spectrum_C[i,:],spectrum_O[i,:]]
                                         )
                normalfactor.append(norm_factor[i,:])

    #         pseudodata_tmp.append([data[0,:,0],spectrum_Li[i,:],
    #                                            spectrum_Be[i,:],spectrum_B[i,:],
    #                                            spectrum_C[i,:],spectrum_O[i,:]]
    #                                          )
    #         normalfactor.append(norm_factor[i,:])

        normalfactor = np.array(normalfactor)     
        pseudodata = np.zeros((len(pseudodata_tmp),84,6))
        for i in range(6):
            pseudodata[:,:,i] = np.array(pseudodata_tmp)[:,i,:]


        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))
        return normalfactor, pseudodata
        
#     self.normalfactor = normalfactor
#     self.pseudodata = pseudodata
    
#=========================================================================================================#


