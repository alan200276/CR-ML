import numpy as np
from sklearn.model_selection import train_test_split
import random
import time
from scipy import interpolate

class Mock_Data_to_NumpyArray:
    def __init__(self, mock_data_file):
        import time
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        print("Now loading...")
        para_smoothly_raw = []
        data_smoothly_raw = []
        chisq = []
        with open(mock_data_file,"r") as fname1:
            for i,t in enumerate(fname1.readlines()):
                if i%86==0:
                    try:
                        para_smoothly_raw.append(np.array(t.split()[1:]).astype(np.float)) ### take away the word "Para:"\
                    except ValueError:
                        pass
                elif i%86==85:  ##this item is chisq
                    chisq.append(float(t)) 
                else:
                    try:
                        data_smoothly_raw.append(np.array(t.split()).astype(np.float)) 
                    except ValueError:
                        pass

        parameter = np.array(para_smoothly_raw)
        spectrum = np.array(data_smoothly_raw)
        chisq = np.array(chisq)
        print("Total data: ",parameter.shape[0])

        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        print("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))
        
        self.parameter, self.spectrum, self.chisq = parameter, spectrum, chisq

        
        
        
class Mock_Data_PreProcessing:
    def __init__(self, origin_parameter, new_parameter, spectrum, usedata = False):
        self.len = len(origin_parameter)

        self.origin_parameter = origin_parameter
        self.new_parameter = new_parameter
        self.origin_Ap = origin_parameter[:,5]
        self.new_Ap = new_parameter[:,5]
        self.origin_NLi, self.origin_NBe, self.origin_NO = origin_parameter[:,11], origin_parameter[:,12], origin_parameter[:,13]
        self.new_NLi, self.new_NBe, self.new_NO = new_parameter[:,11], new_parameter[:,12], new_parameter[:,13]
        
        if usedata == False:
            self.E = np.reshape(spectrum[:,0],(self.len,84))
            self.Li = np.reshape(spectrum[:,6],(self.len,84))
            self.Be = np.reshape(spectrum[:,7],(self.len,84))
            self.B = np.reshape(spectrum[:,8],(self.len,84))
            self.C = np.reshape(spectrum[:,9],(self.len,84))
            self.O = np.reshape(spectrum[:,10],(self.len,84))
        
        
        if usedata == True:
            self.E = spectrum[:,:,0]
            self.Li = spectrum[:,:,1]
            self.Be = spectrum[:,:,2]
            self.B = spectrum[:,:,3]
            self.C = spectrum[:,:,4]
            self.O = spectrum[:,:,5]
        
        self.newLi = self.Li/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))/np.reshape(self.origin_NLi,(self.len,1))*np.reshape(self.new_NLi,(self.len,1))
        self.newBe = self.Be/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))/np.reshape(self.origin_NBe,(self.len,1))*np.reshape(self.new_NBe,(self.len,1))
        self.newB = self.B/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))
        self.newC = self.C/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))
        self.newO = self.O/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))/np.reshape(self.origin_NO,(self.len,1))*np.reshape(self.new_NO,(self.len,1))
    
        data = np.zeros((self.len,84,6))
        spectrumlist = [self.E, self.newLi, self.newBe, self.newB, self.newC, self.newO]
        for i in range(6):
            data[:,:,i] = spectrumlist[i]

        self.data = data
        
        
class Mock_Data_Processing_for_Training:
    def __init__(self, parameter, E=0, Li=0, Be=0, B=0, C=0, O=0, data=0, usedata = False):
        self.len = len(parameter)
        self.E, self.Li, self.Be, self.B, self.C, self.O = E, Li, Be, B, C, O
        self.parameter = parameter
        self.NLi, self.NBe, self.NO = parameter[:,11], parameter[:,12], parameter[:,13]
        self.LiC, self.BeC, self.BC = 0, 0, 0
        self.LiO, self.BeO, self.BO = 0, 0, 0
        self.CC_110, self.OC = 0, 0
        self.input_train, self.input_test, self.source_train, self.source_test = 0, 0, 0, 0
        self.total_data_in_ratio = 0
        if usedata == True:
            self.len = len(data)
            self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        
            
    def spectrum_ratio(self):
        import time
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        """
        Whitening
        """
        total_data_in_ratio = np.zeros((self.len, 84, 8))
        for i in range(self.len):
            total_data_in_ratio[i,:,0] = (self.Li[i,:]/self.NLi[i])/(self.C[i,:]/self.C[i,52]) # Li/C
            total_data_in_ratio[i,:,1] = (self.Be[i,:]/self.NBe[i])/(self.C[i,:]/self.C[i,52]) # Be/C
            total_data_in_ratio[i,:,2] =  self.B[i,:]/(self.C[i,:]/self.C[i,52]) # B/C

            total_data_in_ratio[i,:,3] = (self.Li[i,:]/self.NLi[i])/(self.O[i,:]/self.NO[i]) # Li/O
            total_data_in_ratio[i,:,4] = (self.Be[i,:]/self.NBe[i])/(self.O[i,:]/self.NO[i]) # Be/O
            total_data_in_ratio[i,:,5] = self.B[i,:]/(self.O[i,:]/self.NO[i]) # B/O

            total_data_in_ratio[i,:,6] = self.C[i,:]/self.C[i,52] # C/C_109.5
            total_data_in_ratio[i,:,7] = (self.O[i,:]/self.NO[i])/self.C[i,52] # O/C_109.5
#             """for pseudo data"""
#             total_data_in_ratio[i,:,6] = self.C[i,:] # C/
#             total_data_in_ratio[i,:,7] = (self.O[i,:]/self.NO[i])/(self.C[i,:]/self.C[i,52]) # O/C
            
        self.LiC, self.BeC, self.BC = total_data_in_ratio[:,:,0], total_data_in_ratio[:,:,1], total_data_in_ratio[:,:,2]
        self.LiO, self.BeO, self.BO = total_data_in_ratio[:,:,3], total_data_in_ratio[:,:,4], total_data_in_ratio[:,:,5]
        self.CC_110, self.OC = total_data_in_ratio[:,:,6], total_data_in_ratio[:,:,7]
        self.total_data_in_ratio = total_data_in_ratio
        
        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        
    def Train_Test_split(self, splitrate = 0.1, split = True):
        from sklearn.model_selection import train_test_split
        import random
        import time
        def Apply_Norm(xin, xmin, sc_factor):
            xout=(xin[:]-xmin)/sc_factor
            return xout
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        print("\033[3;33m{}\033[0;m".format("Prepare Ratio"))
        self.spectrum_ratio()
        """     
        random split traning sample and test sample, 10% for test
        """
        if split == True:
            ran = random.randint(0,100000)
            train_raw, predict_raw, train_raw_para, predict_raw_para = train_test_split(self.total_data_in_ratio, self.parameter, test_size=splitrate,
                                                random_state = np.random.seed(ran), shuffle = True )
        elif split == False:
            train_raw = self.total_data_in_ratio
            train_raw_para = self.parameter
            predict_raw = np.zeros((0, 84, 8))
            predict_raw_para = np.zeros((0, 14))
        
        train_height, predict_height = train_raw.shape[0], predict_raw.shape[0]
        

        """
        Normolize every spectrum and take log_{10}
        """
        input_num = 8
        input_min    = [-10.25695965, -9.41842931, -8.77788068, -6.26408656, -5.42555622, -4.78500758, -9.43388299, -9.332034187]
        input_factor = [ 5.58788693,  4.50063727,  4.23359167,  5.95765576,  4.86602089,  4.59962876 , 16.58187362, 16.472289475]

        input_train = np.zeros((train_height, input_num, 84))
        input_test = np.zeros((predict_height, input_num, 84))
        ####### train_input 
        for i in range(train_height):
            for j in range(input_num):
                input_train[i,j,:] = Apply_Norm(np.log10(train_raw[i,:,j]), input_min[j], input_factor[j])
        ####### test_input
        for i in range(predict_height):
            for j in range(input_num):
                input_test[i,j,:] = Apply_Norm(np.log10(predict_raw[i,:,j]), input_min[j], input_factor[j])

        D_min, D_max = [1.0, 15.0]
        delta_min, delta_max = [0.20, 0.72]
        zh_min, zh_max=[1.0, 20.0]
        va_min, va_max=[1.0, 60.0]
        eta_min, eta_max=[-4.0, 2.0]
        # Ap_min, Ap_max=[]
        nu1_min, nu1_max=[0.0, 3.0]
        nu2_min, nu2_max=[2.0, 2.8]
        R1_min, R1_max=[2.3, 4.3]
        nu3_min, nu3_max=[1.8, 2.7]
        R2_min, R2_max=[5.0, 6.0]
        para_min    = [D_min, delta_min, zh_min, va_min, eta_min, nu1_min, nu2_min, R1_min, nu3_min, R2_min]
        para_factor = [D_max-D_min, delta_max-delta_min, zh_max-zh_min, va_max-va_min, eta_max-eta_min, 
                       nu1_max-nu1_min, np.round(nu2_max-nu2_min,2), R1_max-R1_min, 
                       np.round(nu3_max-nu3_min,2), R2_max-R2_min]

        """
        Fitting target -> $D_0$, $\delta$, $z_h$, $v_A$, $\eta$, $\nu_1$, $\nu_2$, $log_{10}(R_{br,1})$, $\nu_3$, $log_{10}(R_{br,2})$
        """
        source_train = np.zeros((train_height, 10))
        source_test = np.zeros((predict_height, 10))
        for i in range(source_train.shape[1]):
            if i >= 5:
                j = i + 1
            else:
                j = i
            source_train[:, i] = Apply_Norm(train_raw_para[:,j], para_min[i], para_factor[i])   
            source_test[:, i] = Apply_Norm(predict_raw_para[:,j], para_min[i], para_factor[i])
        

        print("Shape for training Input: ", input_train.shape)
        print("Shape for  testing Input: ", input_test.shape)
        print("Shape for training Target: ", source_train.shape)
        print("Shape for  testing Target: ", source_test.shape)
        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        
        self.input_train, self.input_test = input_train, input_test
        self.source_train, self.source_test = source_train, source_test
        
        
        
class Select_Sample:
    def __init__(self, parameter, data, total_chisq_list, sigma):
        
        self.len = len(parameter)
        self.parameter = parameter
        self.data = data
        self.total_chisq_list = total_chisq_list
        self.sigma = sigma
        
    def Sample(self):
        import time
        ticks_1 = time.time()
        #######################################################################################################
        # 1 sigma: min_chi_square + 15.94 
        # 3 sigma: min_chi_square + 33.20
        # 6 sigma: min_chi_square + 70
        if self.sigma == 1:
            deltachi = 15.94 
        elif self.sigma == 2:
            deltachi = 24.03
        elif self.sigma == 3:
            deltachi = 33.20
        elif self.sigma == 4:
            deltachi = 43.82
        elif self.sigma == 5:
            deltachi = 56.04
        elif self.sigma == 6:
            deltachi = 70
        else:
            print(" ONLY 1 \sigma, 2 \sigma, 3 \sigma, 6 \sigma are available.")
            sys.exit(1)
        total_chisq_list = self.total_chisq_list
        
#         MIN = 347.73266
#         MIN = 347.33
#         MIN = min(min(total_chisq_list),347.73266)
#         print(MIN)
        MIN = min(total_chisq_list)
        
        
        length = np.count_nonzero(total_chisq_list < MIN + deltachi )
#         length = np.count_nonzero(total_chisq_list < min(total_chisq_list) + deltachi )
        data_sigma = np.zeros((length,84,6))
        para_sigma = np.zeros((length, 14))
        chi_sigma = []
        count = 0

        for i in range(len(total_chisq_list)):
            if total_chisq_list[i] < MIN + deltachi:
                para_sigma[count] = self.parameter[i]
                chi_sigma.append(total_chisq_list[i])
                for j in range(6):
                    data_sigma[count,:,j] = self.data[i,:,j]
                count +=1
        print("There are {} data in the {} \sigma region.".format(count,self.sigma))
        #######################################################################################################
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        print("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))

        return para_sigma, data_sigma, np.array(chi_sigma)
    
    

    

class Calculate_Chi_Square:
    def __init__(self, E=[0], Li=0, Be=0, B=0, C=0, O=0, data=0, usedata = False):
        
        self.len = len(E)
        self.E, self.Li, self.Be, self.B, self.C, self.O = E, Li, Be, B, C, O
        if usedata == True:
            self.len = len(data)
            self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        
    def chi_square(self):
        from scipy import interpolate
        import time
        def interpolate_chisq(P,F,S):
            return round(np.sum(((P-np.array(F))**2)/(np.array(S)**2)), 5)
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        ######################################################################################################

        """
        Load Experimental Data
        """
        ####################################################################################
        exp_data_path = "../Data/Exp_Data/"

        LiAMS, LiV = np.load(exp_data_path + "Li_AMS2.npy"),np.load(exp_data_path + "Li_Voyager.npy")
        Li_Eams, Li_Fams, Li_Sams = LiAMS[0], LiAMS[1], LiAMS[2]
        Li_Evy, Li_Fvy, Li_Svy = LiV[0], LiV[1], LiV[2]
        Li_data = [[Li_Eams, Li_Fams, Li_Sams],[Li_Evy, Li_Fvy, Li_Svy]]
        Li_name = ["AMS II", "Voyage"]
        Li_color = ["b","k"]

        BeAMS, BeV = np.load(exp_data_path + "Be_AMS2.npy"), np.load(exp_data_path + "Be_Voyager.npy")
        Be_Eams, Be_Fams, Be_Sams = BeAMS[0], BeAMS[1], BeAMS[2]
        Be_Evy, Be_Fvy, Be_Svy = BeV[0], BeV[1], BeV[2]
        Be_data = [[Be_Eams, Be_Fams, Be_Sams],[Be_Evy, Be_Fvy, Be_Svy]]
        Be_name = ["AMS II", "Voyage"]
        Be_color = ["b","k"]

        BAMS, BV, BA = np.load(exp_data_path + "B_AMS2.npy"), np.load(exp_data_path + "B_Voyager.npy"), np.load(exp_data_path + "B_ACE.npy")
        B_Eams, B_Fams, B_Sams = BAMS[0], BAMS[1], BAMS[2]
        B_Evy, B_Fvy, B_Svy = BV[0], BV[1], BV[2]
        B_Eace,B_Face,B_Sace = BA[0], BA[1], BA[2]
        B_data = [[B_Eams, B_Fams, B_Sams],[B_Evy, B_Fvy, B_Svy],[B_Eace,B_Face,B_Sace]]
        B_name = ["AMS II", "Voyage", "ACE"]
        B_color = ["b","k","m"]

        CAMS, CV, CA = np.load(exp_data_path + "C_AMS2.npy"), np.load(exp_data_path + "C_Voyager.npy"), np.load(exp_data_path + "C_ACE.npy")
        C_Eams, C_Fams, C_Sams = CAMS[0], CAMS[1], CAMS[2]
        C_Evy, C_Fvy, C_Svy = CV[0], CV[1], CV[2]
        C_Eace,C_Face,C_Sace = CA[0], CA[1], CA[2]
        C_data = [[C_Eams, C_Fams, C_Sams],[C_Evy, C_Fvy, C_Svy],[C_Eace,C_Face,C_Sace]]
        C_name = ["AMS II", "Voyage", "ACE"]
        C_color = ["b","k","m"]

        OAMS, OV, OA = np.load(exp_data_path + "O_AMS2.npy"), np.load(exp_data_path + "O_Voyager.npy"), np.load(exp_data_path + "O_ACE.npy")
        O_Eams, O_Fams, O_Sams = OAMS[0], OAMS[1], OAMS[2]
        O_Evy, O_Fvy, O_Svy = OV[0], OV[1], OV[2]
        O_Eace,O_Face,O_Sace = OA[0], OA[1], OA[2]
        O_data = [[O_Eams, O_Fams, O_Sams],[O_Evy, O_Fvy, O_Svy],[O_Eace,O_Face,O_Sace]]
        O_name = ["AMS II", "Voyage", "ACE"]
        O_color = ["b","k","m"]
        ####################################################################################
        # Fit the Spectrum and Calculate Chi-Square
        total_chisq = np.zeros((self.len))
        for i in range(self.len):
            Li_interpolate = interpolate.interp1d((self.E[0]), self.Li[i], kind='cubic')
            Be_interpolate = interpolate.interp1d((self.E[0]), self.Be[i], kind='cubic')
            B_interpolate = interpolate.interp1d((self.E[0]), self.B[i], kind='cubic')
            C_interpolate = interpolate.interp1d((self.E[0]), self.C[i], kind='cubic')
            O_interpolate = interpolate.interp1d((self.E[0]), self.O[i], kind='cubic')

            total_chisq[i] = (
                interpolate_chisq(P=Li_interpolate(Li_Eams), F=Li_Fams, S=Li_Sams)+
                interpolate_chisq(P=Li_interpolate(Li_Evy), F=Li_Fvy, S=Li_Svy)+

                interpolate_chisq(P=Be_interpolate(Be_Eams), F=Be_Fams, S=Be_Sams)+
                interpolate_chisq(P=Be_interpolate(Be_Evy), F=Be_Fvy, S=Be_Svy)+

                interpolate_chisq(P=B_interpolate(B_Eams), F=B_Fams, S=B_Sams)+
                interpolate_chisq(P=B_interpolate(B_Evy), F=B_Fvy, S=B_Svy)+
                interpolate_chisq(P=B_interpolate(B_Eace), F=B_Face, S=B_Sace)+

                interpolate_chisq(P=C_interpolate(C_Eams), F=C_Fams, S=C_Sams)+
                interpolate_chisq(P=C_interpolate(C_Evy), F=C_Fvy, S=C_Svy)+
                interpolate_chisq(P=C_interpolate(C_Eace), F=C_Face, S=C_Sace)+

                interpolate_chisq(P=O_interpolate(O_Eams), F=O_Fams, S=O_Sams)+
                interpolate_chisq(P=O_interpolate(O_Evy), F=O_Fvy, S=O_Svy)+
                interpolate_chisq(P=O_interpolate(O_Eace), F=O_Face, S=O_Sace)
            )

        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        print("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))
        return total_chisq

    
    


# import Mock_Data_Processing_for_Training as MDPfT
# import Calculate_Chi_Square as CCS
# import Select_Sample as SS




class ReCalculateAp:
    def __init__(self, data):
        self.length = len(data)
        self.data = data
        self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        self.ap_5, self.minchi = 0, 0

    def GetBestAp(self):
        import time
        from scipy import interpolate
        
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################

        """
        Load Experimental Data
        """
        ####################################################################################
        LiAMS, LiV = np.load("./Exp_Data/Li_AMS2.npy"),np.load("./Exp_Data/Li_Voyager.npy")
        Li_Eams, Li_Fams, Li_Sams = LiAMS[0], LiAMS[1], LiAMS[2]
        Li_Evy, Li_Fvy, Li_Svy = LiV[0], LiV[1], LiV[2]
        BeAMS, BeV = np.load("./Exp_Data/Be_AMS2.npy"), np.load("./Exp_Data/Be_Voyager.npy")
        Be_Eams, Be_Fams, Be_Sams = BeAMS[0], BeAMS[1], BeAMS[2]
        Be_Evy, Be_Fvy, Be_Svy = BeV[0], BeV[1], BeV[2]
        BAMS, BV, BA = np.load("./Exp_Data/B_AMS2.npy"), np.load("./Exp_Data/B_Voyager.npy"), np.load("./Exp_Data/B_ACE.npy")
        B_Eams, B_Fams, B_Sams = BAMS[0], BAMS[1], BAMS[2]
        B_Evy, B_Fvy, B_Svy = BV[0], BV[1], BV[2]
        B_Eace,B_Face,B_Sace = BA[0], BA[1], BA[2]
        CAMS, CV, CA = np.load("./Exp_Data/C_AMS2.npy"), np.load("./Exp_Data/C_Voyager.npy"), np.load("./Exp_Data/C_ACE.npy")
        C_Eams, C_Fams, C_Sams = CAMS[0], CAMS[1], CAMS[2]
        C_Evy, C_Fvy, C_Svy = CV[0], CV[1], CV[2]
        C_Eace,C_Face,C_Sace = CA[0], CA[1], CA[2]
        OAMS, OV, OA = np.load("./Exp_Data/O_AMS2.npy"), np.load("./Exp_Data/O_Voyager.npy"), np.load("./Exp_Data/O_ACE.npy")
        O_Eams, O_Fams, O_Sams = OAMS[0], OAMS[1], OAMS[2]
        O_Evy, O_Fvy, O_Svy = OV[0], OV[1], OV[2]
        O_Eace,O_Face,O_Sace = OA[0], OA[1], OA[2]
        ####################################################################################

        def findbestAp(E,Li,Be,B,C,O):
            aplist = np.linspace(1.9135/5.,7.0114/5.,120)
            chisq = []
            for ap in aplist:     
                norm_interpolate_Li = interpolate.interp1d((E), (Li*ap), kind='cubic')
                norm_interpolate_Be = interpolate.interp1d((E), (Be*ap), kind='cubic')
                pred_interpolate_B  = interpolate.interp1d((E), (B*ap), kind='cubic')
                pred_interpolate_C  = interpolate.interp1d((E), (C*ap), kind='cubic')
                norm_interpolate_O  = interpolate.interp1d((E), (O*ap), kind='cubic')
                chisq.append(np.sum((norm_interpolate_Li(Li_Eams)-np.array(Li_Fams))**2/np.array(Li_Sams)**2)+
                                  np.sum((norm_interpolate_Li(Li_Evy)-np.array(Li_Fvy))**2/np.array(Li_Svy)**2)+

                                  np.sum((norm_interpolate_Be(Be_Eams)-np.array(Be_Fams))**2/np.array(Be_Sams)**2)+
                                  np.sum((norm_interpolate_Be(Be_Evy)-np.array(Be_Fvy))**2/np.array(Be_Svy)**2)+

                                  np.sum((pred_interpolate_B(B_Eams)-np.array(B_Fams))**2/np.array(B_Sams)**2)+
                                  np.sum((pred_interpolate_B(B_Evy)-np.array(B_Fvy))**2/np.array(B_Svy)**2)+
                                  np.sum((pred_interpolate_B(B_Eace)-np.array(B_Face))**2/np.array(B_Sace)**2)+

                                  np.sum((pred_interpolate_C(C_Eams)-np.array(C_Fams))**2/np.array(C_Sams)**2)+
                                  np.sum((pred_interpolate_C(C_Evy)-np.array(C_Fvy))**2/np.array(C_Svy)**2)+
                                  np.sum((pred_interpolate_C(C_Eace)-np.array(C_Face))**2/np.array(C_Sace)**2)+

                                  np.sum((norm_interpolate_O(O_Eams)-np.array(O_Fams))**2/np.array(O_Sams)**2)+
                                  np.sum((norm_interpolate_O(O_Evy)-np.array(O_Fvy))**2/np.array(O_Svy)**2)+
                                  np.sum((norm_interpolate_O(O_Eace)-np.array(O_Face))**2/np.array(O_Sace)**2)
                                 )
            return chisq.index(min(chisq)), chisq


        aplist = np.linspace(1.9135/5.,7.0114/5.,120)
        ap_5, minchi = [], []
        for i in range(self.length):
            index, chisq = findbestAp(self.E[i],self.Li[i],self.Be[i],self.B[i],self.C[i],self.O[i])
            minchi.append(min(chisq))
            ap_5.append(aplist[index])
            if (i+1)%100 == 0:
                print("{} data are finished".format(i+1))
        ap_5 = np.array(ap_5)
        minchi = np.array(minchi)
        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        
        self.ap_5, self.minchi = ap_5, minchi
        
        
class ReCalculateN:
    import time
    def __init__(self, parameter,data, ap=1,index=0):
        self.length = len(data)
        self.data = data
        self.parameter = parameter
        self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        self.ap = ap
        self.new_factor, self.new_chi = 0, 0
        self.index = index
    
    def GetBestN(self):
        import time
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################

        """
        Load Experimental Data
        """
        ####################################################################################
        LiAMS, LiV = np.load("./Exp_Data/Li_AMS2.npy"),np.load("./Exp_Data/Li_Voyager.npy")
        Li_Eams, Li_Fams, Li_Sams = LiAMS[0], LiAMS[1], LiAMS[2]
        Li_Evy, Li_Fvy, Li_Svy = LiV[0], LiV[1], LiV[2]
        BeAMS, BeV = np.load("./Exp_Data/Be_AMS2.npy"), np.load("./Exp_Data/Be_Voyager.npy")
        Be_Eams, Be_Fams, Be_Sams = BeAMS[0], BeAMS[1], BeAMS[2]
        Be_Evy, Be_Fvy, Be_Svy = BeV[0], BeV[1], BeV[2]
        BAMS, BV, BA = np.load("./Exp_Data/B_AMS2.npy"), np.load("./Exp_Data/B_Voyager.npy"), np.load("./Exp_Data/B_ACE.npy")
        B_Eams, B_Fams, B_Sams = BAMS[0], BAMS[1], BAMS[2]
        B_Evy, B_Fvy, B_Svy = BV[0], BV[1], BV[2]
        B_Eace,B_Face,B_Sace = BA[0], BA[1], BA[2]
        CAMS, CV, CA = np.load("./Exp_Data/C_AMS2.npy"), np.load("./Exp_Data/C_Voyager.npy"), np.load("./Exp_Data/C_ACE.npy")
        C_Eams, C_Fams, C_Sams = CAMS[0], CAMS[1], CAMS[2]
        C_Evy, C_Fvy, C_Svy = CV[0], CV[1], CV[2]
        C_Eace,C_Face,C_Sace = CA[0], CA[1], CA[2]
        OAMS, OV, OA = np.load("./Exp_Data/O_AMS2.npy"), np.load("./Exp_Data/O_Voyager.npy"), np.load("./Exp_Data/O_ACE.npy")
        O_Eams, O_Fams, O_Sams = OAMS[0], OAMS[1], OAMS[2]
        O_Evy, O_Fvy, O_Svy = OV[0], OV[1], OV[2]
        O_Eace,O_Face,O_Sace = OA[0], OA[1], OA[2]
        ####################################################################################
        def findbestN(E,Li,Be,B,C,O,parameter,ap=1):

            def interpolate_chisq(P,F,S):
                return round(np.sum(((P-np.array(F))**2)/(np.array(S)**2)), 5)

    #         spectrum_E =  data[:,0]
    #         spectrum_Li = data[:,1]
    #         spectrum_Be = data[:,2]
    #         spectrum_B = data[:,3]
    #         spectrum_C = data[:,4]
    #         spectrum_O = data[:,5]
            N_Li = parameter[11]
            N_Be = parameter[12]
            N_O = parameter[13]
            new_N = np.linspace(0.8,1.2,200)

            chisq_scan_Li, chisq_scan_Be, chisq_scan_O = [], [], []
            for element in new_N:       
                pred_interpolate_Li = interpolate.interp1d((E), (Li/N_Li*element*ap), kind='cubic')
                pred_interpolate_Be = interpolate.interp1d((E), (Be/N_Be*element*ap), kind='cubic')
                pred_interpolate_O = interpolate.interp1d((E), (O/N_O*element*ap), kind='cubic')
                chisq_scan_Li.append(interpolate_chisq(P=pred_interpolate_Li(Li_Eams), F=Li_Fams, S=Li_Sams)+
                                     interpolate_chisq(P=pred_interpolate_Li(Li_Evy), F=Li_Fvy, S=Li_Svy))
                chisq_scan_Be.append(interpolate_chisq(P=pred_interpolate_Be(Be_Eams), F=Be_Fams, S=Be_Sams)+
                                     interpolate_chisq(P=pred_interpolate_Be(Be_Evy), F=Be_Fvy, S=Be_Svy))
                chisq_scan_O.append(interpolate_chisq(P=pred_interpolate_O(O_Eams), F=O_Fams, S=O_Sams)+
                                    interpolate_chisq(P=pred_interpolate_O(O_Evy), F=O_Fvy, S=O_Svy)+
                                    interpolate_chisq(P=pred_interpolate_O(O_Eace), F=O_Face, S=O_Sace))

            new_normal_factor = np.zeros(3)
            new_normal_factor[0] = new_N[chisq_scan_Li.index(min(chisq_scan_Li))]
            new_normal_factor[1] = new_N[chisq_scan_Be.index(min(chisq_scan_Be))]
            new_normal_factor[2] = new_N[chisq_scan_O.index(min(chisq_scan_O))]

            '''
            Calculate New chisquare
            '''
            Li_interpolate = interpolate.interp1d(E, Li/N_Li*new_normal_factor[0]*ap, kind='cubic')
            Be_interpolate = interpolate.interp1d(E, Be/N_Be*new_normal_factor[1]*ap, kind='cubic')
            B_interpolate = interpolate.interp1d(E, B*ap, kind='cubic')
            C_interpolate = interpolate.interp1d(E, C*ap, kind='cubic')
            O_interpolate = interpolate.interp1d(E, O/N_O*new_normal_factor[2]*ap, kind='cubic')

            total_chisq = (
                interpolate_chisq(P=Li_interpolate(Li_Eams), F=Li_Fams, S=Li_Sams)+
                interpolate_chisq(P=Li_interpolate(Li_Evy), F=Li_Fvy, S=Li_Svy)+

                interpolate_chisq(P=Be_interpolate(Be_Eams), F=Be_Fams, S=Be_Sams)+
                interpolate_chisq(P=Be_interpolate(Be_Evy), F=Be_Fvy, S=Be_Svy)+

                interpolate_chisq(P=B_interpolate(B_Eams), F=B_Fams, S=B_Sams)+
                interpolate_chisq(P=B_interpolate(B_Evy), F=B_Fvy, S=B_Svy)+
                interpolate_chisq(P=B_interpolate(B_Eace), F=B_Face, S=B_Sace)+

                interpolate_chisq(P=C_interpolate(C_Eams), F=C_Fams, S=C_Sams)+
                interpolate_chisq(P=C_interpolate(C_Evy), F=C_Fvy, S=C_Svy)+
                interpolate_chisq(P=C_interpolate(C_Eace), F=C_Face, S=C_Sace)+

                interpolate_chisq(P=O_interpolate(O_Eams), F=O_Fams, S=O_Sams)+
                interpolate_chisq(P=O_interpolate(O_Evy), F=O_Fvy, S=O_Svy)+
                interpolate_chisq(P=O_interpolate(O_Eace), F=O_Face, S=O_Sace)
            )


            return new_normal_factor, total_chisq


        """
        Load Mock Data
        """   
        ap_5 = self.ap
        new_factor = np.zeros((self.length,3))
        new_chi = np.zeros(self.length)
        if self.index == 1:
            for i in range(self.length):
                new_factor[i,:], new_chi[i] = findbestN(self.E[i], self.Li[i], self.Be[i], self.B[i], self.C[i], self.O[i], self.parameter[i,:])
                if (i+1)%100 == 0:
                    print("{} data are finished".format(i+1))
        elif self.index == 0:
            for i in range(self.length):
                new_factor[i,:], new_chi[i] = findbestN(self.E[i], self.Li[i], self.Be[i], self.B[i], self.C[i], self.O[i], self.parameter[i,:],ap=ap_5[i])

                if (i+1)%100 == 0:
                    print("{} data are finished".format(i+1))

        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        self.new_factor, self.new_chi = new_factor, new_chi
          
          
          
        
class  New_Parameter:
    import time
    
    def __init__(self, parameter, new_factor, ap_5=1, index=0):
        self.length = len(parameter)
        self.factor = new_factor
        self.parameter = parameter
        
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        if index == 1 :
            new_parameter = np.zeros((parameter.shape[0],14))
            new_parameter[:,0:11] = parameter[:,0:11]
            new_parameter[:,11:14] = new_factor
        elif index == 0:
            new_parameter = np.zeros((parameter.shape[0],14))
            new_parameter[:,0:5] = parameter[:,0:5]
            new_parameter[:,5] = ap_5*5
            new_parameter[:,6:11] = parameter[:,6:11]
            new_parameter[:,11:14] = new_factor

        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        
        self.new_parameter = new_parameter      
        

class Recovery:
    def __init__(self, prediction):
        self.length = len(prediction)
            
        D_min, D_max=[1.0, 15.0]
        delta_min, delta_max=[0.20, 0.72]
        zh_min, zh_max=[1.0, 20.0]
        va_min, va_max=[1.0, 60.0]
        eta_min, eta_max=[-4.0, 2.0]
        # Ap_min, Ap_max=[]
        nu1_min, nu1_max=[0.0, 3.0]
        nu2_min, nu2_max=[2.0, 2.8]
        R1_min, R1_max=[2.3, 4.3]
        nu3_min, nu3_max=[1.8, 2.7]
        R2_min, R2_max=[5.0, 6.0]
        para_min    = [D_min, delta_min, zh_min, va_min, eta_min, nu1_min, nu2_min, R1_min, nu3_min, R2_min]
        para_factor = [D_max-D_min, delta_max-delta_min, zh_max-zh_min, va_max-va_min, eta_max-eta_min, 
                       nu1_max-nu1_min, np.round(nu2_max-nu2_min,2), R1_max-R1_min, 
                       np.round(nu3_max-nu3_min,2), R2_max-R2_min]

        para_recovery = np.zeros((len(prediction),10))
        for i in range(10):
            para_recovery[:,i] = prediction[:,i]*para_factor[i]+para_min[i]
            
            
        self.para_recovery = para_recovery  
        self.D,  self.delta, self.Zh, self.va, self.eta = para_recovery[:,0],para_recovery[:,1],para_recovery[:,2],para_recovery[:,3],para_recovery[:,4]
        self.nu1, self.nu2, self.R1, self.nu3, self.R2 = para_recovery[:,5],para_recovery[:,6],para_recovery[:,7],para_recovery[:,8],para_recovery[:,9]
               
            
            
            

            
###############################################            

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
            

#     def Create_Pseudodata(parameter,data,chi, LOW= 10 , HIGH = 30 , number = 100, index=0):
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
        
        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        """
        Load Experimental Data
        """
        ####################################################################################
        LiAMS, LiV = np.load("./Exp_Data/Li_AMS2.npy"),np.load("./Exp_Data/Li_Voyager.npy")
        Li_Eams, Li_Fams, Li_Sams = LiAMS[0], LiAMS[1], LiAMS[2]
        Li_Evy, Li_Fvy, Li_Svy = LiV[0], LiV[1], LiV[2]
        BeAMS, BeV = np.load("./Exp_Data/Be_AMS2.npy"), np.load("./Exp_Data/Be_Voyager.npy")
        Be_Eams, Be_Fams, Be_Sams = BeAMS[0], BeAMS[1], BeAMS[2]
        Be_Evy, Be_Fvy, Be_Svy = BeV[0], BeV[1], BeV[2]
        BAMS, BV, BA = np.load("./Exp_Data/B_AMS2.npy"), np.load("./Exp_Data/B_Voyager.npy"), np.load("./Exp_Data/B_ACE.npy")
        B_Eams, B_Fams, B_Sams = BAMS[0], BAMS[1], BAMS[2]
        B_Evy, B_Fvy, B_Svy = BV[0], BV[1], BV[2]
        B_Eace,B_Face,B_Sace = BA[0], BA[1], BA[2]
        CAMS, CV, CA = np.load("./Exp_Data/C_AMS2.npy"), np.load("./Exp_Data/C_Voyager.npy"), np.load("./Exp_Data/C_ACE.npy")
        C_Eams, C_Fams, C_Sams = CAMS[0], CAMS[1], CAMS[2]
        C_Evy, C_Fvy, C_Svy = CV[0], CV[1], CV[2]
        C_Eace,C_Face,C_Sace = CA[0], CA[1], CA[2]
        OAMS, OV, OA = np.load("./Exp_Data/O_AMS2.npy"), np.load("./Exp_Data/O_Voyager.npy"), np.load("./Exp_Data/O_ACE.npy")
        O_Eams, O_Fams, O_Sams = OAMS[0], OAMS[1], OAMS[2]
        O_Evy, O_Fvy, O_Svy = OV[0], OV[1], OV[2]
        O_Eace,O_Face,O_Sace = OA[0], OA[1], OA[2]
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
        print("\033[3;33mTime for Search The Pack of Spectrum : {:.4f} min\033[0;m".format((ticks_Search_The_Pack_2-ticks_Search_The_Pack_1)/60.))
        
        
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
        print("\033[3;33mTime for Create Pseudodata : {:.4f} min\033[0;m".format((ticks_Pseudodata_2-ticks_Pseudodata_1)/60.))
        
        
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


        print("Pseudo Normal Factor")
        print(" {:^4} {:^6} {:^6} {:^6}".format("","Li","Be","O"))
        print(" {:^4} {:^6} {:^6} {:^6}".format("#",len(norm_factor[:,11]),len(norm_factor[:,12]),len(norm_factor[:,13])))
        print(" {:^4} {:^4.4f} {:^4.4f} {:^4.4f}".format("Max",max(norm_factor[:,11]),max(norm_factor[:,12]),max(norm_factor[:,13])))
        print(" {:^4} {:^4.4f} {:^4.4f} {:^4.4f}".format("Min",min(norm_factor[:,11]),min(norm_factor[:,12]),min(norm_factor[:,13])))
        print(" {:^4} {:^4.4f} {:^4.4f} {:^4.4f}".format("Ave.",np.average(norm_factor[:,11]),np.average(norm_factor[:,12]),np.average(norm_factor[:,13])))
        
        
        ticks_N_2 = time.time()
        print("\033[3;33mTime for N_Li : {:.4f} min\033[0;m".format((ticks_N_2-ticks_N_1)/60.))
        

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
            print("\033[3;33mTime for Recorver_Spectrum : {:.4f} min\033[0;m".format((ticks_Recorver_Spectrum_2-ticks_Recorver_Spectrum_1)/60.))
        
            
            
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
        print("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))
        return normalfactor, pseudodata
        
#     self.normalfactor = normalfactor
#     self.pseudodata = pseudodata
    




