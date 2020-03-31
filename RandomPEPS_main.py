

import numpy as np
import copy as cp
from scipy import linalg
import matplotlib.pyplot as plt
import ncon
import time
import pickle
import pandas as pd



import SimpleUpdate as BP
import BPupdate_PEPS_smart_trancation2 as gPEPS
import ncon_lists_generator as nlg
import DoubleEdgeFactorGraphs as defg
import StructureMatrixGenerator as tnf
import RandomPEPS as hmf
import bmpslib as bmps

def randomPEPSmainFunction():

    #
    #################################################    MAIN    ###########################################################
    #
    flag_run_new_experiment = 1
    flag_save_variables = 0
    flag_load_data = 0
    flag_calculating_expectations = 1
    flag_save_xlsx = 0


    #
    ############################################    EXPERIMENT PARAMETERS    ###############################################
    #

    #np.random.seed(seed=9)

    N, M = 10, 10

    bc = 'open'
    dE = 1e-5
    t_max = 200
    dumping = 0.2
    epsilon = 1e-5
    D_max = [3]
    mu = -1
    sigma = 0
    Jk = np.random.normal(mu, sigma, np.int((N - 1) * M + (M - 1) * N))
    #Jk = np.random.normal(mu, sigma, np.int(2 * (N * M)))
    dt = [0.5, 0.1, 0.05, 0.01, 0.005]
    iterations = 10


    if bc == 'open':
        smat, imat = tnf.finitePEPSobcStructureMatrixGenerator(N, M)
    elif bc == 'periodic':
        smat, imat = tnf.squareFinitePEPSpbcStructureMatrixGenerator(N * M)


    Dp = [16, 32]
    p = 2
    h = 0
    environment_size = [0, 1, 2]

    #
    ############################################  RUN AND COLLECT DATA  ####################################################
    #
    if flag_run_new_experiment:

        BP_data = []
        SU_data = []

        for D in D_max:
            b, time_su = hmf.RandomPEPS_SU(N, M, Jk, dE, D, bc, dt, iterations)
            TT0, LL0 = b[0], b[1]
            a, time_bp = hmf.RandomPEPS_BP(N, M, Jk, dE, D, t_max, epsilon, dumping, bc, dt, iterations, [TT0, LL0])
            BP_data.append(a)
            SU_data.append(b)


    #
    #################################################  SAVING VARIABLES  ###################################################
    #
    if flag_save_variables:

        parameters = [['N, M', [N, M]], ['dE', dE], ['t_max', t_max], ['dumping', dumping], ['epsilon', epsilon], ['D_max', D_max]]
        file_name = "2019_02_23_1_100_OBC_Random_PEPS"
        pickle.dump(parameters, open(file_name + '_parameters.p', "wb"))
        pickle.dump(BP_data, open(file_name + '_BP.p', "wb"))
        pickle.dump(SU_data, open(file_name + '_gPEPS.p', "wb"))



    #
    #################################################   LOADING DATA   #####################################################
    #
    if flag_load_data:

        file_name_bp = "2019_02_23_1_100_OBC_Random_PEPS_BP.p"
        file_name_gpeps = "2019_02_23_1_100_OBC_Random_PEPS_gPEPS.p"
        file_name1 = "2019_02_23_1_100_OBC_Random_PEPS_parameters.p"

        data_bp = pickle.load(open(file_name_bp, "rb"))
        data_su = pickle.load(open(file_name_gpeps, "rb"))
        data_params = pickle.load(open(file_name1, "rb"))


    rho_SU = []
    rho_SU_0 = []
    rho_SU_bmps = []
    rho_SU_0_bmps = []
    rho_SU_bmps_single = []
    rho_SU_0_bmps_single = []

    #rho_SU_2site = []
    #rho_SU_0_2site = []
    #rho_BP_2site = []

    #
    ############################################  CALCULATING EXPECTATIONS  ################################################
    #
    data_bp = BP_data
    data_su = SU_data
    data_params = [['N, M', [N, M]], ['dE', dE], ['t_max', t_max], ['dumping', dumping], ['epsilon', epsilon], ['D_max', D_max]]

    indices = []
    counter = 0
    counter2 = N * (M - 1) - 1 + (N - 1) * (M - 1) + 1
    counter3 = 0
    while counter3 < N:
        indices += range(counter, counter + M - 1)
        indices.append(counter2)
        counter += (M - 1)
        counter2 += 1
        counter3 += 1
    indices.pop()

    print('calc_rdms')
    if flag_calculating_expectations:
        for ii in range(len(data_params[5][1])):

            graph = data_bp[ii]
            TT_SU_0, LL_SU_0, TT_SU, LL_SU = data_su[ii]
            TT_SU_bmps = cp.deepcopy(TT_SU)
            TT_SU_0_bmps = cp.deepcopy(TT_SU_0)

        #
        ######### CALCULATING REDUCED DENSITY MATRICES  ########
        #

            for i in range(len(TT_SU)):
                rho_SU.append(BP.singleSiteRDM(i, TT_SU, LL_SU, smat))
                #rho_SU_0.append(BP.tensor_reduced_dm(i, TT_SU_0, LL_SU_0, smat))
            rho_BP = graph.rdm_using_factors()

            #for Ek in range(len(LL_SU)):
            #    rho_SU_2site.append(BP.two_site_reduced_density_matrix(Ek, TT_SU, LL_SU, smat))
            #    rho_SU_0_2site.append(BP.two_site_reduced_density_matrix(Ek, TT_SU_0, LL_SU_0, smat))
            #    rho_BP_2site.append(BP.BP_two_site_rdm_using_factor_beliefs(Ek, graph, smat))


            '''
            TT_SU_bmps = BP.absorb_all_sqrt_bond_vectors(TT_SU_bmps, LL_SU, smat)
            TT_SU_bmps = tnf.PEPS_OBC_broadcast_to_Itai(TT_SU_bmps, [N, M], p, data_params[5][1][ii])
            SU_peps = bmps.peps(N, M)
            for t, T in enumerate(TT_SU_bmps):
                i, j = np.unravel_index(t, [N, M])
                SU_peps.set_site(T, i, j)
            for dp_idx, dp in enumerate(Dp):
                print('Dp:', dp)
                rho_SU_bmps.append(bmps.calculate_PEPS_2RDM(SU_peps, dp))
                rho_SU_bmps_single.append([])
                # 4x4 [0, 1, 2, 21, 3, 4, 5, 22, 6, 7, 8, 23, 9, 10, 11]
    
                for jj in indices:
                    rho_SU_bmps_single[dp_idx].append(np.einsum(rho_SU_bmps[dp_idx][jj], [0, 1, 2, 2], [0, 1]))
                rho_SU_bmps_single[dp_idx].append(np.einsum(rho_SU_bmps[dp_idx][jj], [1, 1, 2, 3], [2, 3]))
            '''
            TT_SU_0_bmps = BP.absorbAllTensorNetWeights(TT_SU_0_bmps, LL_SU_0, smat)
            TT_SU_0_bmps = tnf.PEPS_OBC_broadcast_to_Itai(TT_SU_0_bmps, [N, M], p, data_params[5][1][ii])
            SU_0_peps = bmps.peps(N, M)
            for t, T in enumerate(TT_SU_0_bmps):
                i, j = np.unravel_index(t, [N, M])
                SU_0_peps.set_site(T, i, j)
            for dp_idx, dp in enumerate(Dp):
                print('Dp:', dp)
                rho_SU_0_bmps.append(bmps.calculate_PEPS_2RDM(SU_0_peps, dp))
                rho_SU_0_bmps_single.append([])
                for jj in indices:
                    rho_SU_0_bmps_single[dp_idx].append(np.einsum(rho_SU_0_bmps[dp_idx][jj], [0, 1, 2, 2], [0, 1]))
                rho_SU_0_bmps_single[dp_idx].append(np.einsum(rho_SU_0_bmps[dp_idx][jj], [1, 1, 2, 3], [2, 3]))
    '''
        #
        ###################################################  PLOTTING DATA  ####################################################
        #
    # ttd stand for "total trace distance"
    ttd_su_su0 = [0] * len(Dp)
    ttd_su_bp = [0] * len(Dp)
    ttd_bp_su0 = [0] * len(Dp)
    ttd_su_su_bmps = [0] * len(Dp)
    ttd_su_su0_bmps = [0] * len(Dp)
    ttd_su0_su0_bmps = [0] * len(Dp)
    ttd_bp_su_bmps = [0] * len(Dp)
    ttd_bp_su0_bmps = [0] * len(Dp)
    
    #ttd_su_bp_2site = 0
    #ttd_su_su0_2site = 0
    #ttd_bp_su0_2site = 0
    #ttd_su_bmps_bp = []
    #ttd_su_0_bmps_bp = []
    for dp_idx, dp in enumerate(Dp):
        for i in range(len(rho_SU)):
            ttd_su_su0[dp_idx] += BP.trace_distance(rho_SU[i], rho_SU_0[i])
            ttd_su_bp[dp_idx] += BP.trace_distance(rho_SU[i], rho_BP[i])
            ttd_bp_su0[dp_idx] += BP.trace_distance(rho_BP[i], rho_SU_0[i])
            ttd_su_su_bmps[dp_idx] += BP.trace_distance(rho_SU[i], rho_SU_bmps_single[dp_idx][i])
            ttd_su_su0_bmps[dp_idx] += BP.trace_distance(rho_SU[i], rho_SU_0_bmps_single[dp_idx][i])
            ttd_su0_su0_bmps[dp_idx] += BP.trace_distance(rho_SU_0[i], rho_SU_0_bmps_single[dp_idx][i])
            ttd_bp_su_bmps[dp_idx] += BP.trace_distance(rho_BP[i], rho_SU_bmps_single[dp_idx][i])
            ttd_bp_su0_bmps[dp_idx] += BP.trace_distance(rho_BP[i], rho_SU_0_bmps_single[dp_idx][i])
    
        #for i in range(len(rho_BP_2site)):
        #    ttd_su_bp_2site += BP.trace_distance(rho_SU_2site[i], rho_BP_2site[i])
        #    ttd_su_su0_2site += BP.trace_distance(rho_SU_2site[i], rho_SU_0_2site[i])
        #    ttd_bp_su0_2site += BP.trace_distance(rho_SU_0_2site[i], rho_BP_2site[i])
    
        #for i in range(len(Dp)):
        #    tot_su_bp = 0
        #    tot_su_0_bp = 0
            #for ii in range(len(rho_SU_bmps)):
            #    tot_su_bp += BP.trace_distance(rho_SU_bmps[i][ii].reshape(4, 4), rho_SU_2site[ii])
            #    tot_su_0_bp += BP.trace_distance(rho_SU_0_bmps[i][ii].reshape(4, 4), rho_SU_2site[ii])
            #ttd_su_bmps_bp.append(tot_su_bp)
            #ttd_su_0_bmps_bp.append(tot_su_0_bp)
        
        
        print('Dp:', dp)
        print('------------ Total Trace Distance (single site) ------------')
        print('SU - SU0 : ', ttd_su_su0[dp_idx])
        print('SU - BP : ', ttd_su_bp[dp_idx])
        print('BP - SU0 : ', ttd_bp_su0[dp_idx])
        print('SU - SU_bmps : ', ttd_su_su_bmps[dp_idx])
        print('SU - SU0_bmps : ', ttd_su_su0_bmps[dp_idx])
        print('SU0 - SU0_bmps : ', ttd_su0_su0_bmps[dp_idx])
        print('BP - SU_bmps : ', ttd_bp_su_bmps[dp_idx])
        print('BP - SU0_bmps : ', ttd_bp_su0_bmps[dp_idx])
        print('------------------------------------------------------------')
        print('\n')
        
    
    plt.figure()
    plt.title('Total single-site trace distance for a 10x10 PEPS with D = 3')
    plt.plot(Dp, np.log(np.array(ttd_su_su0)))
    plt.plot(Dp, np.log(np.array(ttd_su_bp)))
    plt.plot(Dp, np.log(np.array(ttd_bp_su0)))
    plt.plot(Dp, np.log(np.array(ttd_su_su_bmps)), 'x', markersize=10)
    plt.plot(Dp, np.log(np.array(ttd_su_su0_bmps)), '+', markersize=10)
    plt.plot(Dp, np.log(np.array(ttd_su0_su0_bmps)), '>')
    plt.plot(Dp, np.log(np.array(ttd_bp_su_bmps)), 'o')
    plt.plot(Dp, np.log(np.array(ttd_bp_su0_bmps)), '<')
    plt.legend(['SU - 0', 'SU - BP', 'BP - 0', 'SU - SU_bmps', 'SU - 0_bmps', '0 - 0_bmps', 'BP - SU_bmps', 'BP - 0_bmps'])
    plt.grid('on')
    plt.ylabel('Log10(sum_{i}(TraceDistance(Ai, Bi)))')
    plt.xlabel('Dp')
    plt.show()
    '''

    #
    #################################################  SAVING DATA TO XLSX  ################################################
    #

    if flag_save_xlsx:
        '''
        save_list = [E_BP, E_BP_factor_belief, E_gPEPS, E_BP_bmps, E_gPEPS_bmps]
        df = pd.DataFrame(save_list, columns=range(len(Dp) * (len(data_params[5][1]) - 5)), index=['E BP', 'E BP factor belief', 'E gPEPS', 'E BP bmps', 'E gPEPS bmps'])
        filepath = 'energies16AFH_D7_64.xlsx'
        df.to_excel(filepath, index=True)
        '''

    #return time_bp, time_su, ttd_su_su0, ttd_su_bp, ttd_bp_su0, ttd_su_su_bmps, ttd_su_su0_bmps, ttd_su0_su0_bmps, ttd_bp_su_bmps, ttd_bp_su0_bmps
    return rho_SU_0_bmps_single, rho_SU, rho_BP



