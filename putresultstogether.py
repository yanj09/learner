# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:25:38 2017

@author: Jia
"""
from __future__ import division

import os
os.chdir("C:/Users/Jia/Desktop/research_asthma/1_results")

import pandas as pd
import numpy as np
def generate_outcome_tables(commonname, scenariolist, K, xlist, ylist, sumy, outputfilename):
    # read files
    ori_dfs_S = [pd.read_csv(commonname + '{}_MIP_Sbytime.csv'.format(i), header=None) for i in scenariolist ]
    ori_dfs_y = [pd.read_csv(commonname + '{}_MIP_ybytime.csv'.format(i), header=None) for i in scenariolist ]
    # aggreate data by illness degree
    def aggregateilldg(df):
        df['illdg'] = [1, 2, 2, 3] * (4 * K) # K, define illdg
        return df.groupby(['illdg']).sum()        
    dfs_S = [aggregateilldg(df) for df in ori_dfs_S]
    dfs_y = [aggregateilldg(df) for df in ori_dfs_y]
    # compute redundant
    y_I = np.zeros( len(sumy) , dtype=np.int)
    y_II_III = np.zeros( len(sumy) , dtype=np.int)
    percentage_y_III = np.zeros( len(sumy), dtype=np.float64 )
    for index, df in enumerate(dfs_y):
        sum_col = df.values.sum(axis=1)
        # depend on illdg
        sum_II_III = (sum_col[1] + sum_col[2]) 
        y_I[index] = sumy[index] * len(df.columns) - sum_II_III 
        y_II_III[index] = sum_II_III
        percentage_y_III[index] = sum_col[2]/sum_II_III 
                
    percentage_S_I = np.zeros( len(sumy), dtype=np.float64 )
    percentage_S_II = np.zeros( len(sumy), dtype=np.float64 )
    percentage_S_III = np.zeros( len(sumy), dtype=np.float64 )
    for index, df in enumerate(dfs_S):
        sum_col = df.values.sum(axis=1)
        sum_all = sum_col.sum()
        # depend on illdg
        percentage_S_I[index] = sum_col[0]/sum_all
        percentage_S_II[index] = sum_col[1]/sum_all
        percentage_S_III[index] = sum_col[2]/sum_all
    
    agg = []
    # reshape by ylist
    Nrow = len(ylist)
    Ncol = len(xlist)
    y_Is = np.reshape(y_I, [Nrow, Ncol])
    y_II_IIIs = np.reshape(y_II_III, [Nrow, Ncol])
    percentage_y_IIIs = np.reshape(percentage_y_III, [Nrow, Ncol])
    percentage_S_Is = np.reshape(percentage_S_I, [Nrow, Ncol])
    percentage_S_IIs = np.reshape(percentage_S_II, [Nrow, Ncol])
    percentage_S_IIIs = np.reshape(percentage_S_III, [Nrow, Ncol])
    for i in range( len(ylist) ):
        agg += [[ylist[i]] + ['y_I'] + list(y_Is[i,])]
        agg += [[ylist[i]] + ['y_II_III'] + list(y_II_IIIs[i,])]
        agg += [[ylist[i]] + ['%y_III'] + list(percentage_y_IIIs[i,])]
        agg += [[ylist[i]] + ['%S_I'] + list(percentage_S_Is[i,])]
        agg += [[ylist[i]] + ['%S_II'] + list(percentage_S_IIs[i,])]
        agg += [[ylist[i]] + ['%S_III'] + list(percentage_S_IIIs[i,])]
    agg = pd.DataFrame(agg)
    agg.to_csv(outputfilename, header=False, index=False)
    
def generate_technicalnotes_tables(commonname, scenariolist, xlist, ylist, outputfilename):
    Nrow = len(ylist)
    Ncol = len(xlist)
    # optimality gap
    opt_gaps = []
    for fname in [commonname + '{}_log_mip'.format(_fn) for _fn in scenariolist]:
        with open(fname, 'rb') as fh:
            #first = next(fh).decode()   
            fh.seek(-1024, 2)
            last = fh.readlines()[-1].decode()
            opt_gaps += [str(last.split(' ')[-1].split('\n')[0])]
    shaped_gaps = np.reshape(opt_gaps, [Nrow, Ncol])
    heu_1s = []
    heu_2s = []
    for fname in [commonname + '{}_optgap.csv'.format(_fn) for _fn in scenariolist]:
        with open(fname, 'r') as fh:
            f_line = fh.readline()
            heu_1_gap = '%.4f' % (float(f_line.split(',')[-2]) * 100) + '%'
            heu_2_gap = '%.4f' % (float(f_line.split(',')[-1]) * 100) + '%'
            heu_1s += [heu_1_gap]
            heu_2s += [heu_2_gap]            
    heu_1s = np.reshape(heu_1s, [Nrow, Ncol])
    heu_2s = np.reshape(heu_2s, [Nrow, Ncol])
    heu_1_times = []
    heu_2_times = []
    for fname in [commonname + '{}_timeinseconds.csv'.format(_fn) for _fn in scenariolist]:
        with open(fname, 'r') as fh:
            f_line = fh.readline()
            h1_time = float(f_line.split(',')[0])
            h2_time = float(f_line.split(',')[1])
            heu_1_times += [h1_time]
            heu_2_times += [h2_time]
    heu_1_times = np.reshape(heu_1_times, [Nrow, Ncol])
    heu_2_times = np.reshape(heu_2_times, [Nrow, Ncol])
    agg = []
    for i in range( len(ylist) ):
        agg += [[ylist[i]] + ['Optgap'] + list(shaped_gaps[i,])]
        agg += [[ylist[i]] + ['H1gap'] + list(heu_1s[i,])]
        agg += [[ylist[i]] + ['H2gap'] + list(heu_2s[i,])]
        agg += [[ylist[i]] + ['H1timesec'] + list(heu_1_times[i,])]
        agg += [[ylist[i]] + ['H2timesec'] + list(heu_2_times[i,])]   
    agg = pd.DataFrame(agg)
    agg.to_csv(outputfilename, header=False, index=False)
    

def generate_tn_89and1school_tables(commonname, manyschoolscenariolist, oneschoolscenariolist, xlist, ylist, outputfilename): 
    Nrow = len(ylist)
    Ncol = len(xlist)
    # optimality gap
    opt_gap_89schools = []
    for fname in [commonname + '{}_log_mip'.format(_fn) for _fn in manyschoolscenariolist]:
        with open(fname, 'rb') as fh:
            #first = next(fh).decode()   
            fh.seek(-1024, 2)
            last = fh.readlines()[-1].decode()
            opt_gap_89schools += [str(last.split(' ')[-1].split('\n')[0])]            
    shaped_gaps_89schools = np.reshape(opt_gap_89schools, [Nrow, Ncol])
    opt_gap_1school = []
    for fname in [commonname + '{}_log_mip'.format(_fn) for _fn in oneschoolscenariolist]:
        with open(fname, 'rb') as fh:
            #first = next(fh).decode()   
            fh.seek(-1024, 2)
            last = fh.readlines()[-1].decode()
            opt_gap_1school += [str(last.split(' ')[-1].split('\n')[0])]
    shaped_gaps_1school = np.reshape(opt_gap_1school, [Nrow, Ncol])
    # objective values, and relative difference
    obj_89schools = []
    for fname in [commonname + '{}_objvalues.csv'.format(_fn) for _fn in manyschoolscenariolist]:
        with open(fname, 'r') as fh:
            f_line = fh.readline()
            obj = float(f_line.split(',')[0])
            obj_89schools += [obj]                
    obj_1school = []
    per_obj_89and1school = []
    for index, fname in enumerate([commonname + '{}_objvalues.csv'.format(_fn) for _fn in oneschoolscenariolist]):
        with open(fname, 'r') as fh:
            f_line = fh.readline()
            obj = float(f_line.split(',')[0])
            obj_1school += [obj]
            per = (obj - obj_89schools[index])/obj_89schools[index]
            per_obj_89and1school += [per]       
    obj_89schools = np.reshape(obj_89schools, [Nrow, Ncol])     
    obj_1school = np.reshape(obj_1school, [Nrow, Ncol])
    per_obj_89and1school = np.reshape(per_obj_89and1school, [Nrow, Ncol])    
    
    agg = []
    for i in range( len(ylist) ):
        agg += [[ylist[i]] + ['Optgap89schools'] + list(shaped_gaps_89schools[i,])]
        agg += [[ylist[i]] + ['Optgap1school'] + list(shaped_gaps_1school[i,])]
        agg += [[ylist[i]] + ['Obj89schools'] + list(obj_89schools[i,])]
        agg += [[ylist[i]] + ['Obj1school'] + list(obj_1school[i,])]
        agg += [[ylist[i]] + ['Relativedifference'] + list(per_obj_89and1school[i,])]   
    agg = pd.DataFrame(agg)
    agg.to_csv(outputfilename, header=False, index=False)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def generate_plots(commonname, scenariolist, K, xcommonname, xlist, ycommonname, ylist, sumy, firstperiodsflag, numperiods, outputfilename1, outputfilename2, dpinum):
    # read files
    ori_dfs_S = [pd.read_csv(commonname + '{}_MIP_Sbytime.csv'.format(i), header=None) for i in scenariolist ]
    ori_dfs_y = [pd.read_csv(commonname + '{}_MIP_ybytime.csv'.format(i), header=None) for i in scenariolist ]
    # aggreate data by illness degree
    def aggregateilldg(df):
        df['illdg'] = [1, 2, 2, 3] * (4 * K) # K, define illdg
        return df.groupby(['illdg']).sum()        
    dfs_S = [aggregateilldg(df) for df in ori_dfs_S]
    dfs_y = [aggregateilldg(df) for df in ori_dfs_y]
    # compute redundant
    for index, df in enumerate(dfs_y):
        for ycol in range(len(df.columns)):
            df.values[0, ycol] = sumy[index] - ( df.values[1, ycol] + df.values[2, ycol] )
        
    # depend on illdg
    colorsopt = ['lightgrey','darkgrey','dimgray']
    patch1 = mpatches.Patch(color=colorsopt[0])
    patch2 = mpatches.Patch(color=colorsopt[1])
    patch3 = mpatches.Patch(color=colorsopt[2])
    Nrow = len(ylist)
    Ncol = len(xlist)
    # S
    f, axes = plt.subplots(Nrow, Ncol, sharey=True, sharex=True, figsize=(15,10))
    for i in range(Nrow):
        for j in range(Ncol):
            if firstperiodsflag:
                axes[i, j].stackplot(range(numperiods), dfs_S[i*Ncol + j].iloc[0,:numperiods], 
                        dfs_S[i*Ncol + j].iloc[1,:numperiods], dfs_S[i*Ncol + j].iloc[2,:numperiods], colors = colorsopt)
            else:
                axes[i, j].stackplot(range(len(dfs_S[i*Ncol + j].columns)), dfs_S[i*Ncol + j].iloc[[0]], 
                        dfs_S[i*Ncol + j].iloc[[1]], dfs_S[i*Ncol + j].iloc[[2]], colors = colorsopt)
    for j in range(Ncol):
        axes[0, j].set_title(xcommonname.format(xlist[j]))
    for i in range(Nrow):
        axes[i, 0].set_ylabel(ycommonname.format(ylist[i]))
    f.text(0.5, 0.04, 'Time horizon (month)', ha='center')
    f.text(0.06, 0.5, 'Number of patients in illness groups', va='center', rotation='vertical')
    plt.figlegend( handles=[patch1, patch2, patch3],
               labels=('Illness group I', 'Illness group II', 'Illness group III'),
               loc='upper center', ncol=3)
    plt.savefig(outputfilename1 + 'statebytime' + outputfilename2 + '.png', dpi = dpinum)
    #plt.show()
    # y
    f, axes = plt.subplots(Nrow, Ncol, sharey=True, sharex=True, figsize=(15,10))
    for i in range(Nrow):
        for j in range(Ncol):
            #sumy = dfs_y[i*4 + j].sum()
            if firstperiodsflag:
                axes[i, j].stackplot(range(numperiods), dfs_y[i*Ncol + j].iloc[0,:numperiods]/sumy[i*Ncol + j], 
                        dfs_y[i*Ncol + j].iloc[1,:numperiods]/sumy[i*Ncol + j], dfs_y[i*Ncol + j].iloc[2,:numperiods]/sumy[i*Ncol + j], colors = colorsopt)
            else:
                axes[i, j].stackplot(range(len(dfs_y[i*Ncol + j].columns)), dfs_y[i*Ncol + j].iloc[[0]]/sumy[i*Ncol + j], 
                        dfs_y[i*Ncol + j].iloc[[1]]/sumy[i*Ncol + j], dfs_y[i*Ncol + j].iloc[[2]]/sumy[i*Ncol + j], colors = colorsopt)
    for j in range(Ncol):
        axes[0, j].set_title(xcommonname.format(xlist[j]))
    for i in range(Nrow):
        axes[i, 0].set_ylabel(ycommonname.format(ylist[i]))
        vals = axes[i, 0].get_yticks()
        axes[i, 0].set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
    f.text(0.5, 0.04, 'Time horizon (month)', ha='center')
    f.text(0.04, 0.5, 'Percentage of resource allocated to illness groups', va='center', rotation='vertical')
    plt.figlegend( handles=[patch1, patch2, patch3],
               labels=('Illness group I', 'Illness group II', 'Illness group III'),
               loc='upper center', ncol=3)
    plt.savefig(outputfilename1 + 'ybytime' + outputfilename2 + '.png', dpi = dpinum)
    #plt.show()

# Table 2   
K = 1
capacity = [16,32,48,64,128]
distribution = [1,2,4]
commonname = "Long_term_discounted_2hr_"
scenariolist = [23,22,314,21,20,19,18,315,17,16,15,14,316,13,12]
sumy = [16*16,32*16,48*16,64*16,128*16]*3
outputfilename = '0_long_term_discounted_003_y_S.csv'
generate_outcome_tables(commonname, scenariolist, K, capacity, distribution, sumy, outputfilename)
# Table 4
outputfilename = '0_long_term_discounted_003_gap_time.csv'
generate_technicalnotes_tables(commonname, scenariolist, capacity, distribution, outputfilename)

dpinum = 600
outputfilename1 = 'oneschool_'
outputfilename2 = '_discounted003'
capacity = [16,32,48,64]
scenariolist = [23,22,314,21,19,18,315,17,15,14,316,13]
sumy = [16*16,32*16,48*16,64*16]*3
xcommonname = 'C={}'
ycommonname = 'Initial distribution {}'
firstperiodsflag = False
numperiods = 10
# all periods
generate_plots(commonname, scenariolist, K, xcommonname, capacity, ycommonname, distribution, sumy, 
               firstperiodsflag, numperiods, outputfilename1, outputfilename2, dpinum)
# first 30 periods
outputfilename2 = '_discounted003_1st30'
firstperiodsflag = True
numperiods = 30
generate_plots(commonname, scenariolist, K, xcommonname, capacity, ycommonname, distribution, sumy, 
               firstperiodsflag, numperiods, outputfilename1, outputfilename2, dpinum)
               
# Tables Appendix
K = 1
capacity = [16,32,48,64,128]
distribution = [1,2,4]
commonname = "Long_term_2hr_"
scenariolist = [23,22,314,21,20,19,18,315,17,16,15,14,316,13,12]
sumy = [16*16,32*16,48*16,64*16,128*16]*3
outputfilename = '0_long_term_no_discounting_y_S.csv'
generate_outcome_tables(commonname, scenariolist, K, capacity, distribution, sumy, outputfilename)
outputfilename = '0_long_term_no_discounting_gap_time.csv'
generate_technicalnotes_tables(commonname, scenariolist, capacity, distribution, outputfilename)

dpinum = 600
outputfilename1 = 'oneschool_'
outputfilename2 = ''
capacity = [16,32,48,64]
scenariolist = [23,22,314,21,19,18,315,17,15,14,316,13]
sumy = [16*16,32*16,48*16,64*16]*3
xcommonname = 'C={}'
ycommonname = 'Initial distribution {}'
firstperiodsflag = False
numperiods = 10
generate_plots(commonname, scenariolist, K, xcommonname, capacity, ycommonname, distribution, sumy, 
               firstperiodsflag, numperiods, outputfilename1, outputfilename2, dpinum)
               
# 89 schools
K = 89
capacity = [16,32,48]
decisionperiods = [6,12,18]
commonname = "T_C_2hr_"
# initial distribution 4
scenariolist = [24,27,30,25,28,31,26,29,32]
sumy = [16*16]*3 + [32*16]*3 + [48*16]*3
outputfilename = '0_89schools_dist4_y_S.csv'
generate_outcome_tables(commonname, scenariolist, K, decisionperiods, capacity, sumy, outputfilename)
outputfilename = '0_89schools_dist4_gap_time.csv'
generate_technicalnotes_tables(commonname, scenariolist, decisionperiods, capacity, outputfilename)
oneschoolscenariolist = [239,242,245,240,243,246,241,244,247]
outputfilename = '0_89schools_dist4_89and1school.csv'
generate_tn_89and1school_tables(commonname, scenariolist, oneschoolscenariolist, decisionperiods, capacity, outputfilename)
# plot dist 4
dpinum = 100
outputfilename1 = '89school_'
outputfilename2 = '_dist4'
scenariolist = [24,25,26,27,28,29,30,31,32] # a different layout, switch x axis and y axis
sumy = [16*16,32*16,48*16]*3
xcommonname = 'C={}'
ycommonname = 'T={}'
firstperiodsflag = False
numperiods = 10
generate_plots(commonname, scenariolist, K, xcommonname, capacity, ycommonname, decisionperiods, sumy, 
               firstperiodsflag, numperiods, outputfilename1, outputfilename2, dpinum)
               
# initial distribution 1
scenariolist = [56,59,62,57,60,63,58,61,64]
sumy = [16*16]*3 + [32*16]*3 + [48*16]*3
outputfilename = '0_89schools_dist1_y_S.csv'
generate_outcome_tables(commonname, scenariolist, K, decisionperiods, capacity, sumy, outputfilename)
outputfilename = '0_89schools_dist1_gap_time.csv'
generate_technicalnotes_tables(commonname, scenariolist, decisionperiods, capacity, outputfilename)
oneschoolscenariolist = [269,272,275,270,273,276,271,274,277]
outputfilename = '0_89schools_dist1_89and1school.csv'
generate_tn_89and1school_tables(commonname, scenariolist, oneschoolscenariolist, decisionperiods, capacity, outputfilename)
               
# initial distribution 2
scenariolist = [39,42,45,40,43,46,41,44,47]
sumy = [16*16]*3 + [32*16]*3 + [48*16]*3
outputfilename = '0_89schools_dist2_y_S.csv'
generate_outcome_tables(commonname, scenariolist, K, decisionperiods, capacity, sumy, outputfilename)
outputfilename = '0_89schools_dist2_gap_time.csv'
generate_technicalnotes_tables(commonname, scenariolist, decisionperiods, capacity, outputfilename)
oneschoolscenariolist = [254,257,260,255,258,261,256,259,262]
outputfilename = '0_89schools_dist2_89and1school.csv'
generate_tn_89and1school_tables(commonname, scenariolist, oneschoolscenariolist, decisionperiods, capacity, outputfilename)

# initial distribution 3
scenariolist = [299,302,305,300,303,306,301,304,307]
sumy = [16*16]*3 + [32*16]*3 + [48*16]*3
outputfilename = '0_89schools_dist3_y_S.csv'
generate_outcome_tables(commonname, scenariolist, K, decisionperiods, capacity, sumy, outputfilename)
outputfilename = '0_89schools_dist3_gap_time.csv'
generate_technicalnotes_tables(commonname, scenariolist, decisionperiods, capacity, outputfilename)
oneschoolscenariolist = [284,287,290,285,288,291,286,289,292]
outputfilename = '0_89schools_dist3_89and1school.csv'
generate_tn_89and1school_tables(commonname, scenariolist, oneschoolscenariolist, decisionperiods, capacity, outputfilename)

# initial distribution 5
scenariolist = [317,320,323,318,321,324,319,322,325]
sumy = [16*16]*3 + [32*16]*3 + [48*16]*3
outputfilename = '0_89schools_dist5_y_S.csv'
generate_outcome_tables(commonname, scenariolist, K, decisionperiods, capacity, sumy, outputfilename)
outputfilename = '0_89schools_dist5_gap_time.csv'
generate_technicalnotes_tables(commonname, scenariolist, decisionperiods, capacity, outputfilename)
oneschoolscenariolist = [332,335,338,333,336,339,334,337,340]
outputfilename = '0_89schools_dist5_89and1school.csv'
generate_tn_89and1school_tables(commonname, scenariolist, oneschoolscenariolist, decisionperiods, capacity, outputfilename)
