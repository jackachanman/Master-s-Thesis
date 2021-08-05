 #-*- coding: utf-8 -*-
"""
Created on Wed Feb 06 10:58:02 2019

@author: jacky
"""
""" 2019-04-18 UPDATE: 'Base' scenario to 'Basic' scenario. final demand scenarios
from 10, 45, 90, to 2.5,5,10 """

""" gva_rr_aij needs to be changed """ 


""" 2019-05-23 deleted some useless code""" 

""" 2019-07-03 : excluding households from total output, energy, and job calcs """
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy import stats
from scipy.interpolate import griddata
from matplotlib import cm
import scipy.interpolate as interp
import copy
from matplotlib.patches import Patch



dx = {}
dx_r = {}
for ind in ar:
    if ind != 'Existing':  
        dx[ind] = {}
        for s in scenario:
            dx[ind][s] = pd.DataFrame(
                    index = pd.concat([industry_hr,industry_hr],axis=0),)
            dx[ind][s]['Total Industry Output (x1000)'] =\
                                x_r_r_irio[ind][s] - x_r_r_irio['Existing']  
            dx[ind][s]['Total Industry Output (%)'] =100*(x_r_r_irio[ind][s] - \
                                x_r_r_irio['Existing'] )/x_r_r_irio['Existing'] 
            dx[ind][s]['Total Industry Energy Use (TJ)'] = \
            x_r_r_irio_energy[ind][s].values - x_r_r_irio_energy['Existing'].values
            dx[ind][s]['Total Industry Energy Use (%)'] =\
                                100*(x_r_r_irio_energy[ind][s].values -\
                                x_r_r_irio_energy['Existing'].values)/\
                                     x_r_r_irio_energy['Existing'].values
                                     
for ind in dx:
    dx_r[ind] = {}
    for s in scenario:
        dx_r[ind][s] = {}
        IRIO_split(dx_r[ind][s],dx[ind][s])        
                          
""" CALCULATE AGGREGATE DIFFERENCES """         
# calculate aggregate 
s_d = pd.read_excel('summary_detail_concordance_matrix.xlsx', index_col=0)
s_d = s_d.fillna(0)
s_d = s_d.T                                                                     # concordance matrix (33x227)
s_d_irio = np.vstack([s_d.values,np.zeros_like(s_d)])
s_d_irio = np.hstack([s_d_irio,np.vstack([np.zeros_like(s_d),s_d.values])])     # concatanting so that concordance matrix can be applied to IRIO dimensions
ind_summ =pd.Series(pd.read_csv('Input_SUIC_S.csv').iloc[:,0])

def IRIO_split_summ(new_dict, irio_var) :
    new_dict['Rest of ON'] = irio_var.iloc[0:2*len(ind_summ)/len(A_r_r['Existing']),:]
    new_dict['RoW'] = irio_var.iloc[2*len(ind_summ)/len(A_r_r['Existing']):2*len(ind_summ),:] 


x_r_r_irio_sum = {}
x_r_r_irio_energy_sum = {}
for ind in ar:
    x_r_r_irio_sum[ind] = {}
    x_r_r_irio_energy_sum[ind] = {}
    if ind == 'Existing':
        x_r_r_irio_sum[ind] = s_d_irio * x_r_r_irio[ind]
        x_r_r_irio_energy_sum[ind] = s_d_irio* np.matrix(x_r_r_irio_energy[ind].values)
    elif ind != 'Existing':
        for s in scenario:
            x_r_r_irio_sum[ind][s] = s_d_irio * x_r_r_irio[ind][s]
            x_r_r_irio_energy_sum[ind][s] = s_d_irio* np.matrix(x_r_r_irio_energy[ind][s])
agg_dx = {}
agg_dx_r = {}
for ind in ar:
    if ind != 'Existing':
        agg_dx[ind] = {}
        agg_dx_r[ind] = {}
        for s in scenario:
            agg_dx[ind][s] = pd.DataFrame(index = pd.concat([ind_summ,ind_summ],axis=0),)
            agg_dx[ind][s]['Total Industry Output (x1000)'] = x_r_r_irio_sum[ind][s] - x_r_r_irio_sum['Existing']  
            agg_dx[ind][s]['Total Industry Output (%)'] = 100*(x_r_r_irio_sum[ind][s] - x_r_r_irio_sum['Existing']  )/x_r_r_irio_sum['Existing']
            agg_dx[ind][s]['Total Industry Energy Use (TJ)'] = x_r_r_irio_energy_sum[ind][s] - x_r_r_irio_energy_sum['Existing']
            agg_dx[ind][s]['Total Industry Energy Use (%)'] = 100*(x_r_r_irio_energy_sum[ind][s] - x_r_r_irio_energy_sum['Existing'])/x_r_r_irio_energy_sum['Existing']
            agg_dx_r[ind][s] = {}       
#            agg_dx_r[ = {}
            IRIO_split_summ(agg_dx_r[ind][s],agg_dx[ind][s])
            
""" CALCULATE EMPLOYMENT CHANGES """
# employment data has their own aggregation of industries. make concordance matrix from excel file
e_a_d = pd. read_excel('emp_agg_detail_concordance_matrix.xlsx',index_col=0)
e_a_d = e_a_d.fillna(0)
e_a_d = e_a_d.T
e_a_d_irio = np.vstack([e_a_d.values,np.zeros_like(e_a_d)])
e_a_d_irio = np.hstack([e_a_d_irio, np.vstack([np.zeros_like(e_a_d),e_a_d.values])]) # concatenating so that concordance matrix can be applied to IRIO dimensions
ind_emp_agg = pd.Series(pd.read_csv('Employment_aggregated_industries.csv').iloc[:,0])
 
def IRIO_split_emp(new_dict, irio_var) :
    new_dict['Rest of ON'] = irio_var.iloc[0:2*len(ind_emp_agg)/len(A_r_r['Existing']),:]
    new_dict['RoW'] = irio_var.iloc[2*len(ind_emp_agg)/len(A_r_r['Existing']):2*len(ind_emp_agg),:] 
 
# irio output aggregated to the level of employment data
x_r_r_irio_emp = {}
x_r_r_irio_energy_emp = {}
for ind in ar:
    x_r_r_irio_emp[ind] = {}
    x_r_r_irio_energy_emp[ind] = {}
    if ind == 'Existing':
        x_r_r_irio_emp[ind] = e_a_d_irio * x_r_r_irio[ind]
        x_r_r_irio_energy_emp[ind] = e_a_d_irio* np.matrix(x_r_r_irio_energy[ind].values)
    elif ind != 'Existing':
        for s in scenario:
            x_r_r_irio_emp[ind][s] = e_a_d_irio * x_r_r_irio[ind][s]
            x_r_r_irio_energy_emp[ind][s] = e_a_d_irio* np.matrix(x_r_r_irio_energy[ind][s])
                     
""" Calculate employment intensities and job changes"""
temp = {}
#temp2 = {}
#job_tot = {}                                                                    # total # of jobs for the two regions
#tot_emp_int = {}                                                                 # job_tot divide by total output of each region
#emp_int = {}
#test = {}
for row in A_r_r['Existing']:
#    temp[row] =employ.loc[:,row].values.reshape((len(ind_emp_agg)-1,1))         # minus one is necessary because employment data does not have an industry for 'household labour'
#    temp[row] = np.vstack([temp[row],[0]])                                      # add zero to act as filler for employment data for 'household labour'
    temp[row] = copy.deepcopy(employ.loc[:,row].values.reshape((len(ind_emp_agg),1))  )        # closed model
    # household intensities should be zero
    temp[row][-1] = 0
#    temp2[row] = temp[row]/(x_rr_i['Existing'][row].values.sum(axis=0)-x_rr_i['Existing'][row].loc['Household labour','Total Industry Output'])
#    job_tot[row] = temp[row].sum(axis=0)
#    tot_emp_int[row] = job_tot[row]/(x_rr_i['Existing'][row].values.sum(axis=0)-x_rr_i['Existing'][row].loc['Household Labour','Total Industry Output'])
#    test[row] = temp2[row].sum(axis=0) - tot_emp_int[row]

#emp_int =  np.concatenate((temp2['Rest of ON'],temp2['RoW']),axis=0) 
emp_int =  np.concatenate((temp['Rest of ON'],temp['RoW']),axis=0) 
emp_int = np.divide(emp_int, x_r_r_irio_emp['Existing'])
#test = np.concatenate((temp['Rest of ON'],temp['RoW']),axis=0) 
#test = np.linalg.inv(np.diagflat(x_r_r_irio_emp['Existing']))*np.matrix(np.concatenate((temp['Rest of ON'],temp['RoW']),axis=0))
#test2 = emp_int - test
#emp_int = emp_int/x_rr_i['Existing'][row].values.sum(axis=0)

del temp

emp_int_r = {}
temp = pd.DataFrame(emp_int, index = pd.concat([ind_emp_agg,ind_emp_agg],axis=0),columns =['Job intensity (jobs/$ output)'])
for row in A_r_r['Existing']:
    emp_int_r[row] = pd.DataFrame(index = ind_emp_agg, columns =['Job intensity (jobs/$ output)'])
IRIO_split_emp(emp_int_r, temp)
del temp

emp_dx = {}                                                                     # dx_r aggregated to classification that employment is in
emp_dx_r = {}
for ind in ar:
    if ind != 'Existing':
        emp_dx[ind] = {}
        emp_dx_r[ind] = {}        
        for s in scenario:
            emp_dx[ind][s] = pd.DataFrame(index = pd.concat([ind_emp_agg,ind_emp_agg],axis=0),)
            emp_dx[ind][s]['Total Industry Output (x1000)'] = x_r_r_irio_emp[ind][s] - x_r_r_irio_emp['Existing']  
            emp_dx[ind][s]['Total Industry Output (%)'] = 100*(x_r_r_irio_emp[ind][s] - x_r_r_irio_emp['Existing']  )/x_r_r_irio_emp['Existing']
            emp_dx[ind][s]['Total Industry Energy Use (TJ)'] = x_r_r_irio_energy_emp[ind][s] - x_r_r_irio_energy_emp['Existing']
            emp_dx[ind][s]['Total Industry Energy Use (%)'] = 100*(x_r_r_irio_energy_emp[ind][s] - x_r_r_irio_energy_emp['Existing'])/x_r_r_irio_energy_emp['Existing']
            emp_dx_r[ind][s] = {}       
#            emp_dx_r = {}
            IRIO_split_emp(emp_dx_r[ind][s],emp_dx[ind][s])

job_dx = {}
job_dx_r = {}
job_dx_r_tot = {}
#test = {}
for ind in ar:
    if ind != 'Existing':
        job_dx[ind] = {}
        job_dx_r[ind] = {}
        job_dx_r_tot[ind]  = {}
#        test[ind] = {}
        for s in scenario:
            job_dx[ind][s] = pd.DataFrame(index = pd.concat([ind_emp_agg,ind_emp_agg],axis =0),)
            job_dx[ind][s]['Total Job Changes'] = 1000*np.multiply(emp_int,
                     emp_dx[ind][s]['Total Industry Output (x1000)'].values.reshape(2*len(ind_emp_agg),1))
#            job_dx[ind][s]= 1000*np.diagflat(emp_int)*np.matrix(emp_dx[ind][s]['Total Industry Output (x1000)'].values).T
#            test[ind][s] = emp_dx[ind][s]['Total Industry Output (x1000)'].values.reshape(2*len(ind_emp_agg),1)
#            job_dx[ind][s]['Total Job Changes'] = job_dx[ind][s]['Total Job Changes'] .astype(float).round(0)
            job_dx_r[ind][s] = {}
            IRIO_split_emp(job_dx_r[ind][s], job_dx[ind][s])
            
            job_dx_r_tot[ind][s]= {}
            for row in A_r_r['Existing']:
#                job_dx_r[ind][s][row] = 1000* np.multiply(emp_dx_r[ind][s][row]['Total Industry Output (x1000)'].values.reshape(len(ind_emp_agg),1) , temp2[row])
                job_dx_r_tot[ind][s][row] = float(job_dx_r[ind][s][row].values.sum(axis = 0))
                
""" Testing industry output changes """
delta_output = {}
delta_energy = {}
#test = {}
delta_output_perc= {}
beta_output = {}
for ind in ar:
    if ind != 'Existing':
        delta_output[ind] = {}
        delta_energy[ind] = {}
#        test[ind] = {}
        delta_output_perc[ind]= {}
        beta_output[ind] = {}
        for s in scenario:
            delta_output[ind][s] = {}
            delta_energy[ind][s] = {}
            delta_output_perc[ind][s]= {}
            beta_output[ind][s] = {}
#            test[ind][s] = {}
            for row in A_r_r['Existing']:
                # dont include households
                delta_output[ind][s][row]  = dx_r[ind][s][row]['Total Industry Output (x1000)'][:-1].sum(axis=0) 
                # dont include households
                delta_energy[ind][s][row] = dx_r[ind][s][row]['Total Industry Energy Use (TJ)'][:-1].sum(axis=0)
#                test[ind][s][row] = emp_dx_r[ind][s][row]['Total Industry Output (x1000)'].sum(axis=0) - delta_output[ind][s][row]
                delta_output_perc[ind][s][row]= delta_output[ind][s][row]/x_r_r_irio['Existing'][:-1].sum(axis=0)
                beta_output[ind][s][row] =  float(x_rr_i[ind][s][row].sum(axis=0))
#""" test """
#test_job = {}
#for ind in ar:
#    if ind != 'Existing':
#        test_job[ind] = {}
#        for s in scenario:
#            test_job[ind][s] = {}
#            for row in A_r_r['Existing']:
#                test_job[ind][s][row] = 1000*tot_emp_int[row]*(delta_output[ind][s][row] - dx_r[ind][s][row].loc['Household labour','Total Industry Output (x1000)'])
#test_job_diff = {}
#for ind in ar:
#    if ind != 'Existing':
#        test_job_diff[ind] = {}
#        for s in scenario:
#            test_job_diff[ind][s] = {}
#            for row in A_r_r['Existing']:
#                test_job_diff[ind][s][row] = test_job[ind][s][row] - job_dx_r_tot[ind][s][row]
                
""" PLOTTING GDP for RoW"""

""" calc changes in Household column and final demand vector"""

#a = 'Household final consumption expenditure'
a = 'Household'
# existing GDP
GDP = {}
for ind in ar:
    GDP[ind] = {}
    if ind != 'Existing':
        for s in scenario:
            GDP[ind][s] = {}
            for row in A_r_r['Existing']:
#                GDP[ind][s][row] = float(Z_row_col[ind][s][row][row].loc[:,a].sum(axis=0) +\
#                   f_rr[ind][s][row].sum(axis =0))
                GDP[ind][s][row] = i_gva*(v_coeff[ind][s]*np.diagflat(x_rr[ind][s][row]))*i_hh
    elif ind == 'Existing':
        for row in A_r_r['Existing']:
#            GDP[ind][row] = float(Z_row_col['Existing'][row][row].loc[:,a].sum(axis=0) +\
#                    f_rr['Existing'][row].sum(axis=0)) 
            GDP[ind][row] = i_gva*(v_coeff[ind]*np.diagflat(x_rr[ind][row]))*i_hh
                    
GDP_change = {}

for ind in ar:
    GDP_change[ind] = {}
    if ind != 'Existing':
        for s in scenario:
            GDP_change[ind][s] = {}
            for row in A_r_r['Existing']:
                GDP_change[ind][s][row] = pd.DataFrame(index = ['GDP Change'],
                                                  columns = ['Total Value','Percent'])
                GDP_change[ind][s][row]['Total Value'] = GDP[ind][s][row]-GDP['Existing'][row]
                GDP_change[ind][s][row]['Percent'] = 100*(GDP[ind][s][row]-GDP['Existing'][row])/GDP['Existing'][row]
                
                
"""PLOTTING - scenarios with no final demand changes
"""                 
x_axis = ['Basic', '10%','45%','90%']
x_fake = [0,10,45,90]
x_fd = [0,2.5,5,10]

""" Aggregated data Plot
"""

for ind in dx:
    for s in scenario:
        f = plt.figure(figsize=(15,5))
        agg_dx_r[ind][s]['RoW']['Total Industry Output (%)'].plot(kind ='bar',legend = False)
#        agg_dx_r[ind][s]['RoW'].plot(kind = 'bar', y = 'Total Industry Output (%)', legend = False)
        plt.xticks(rotation = 30 , ha = 'right')
        plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
        plt.ylabel('Percent change, (%)', fontsize = 12, fontweight ='bold')
        plt.title('RoW ' +ind +' : \nChanges in output for aggregated industries \n\''+s+'\' scenario', fontsize = 14, fontweight = 'bold')
        plt.grid(linestyle = '--', linewidth = 0.6)
        plt.show()
  

""" Plotting changes in wages by $"""
y1 = []
y2 = []
for s in x_axis:
#    y1.append(dx_r['Residential building construction'][s]['RoW']\
#                  .loc['Household labour','Total Industry Output (x1000)'])
#    y2.append(dx_r['Non-residential building construction'][s]['RoW']\
#                  .loc['Household labour','Total Industry Output (x1000)'])
    y1.append(dx_r['Residential building construction'][s]['RoW']\
                  .loc['Household','Total Industry Output (x1000)'])
    y2.append(dx_r['Non-residential building construction'][s]['RoW']\
                  .loc['Household','Total Industry Output (x1000)'])

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x_fake,y1)
line1 = slope1*np.asarray(x_fake)+intercept1

slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x_fake,y2)
line2 = slope2*np.asarray(x_fake)+intercept2

fig= plt.figure()        
# plot linear regression
plt.plot( x_fake, line1,marker = 's', c='C0', label= 'Residential: wages,' + ' y={:.2f}x+{:.2f}'.format(slope1,intercept1))
plt.plot(x_fake, line2,marker = 'o', c ='C1',label='Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope2,intercept2))
plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.35))
plt.title('RoW: Total changes in wages by $', fontweight = 'bold')
plt.xlabel('Percentage of non-structural components reused', fontweight = 'bold')
plt.ylabel('$CAD (x1000)', fontweight = 'bold')
plt.grid()
plt.show()

""" Plotting changes in wages by %"""
y3 = []
y4 = []
y5 = []
y6 = []


for s in x_axis:
#    y3.append(dx_r['Residential building construction'][s]['RoW']\
#                  .loc['Household labour','Total Industry Output (%)'])
#    y4.append(dx_r['Non-residential building construction'][s]['RoW']\
#                  .loc['Household labour','Total Industry Output (%)'])
#    
#    y5.append(dx_r['Residential building construction'][s]['Rest of ON']\
#                  .loc['Household labour','Total Industry Output (%)'])
#    y6.append(dx_r['Non-residential building construction'][s]['Rest of ON']\
#                  .loc['Household labour','Total Industry Output (%)'])
    y3.append(dx_r['Residential building construction'][s]['RoW']\
                  .loc['Household','Total Industry Output (%)'])
    y4.append(dx_r['Non-residential building construction'][s]['RoW']\
                  .loc['Household','Total Industry Output (%)'])
    
    y5.append(dx_r['Residential building construction'][s]['Rest of ON']\
                  .loc['Household','Total Industry Output (%)'])
    y6.append(dx_r['Non-residential building construction'][s]['Rest of ON']\
                  .loc['Household','Total Industry Output (%)'])


slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(x_fake,y3)
line3 = slope3*np.asarray(x_fake)+intercept3

slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(x_fake,y4)
line4 = slope4*np.asarray(x_fake)+intercept4

slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(x_fake,y5)
line5 = slope5*np.asarray(x_fake)+intercept5

slope6, intercept6, r_value6, p_value6, std_err6 = stats.linregress(x_fake,y6)
line6 = slope6*np.asarray(x_fake)+intercept6

fig= plt.figure()        

# plot linear regression
plt.plot( x_fake, line3,marker = 's', c='C0', label= 'RoW: Residential: wages,' + ' y={:.4f}x+{:.4f}'.format(slope3,intercept3))
plt.plot(x_fake, line4,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.4f}x+{:.4f}'.format(slope4,intercept4))
plt.plot( x_fake, line5,marker = 's', c='C2', label= 'Rest of ON: Residential: wages,' + ' y={:.4f}x+{:.4f}'.format(slope5,intercept5))
plt.plot(x_fake, line6,marker = 'o', c ='C3',label='Rest of ON: Non-residential: wages,'+' y={:.4f}x+{:.4f}'.format(slope6,intercept6))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.5))
plt.title('RoW: Total changes in wages by %', fontweight = 'bold')
plt.xlabel('Percentage of non-structural components reused', fontweight = 'bold')
plt.ylabel('%', fontweight = 'bold')
plt.grid()
plt.show()

  
"""plotting changes to output """
y9 = []
y10 = []
for s in x_axis:
    y9.append(delta_output['Residential building construction'][s]['RoW'])
    y10.append(delta_output['Non-residential building construction'][s]['RoW'])

slope9, intercept9, r_value9, p_value9, std_err9 = stats.linregress(x_fake,y9)
line9 = slope9*np.asarray(x_fake)+intercept9

slope10, intercept10, r_value10, p_value10, std_err10 = stats.linregress(x_fake,y10)
line10 = slope10*np.asarray(x_fake)+intercept10

fig= plt.figure()        

# plot linear regression
plt.plot( x_fake, line9,marker = 's', c='C0', label= 'RoW: Residential: wages,' + ' y={:.2f}x+{:.2f}'.format(slope9,intercept9))
#plt.plot(x_fake, line10,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope10,intercept4))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.5))
plt.title('RoW: Total changes in output by $', fontweight = 'bold')
plt.xlabel('Percentage of non-structural components reused', fontweight = 'bold')
plt.ylabel('%', fontweight = 'bold')
plt.grid()
plt.show()


""" PLOTTING - scenarioes with final demand changes"""


x_f_basic = ['Basic','10%','45%','90%']
x_f_2_5 = ['f2.5_basic','f2.5_10','f2.5_45','f2.5_90']
x_f_5 = ['f5_basic','f5_10','f5_45','f5_90']
x_f_10 = ['f10_basic','f10_10','f10_45','f10_90']

""" Residential plotting changes in output versus % structural reuse"""
y11 = []
y12 = []
y13 = []
y14 = []

for s in x_f_basic:
    y11.append(delta_output['Residential building construction'][s]['RoW'])

for s in x_f_2_5:
    y12.append(delta_output['Residential building construction'][s]['RoW'])
for s in x_f_5:
    y13.append(delta_output['Residential building construction'][s]['RoW'])
for s in x_f_10:    
    y14.append(delta_output['Residential building construction'][s]['RoW'])
    
slope11, intercept11, r_value11, p_value11, std_err11 = stats.linregress(x_fake,y11)
line11 = slope11*np.asarray(x_fake)+intercept11

slope12, intercept12, r_value12, p_value12, std_err12 = stats.linregress(x_fake,y12)
line12 = slope12*np.asarray(x_fake)+intercept12

slope13, intercept13, r_value13, p_value13, std_err13 = stats.linregress(x_fake,y13)
line13 = slope13*np.asarray(x_fake)+intercept13

slope14, intercept14, r_value14, p_value14, std_err14 = stats.linregress(x_fake,y14)
line14 = slope14*np.asarray(x_fake)+intercept14


fig= plt.figure()        

# plot linear regression
plt.plot( x_fake, line11,marker = 's', c='C0', label= 'Basic scenario,' + ' y={:.2f}x+{:.2f}'.format(slope11,intercept11))
#plt.plot(x_fake, line10,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope10,intercept4))
plt.plot( x_fake, line12,marker = 's', c='C1', label= '10% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope12,intercept12))
plt.plot( x_fake, line13,marker = 's', c='C2', label= '45% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope13,intercept13))
plt.plot( x_fake, line14,marker = 's', c='C3', label= '90% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope14,intercept14))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6))
plt.title('RoW Residential building construction \n Total change in output in $', fontweight = 'bold')
plt.xlabel('Percentage of non-structural components reused', fontweight = 'bold')
plt.ylabel('Change in Output, ($)', fontweight = 'bold')
plt.grid()
plt.show()

""" Non-residential plotting: changes in output vs % structural reuse """
y15 = []
y16 = []
y17 = []
y18 = []

for s in x_f_basic:
    y15.append(delta_output['Non-residential building construction'][s]['RoW'])

for s in x_f_2_5:
    y16.append(delta_output['Non-residential building construction'][s]['RoW'])
for s in x_f_5:
    y17.append(delta_output['Non-residential building construction'][s]['RoW'])
for s in x_f_10:    
    y18.append(delta_output['Non-residential building construction'][s]['RoW'])
    
slope15, intercept15, r_value15, p_value15, std_err15 = stats.linregress(x_fake,y15)
line15 = slope15*np.asarray(x_fake)+intercept15

slope16, intercept16, r_value16, p_value16, std_err16 = stats.linregress(x_fake,y16)
line16 = slope16*np.asarray(x_fake)+intercept16

slope17, intercept17, r_value17, p_value17, std_err17 = stats.linregress(x_fake,y17)
line17 = slope17*np.asarray(x_fake)+intercept17

slope18, intercept18, r_value18, p_value18, std_err18 = stats.linregress(x_fake,y18)
line18 = slope18*np.asarray(x_fake)+intercept18


fig= plt.figure()        

# plot linear regression
plt.plot( x_fake, line15,marker = 's', c='C0', label= 'Basic scenario,' + ' y={:.2f}x+{:.2f}'.format(slope15,intercept15))
#plt.plot(x_fake, line10,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope10,intercept4))
plt.plot( x_fake, line16,marker = 's', c='C1', label= '10% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope16,intercept16))
plt.plot( x_fake, line17,marker = 's', c='C2', label= '45% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope17,intercept17))
plt.plot( x_fake, line18,marker = 's', c='C3', label= '90% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope18,intercept18))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6))
plt.title('RoW Non-residential building construction \n Total change in output in $', fontweight = 'bold')
plt.xlabel('Percentage of non-structural components reused', fontweight = 'bold')
plt.ylabel('Change in Output, ($)', fontweight = 'bold')
plt.grid()
plt.show()

""" Residential plotting changes in output versus % final demand"""

x_f_basic = ['Basic','10%','45%','90%']
x_f_2_5 = ['f2.5_basic','f2.5_10','f2.5_45','f2.5_90']
x_f_5 = ['f5_basic','f5_10','f5_45','f5_90']
x_f_10 = ['f10_basic','f10_10','f10_45','f10_90']


y19 = []
y20 = []
y21 = []
y22 = []

for s in x_f_basic:
    y19.append(delta_output['Residential building construction'][s]['RoW'])

for s in x_f_2_5:
    y20.append(delta_output['Residential building construction'][s]['RoW'])
for s in x_f_5:
    y21.append(delta_output['Residential building construction'][s]['RoW'])
for s in x_f_10:    
    y22.append(delta_output['Residential building construction'][s]['RoW'])
    
slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(x_fd,y19)
line19 = slope19*np.asarray(x_fd)+intercept19

slope20, intercept20, r_value20, p_value20, std_err20 = stats.linregress(x_fd,y20)
line20 = slope20*np.asarray(x_fd)+intercept20

slope21, intercept21, r_value21, p_value21, std_err21 = stats.linregress(x_fd,y21)
line21 = slope21*np.asarray(x_fd)+intercept21

slope22, intercept22, r_value22, p_value22, std_err22 = stats.linregress(x_fd,y22)
line22 = slope22*np.asarray(x_fd)+intercept22


fig= plt.figure()        

# plot linear regression
plt.plot( x_fd, line19,marker = 's', c='C0', label= 'Basic scenario,' + ' y={:.2f}x+{:.2f}'.format(slope19,intercept19))
#plt.plot(x_fd, line10,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope10,intercept4))
plt.plot( x_fd, line20,marker = 's', c='C1', label= '10% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope20,intercept20))
plt.plot( x_fd, line21,marker = 's', c='C2', label= '45% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope21,intercept21))
plt.plot( x_fd, line22,marker = 's', c='C3', label= '90% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope22,intercept22))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6))
plt.title('RoW Residential building construction \n Total change in output in $', fontweight = 'bold')
plt.xlabel('Percent increase in final demand for \nresidential building construction', fontweight = 'bold')
plt.ylabel('Change in Output, ($)', fontweight = 'bold')
plt.grid()
plt.show()

""" Non-residential plotting: changes in output vs % final demand """
y23 = []
y24 = []
y25 = []
y26 = []

for s in x_f_basic:
    y23.append(delta_output['Non-residential building construction'][s]['RoW'])

for s in x_f_2_5:
    y24.append(delta_output['Non-residential building construction'][s]['RoW'])
for s in x_f_5:
    y25.append(delta_output['Non-residential building construction'][s]['RoW'])
for s in x_f_10:    
    y26.append(delta_output['Non-residential building construction'][s]['RoW'])
    
slope23, intercept23, r_value23, p_value23, std_err23 = stats.linregress(x_fd,y23)
line23 = slope23*np.asarray(x_fd)+intercept23

slope24, intercept24, r_value24, p_value24, std_err24 = stats.linregress(x_fd,y24)
line24 = slope24*np.asarray(x_fd)+intercept24

slope25, intercept25, r_value25, p_value25, std_err25 = stats.linregress(x_fd,y25)
line25 = slope25*np.asarray(x_fd)+intercept25

slope26, intercept26, r_value26, p_value26, std_err26 = stats.linregress(x_fd,y26)
line26 = slope26*np.asarray(x_fd)+intercept26


fig= plt.figure()        

# plot linear regression
plt.plot( x_fd, line23,marker = 's', c='C0', label= 'Basic scenario,' + ' y={:.2f}x+{:.2f}'.format(slope23,intercept23))
#plt.plot(x_fd, line10,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope10,intercept4))
plt.plot( x_fd, line24,marker = 's', c='C1', label= '10% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope24,intercept24))
plt.plot( x_fd, line25,marker = 's', c='C2', label= '45% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope25,intercept25))
plt.plot( x_fd, line26,marker = 's', c='C3', label= '90% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope26,intercept26))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6))
plt.title('RoW Non-residential building construction \n Total change in output in $', fontweight = 'bold')
plt.xlabel('Percent increase in final demand for \nnon-residential building construction', fontweight = 'bold')
plt.ylabel('Change in Output, ($)', fontweight = 'bold')
plt.grid()
plt.show()

            
""" Residential plotting changes in GDP versus % structural reuse"""
x_f_basic = ['Basic','10%','45%','90%']
x_f_2_5 = ['f2.5_basic','f2.5_10','f2.5_45','f2.5_90']
x_f_5 = ['f5_basic','f5_10','f5_45','f5_90']
x_f_10 = ['f10_basic','f10_10','f10_45','f10_90']

y11 = []
y12 = []
y13 = []
y14 = []

for s in x_f_basic:
    y11.append(GDP_change['Residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])

for s in x_f_2_5:
    y12.append(GDP_change['Residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
for s in x_f_5:
    y13.append(GDP_change['Residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
for s in x_f_10:    
    y14.append(GDP_change['Residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
    
slope11, intercept11, r_value11, p_value11, std_err11 = stats.linregress(x_fake,y11)
line11 = slope11*np.asarray(x_fake)+intercept11

slope12, intercept12, r_value12, p_value12, std_err12 = stats.linregress(x_fake,y12)
line12 = slope12*np.asarray(x_fake)+intercept12

slope13, intercept13, r_value13, p_value13, std_err13 = stats.linregress(x_fake,y13)
line13 = slope13*np.asarray(x_fake)+intercept13

slope14, intercept14, r_value14, p_value14, std_err14 = stats.linregress(x_fake,y14)
line14 = slope14*np.asarray(x_fake)+intercept14


fig= plt.figure()        

# plot linear regression
plt.plot( x_fake, line11,marker = 's', c='C0', label= 'Basic scenario,' + ' y={:.2f}x+{:.2f}'.format(slope11,intercept11))
#plt.plot(x_fake, line10,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope10,intercept4))
plt.plot( x_fake, line12,marker = 's', c='C1', label= '10% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope12,intercept12))
plt.plot( x_fake, line13,marker = 's', c='C2', label= '45% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope13,intercept13))
plt.plot( x_fake, line14,marker = 's', c='C3', label= '90% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope14,intercept14))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6))
plt.title('RoW Residential building construction \n Total change in GDP in $', fontweight = 'bold')
plt.xlabel('Percentage of non-structural components reused', fontweight = 'bold')
plt.ylabel('Change in GDP, ($)', fontweight = 'bold')
plt.grid()
plt.show()

""" Non-residential plotting: changes in GDP vs % structural reuse """
y15 = []
y16 = []
y17 = []
y18 = []

for s in x_f_basic:
    y15.append(GDP_change['Non-residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])

for s in x_f_2_5:
    y16.append(GDP_change['Non-residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
for s in x_f_5:
    y17.append(GDP_change['Non-residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
for s in x_f_10:    
    y18.append(GDP_change['Non-residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
    
slope15, intercept15, r_value15, p_value15, std_err15 = stats.linregress(x_fake,y15)
line15 = slope15*np.asarray(x_fake)+intercept15

slope16, intercept16, r_value16, p_value16, std_err16 = stats.linregress(x_fake,y16)
line16 = slope16*np.asarray(x_fake)+intercept16

slope17, intercept17, r_value17, p_value17, std_err17 = stats.linregress(x_fake,y17)
line17 = slope17*np.asarray(x_fake)+intercept17

slope18, intercept18, r_value18, p_value18, std_err18 = stats.linregress(x_fake,y18)
line18 = slope18*np.asarray(x_fake)+intercept18


fig= plt.figure()        

# plot linear regression
plt.plot( x_fake, line15,marker = 's', c='C0', label= 'Basic scenario,' + ' y={:.2f}x+{:.2f}'.format(slope15,intercept15))
#plt.plot(x_fake, line10,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope10,intercept4))
plt.plot( x_fake, line16,marker = 's', c='C1', label= '10% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope16,intercept16))
plt.plot( x_fake, line17,marker = 's', c='C2', label= '45% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope17,intercept17))
plt.plot( x_fake, line18,marker = 's', c='C3', label= '90% final demand increase,' + ' y={:.2f}x+{:.2f}'.format(slope18,intercept18))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6))
plt.title('RoW Non-residential building construction \n Total change in GDP in $', fontweight = 'bold')
plt.xlabel('Percentage of non-structural components reused', fontweight = 'bold')
plt.ylabel('Change in GDP, ($)', fontweight = 'bold')
plt.grid()
plt.show()


""" Residential plotting changes in GDP versus % final demand"""

x_f_basic = ['Basic','10%','45%','90%']
x_f_2_5 = ['f2.5_basic','f2.5_10','f2.5_45','f2.5_90']
x_f_5 = ['f5_basic','f5_10','f5_45','f5_90']
x_f_10 = ['f10_basic','f10_10','f10_45','f10_90']



y11 = []
y12 = []
y13 = []
y14 = []

for s in x_f_basic:
    y11.append(GDP_change['Residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])

for s in x_f_2_5:
    y12.append(GDP_change['Residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
for s in x_f_5:
    y13.append(GDP_change['Residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
for s in x_f_10:    
    y14.append(GDP_change['Residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
    
slope11, intercept11, r_value11, p_value11, std_err11 = stats.linregress(x_fd,y11)
line11 = slope11*np.asarray(x_fd)+intercept11

slope12, intercept12, r_value12, p_value12, std_err12 = stats.linregress(x_fd,y12)
line12 = slope12*np.asarray(x_fd)+intercept12

slope13, intercept13, r_value13, p_value13, std_err13 = stats.linregress(x_fd,y13)
line13 = slope13*np.asarray(x_fd)+intercept13

slope14, intercept14, r_value14, p_value14, std_err14 = stats.linregress(x_fd,y14)
line14 = slope14*np.asarray(x_fd)+intercept14


fig= plt.figure()        

# plot linear regression
plt.plot( x_fd, line11,marker = 's', c='C0', label= 'Basic scenario,' + ' y={:.2f}x+{:.2f}'.format(slope11,intercept11))
#plt.plot(x_fd, line10,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope10,intercept4))
plt.plot( x_fd, line12,marker = 's', c='C1', label= '10% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope12,intercept12))
plt.plot( x_fd, line13,marker = 's', c='C2', label= '45% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope13,intercept13))
plt.plot( x_fd, line14,marker = 's', c='C3', label= '90% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope14,intercept14))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6))
plt.title('RoW Residential building construction \n Total change in GDP in $', fontweight = 'bold')
plt.xlabel('Percent increase in final demand for \nresidential building construction', fontweight = 'bold')
plt.ylabel('Change in GDP, ($)', fontweight = 'bold')
plt.grid()
plt.show()

""" Non-residential plotting: changes in GDP vs % final demand """
y15 = []
y16 = []
y17 = []
y18 = []

for s in x_f_basic:
    y15.append(GDP_change['Non-residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])

for s in x_f_2_5:
    y16.append(GDP_change['Non-residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
for s in x_f_5:
    y17.append(GDP_change['Non-residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
for s in x_f_10:    
    y18.append(GDP_change['Non-residential building construction'][s]['RoW'].loc['GDP Change','Total Value'])
    
slope15, intercept15, r_value15, p_value15, std_err15 = stats.linregress(x_fd,y15)
line15 = slope15*np.asarray(x_fd)+intercept15

slope16, intercept16, r_value16, p_value16, std_err16 = stats.linregress(x_fd,y16)
line16 = slope16*np.asarray(x_fd)+intercept16

slope17, intercept17, r_value17, p_value17, std_err17 = stats.linregress(x_fd,y17)
line17 = slope17*np.asarray(x_fd)+intercept17

slope18, intercept18, r_value18, p_value18, std_err18 = stats.linregress(x_fd,y18)
line18 = slope18*np.asarray(x_fd)+intercept18


fig= plt.figure()        

# plot linear regression
plt.plot( x_fd, line15,marker = 's', c='C0', label= 'Basic scenario,' + ' y={:.2f}x+{:.2f}'.format(slope15,intercept15))
#plt.plot(x_fd, line10,marker = 'o', c ='C1',label='RoW: Non-residential: wages,'+' y={:.2f}x+{:.2f}'.format(slope10,intercept4))
plt.plot( x_fd, line16,marker = 's', c='C1', label= '10% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope16,intercept16))
plt.plot( x_fd, line17,marker = 's', c='C2', label= '45% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope17,intercept17))
plt.plot( x_fd, line18,marker = 's', c='C3', label= '90% reuse,' + ' y={:.2f}x+{:.2f}'.format(slope18,intercept18))


plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.6))
plt.title('RoW Non-residential building construction \n Total change in GDP in $', fontweight = 'bold')
plt.xlabel('Percent increase in final demand for \nnon-residential building construction', fontweight = 'bold')
plt.ylabel('Change in GDP, ($)', fontweight = 'bold')
plt.grid()
plt.show()


""" 3D PLOTS """

""" INDICATE THE REGION OF INTEREST """
region = 'RoW'                                                                  # RoW of Rest of ON
#region = 'Rest of ON'


# percent increase final demand
Y= {}
# percent non-structural component reuse
X = {}
# output
Z_3d = {}
Z_3d_gdp = {}
Z_3d_job = {}
Z_3d_energy = {}
Z_3d_0 = {}
""" for surface plot"""
# percent non structural component
xlist = np.asarray([0,10,45,90])
# percent increase in adaptive reuse cost (final demand increase prior to beta)
ylist = np.asarray([0,2.5,5,10])
zlist = {}
z_gdp = {}
z_job = {}
z_energy = {}
#z_wage = {}
z_0 = {}
""" Y  %increase in adaptive reuse cost (increase to final demand) """
""" if final demand value isnt multiplied by beta, its because of coding purposes"""
for ind in dx:
    Z_3d[ind] = []
    Z_3d_gdp[ind] =[]
    Z_3d_job[ind] = []
    Z_3d_energy[ind] = []
    Z_3d_0[ind] = []
    Y[ind] = []
    X[ind] = []
    zlist[ind] = pd.DataFrame(index = ['0','10','45','90'],
                                 columns = ['0','2.5','5','10'])
    z_gdp[ind] = pd.DataFrame(index = ['0','10','45','90'],
                                 columns = ['0','2.5','5','10'])
    
    z_job[ind] =  pd.DataFrame(index = ['0','10','45','90'],
                                 columns = ['0','2.5','5','10'])
    z_energy[ind] =  pd.DataFrame(index = ['0','10','45','90'],
                                 columns = ['0','2.5','5','10'])
#    z_wage[ind] = pd.DataFrame(index = ['0','10','45','90'],
#                                 columns = ['0','2.5','5','10'])
    z_0[ind] =  pd.DataFrame(index = ['0','10','45','90'],
                                 columns = ['0','2.5','5','10'])
    for s in scenario:
        if s == 'Basic':
            Y[ind].append(0)
            X[ind].append(0)
            zlist[ind].loc['0','0']=delta_output[ind][s][region]
            z_gdp[ind].loc['0','0']=GDP_change[ind][s][region].loc['GDP Change','Total Value']           
#            z_wage[ind].loc['0','0'] = dx_r[ind][s][region].loc['Household labour','Total Industry Output (%)']
            z_job[ind].loc['0','0'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['0','0']=delta_energy[ind][s][region]
            z_0[ind].loc['0','0'] = 0
        elif s == '10%':
            Y[ind].append(0)
            X[ind].append(10)
            zlist[ind].loc['0','10']= delta_output[ind][s][region]
            z_gdp[ind].loc['0','10']=GDP_change[ind][s][region].loc['GDP Change','Total Value']  
            z_job[ind].loc['0','10'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['0','10']=delta_energy[ind][s][region]
            z_0[ind].loc['0','10'] = 0
            
        elif s == '45%':
            Y[ind].append(0)
            X[ind].append(45)
            zlist[ind].loc['0','45']= delta_output[ind][s][region]
            z_gdp[ind].loc['0','45']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['0','45'] = job_dx_r_tot[ind][s][region]            
            z_energy[ind].loc['0','45']=delta_energy[ind][s][region]
            z_0[ind].loc['0','45'] = 0

        elif s == '90%':
            Y[ind].append(0)
            X[ind].append(90)    
            zlist[ind].loc['0','90']= delta_output[ind][s][region]
            z_gdp[ind].loc['0','90']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['0','90'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['0','90']=delta_energy[ind][s][region]
            z_0[ind].loc['0','90'] = 0

        elif s == 'f2.5_basic':
            Y[ind].append(2.5)
            X[ind].append(0)      
            zlist[ind].loc['2.5','0']= delta_output[ind][s][region]
            z_gdp[ind].loc['2.5','0']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['2.5','0'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['2.5','0']=delta_energy[ind][s][region]
            z_0[ind].loc['2.5','0'] = 0

        elif s == 'f2.5_10':
            Y[ind].append(2.5)
            X[ind].append(10)
            zlist[ind].loc['2.5','10']= delta_output[ind][s][region]
            z_gdp[ind].loc['2.5','10']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['2.5','10'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['2.5','10']=delta_energy[ind][s][region]
            z_0[ind].loc['2.5','10'] = 0

        elif s == 'f2.5_45':
            Y[ind].append(2.5)
            X[ind].append(45)
            zlist[ind].loc['2.5','45']= delta_output[ind][s][region]
            z_gdp[ind].loc['2.5','45']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['2.5','45'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['2.5','45']=delta_energy[ind][s][region]
            z_0[ind].loc['2.5','45'] = 0

        elif s == 'f2.5_90':
            Y[ind].append(2.5)
            X[ind].append(90)
            zlist[ind].loc['2.5','90']= delta_output[ind][s][region]
            z_gdp[ind].loc['2.5','90']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['2.5','90'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['2.5','90']=delta_energy[ind][s][region]
            z_0[ind].loc['2.5','90'] = 0

        elif s == 'f5_basic':
            Y[ind].append(5)
            X[ind].append(0)
            zlist[ind].loc['5','0']= delta_output[ind][s][region]
            z_gdp[ind].loc['5','0']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['5','0'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['5','0']=delta_energy[ind][s][region]
            z_0[ind].loc['5','0'] = 0


        elif s == 'f5_10':
            Y[ind].append(5)
            X[ind].append(10)
            zlist[ind].loc['5','10']= delta_output[ind][s][region]
            z_gdp[ind].loc['5','10']=GDP_change[ind][s][region].loc['GDP Change','Total Value']     
            z_job[ind].loc['5','10'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['5','10']=delta_energy[ind][s][region]
            z_0[ind].loc['5','10'] = 0


        elif s == 'f5_45':
            Y[ind].append(5)
            X[ind].append(45)
            zlist[ind].loc['5','45']= delta_output[ind][s][region]
            z_gdp[ind].loc['5','45']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['5','45'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['5','45']=delta_energy[ind][s][region]
            z_0[ind].loc['5','45'] = 0

        elif s == 'f5_90':
            Y[ind].append(5)
            X[ind].append(90)
            zlist[ind].loc['5','90']= delta_output[ind][s][region]
            z_gdp[ind].loc['5','90']=GDP_change[ind][s][region].loc['GDP Change','Total Value']    
            z_job[ind].loc['5','90'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['5','90']=delta_energy[ind][s][region]
            z_0[ind].loc['5','90'] = 0

        elif s == 'f10_basic':
            Y[ind].append(10)
            X[ind].append(0)
            zlist[ind].loc['10','0']= delta_output[ind][s][region]
            z_gdp[ind].loc['10','0']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['10','0'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['10','0']=delta_energy[ind][s][region]
            z_0[ind].loc['10','0'] = 0

        elif s == 'f10_10':
            Y[ind].append(10)
            X[ind].append(10)
            zlist[ind].loc['10','10']= delta_output[ind][s][region]
            z_gdp[ind].loc['10','10']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['10','10'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['10','10']=delta_energy[ind][s][region]
            z_0[ind].loc['10','10'] = 0

        elif s == 'f10_45':
            Y[ind].append(10)
            X[ind].append(45)
            zlist[ind].loc['10','45']= delta_output[ind][s][region]
            z_gdp[ind].loc['10','45']=GDP_change[ind][s][region].loc['GDP Change','Total Value']  
            z_job[ind].loc['10','45'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['10','45']=delta_energy[ind][s][region]
            z_0[ind].loc['10','45'] = 0

        elif s == 'f10_90':
            Y[ind].append(10)
            X[ind].append(90)
            zlist[ind].loc['10','90']= delta_output[ind][s][region]
            z_gdp[ind].loc['10','90']=GDP_change[ind][s][region].loc['GDP Change','Total Value']      
            z_job[ind].loc['10','90'] = job_dx_r_tot[ind][s][region]
            z_energy[ind].loc['10','90']=delta_energy[ind][s][region]
            z_0[ind].loc['10','90'] = 0

        Z_3d[ind].append(delta_output[ind][s][region])   
        Z_3d_gdp[ind].append(GDP_change[ind][s][region].loc['GDP Change','Total Value'])  
        Z_3d_job[ind].append(job_dx_r_tot[ind][s][region])
        Z_3d_energy[ind].append(delta_energy[ind][s][region])
        Z_3d_0[ind].append(0)
        
    zlist[ind] = zlist[ind].apply(pd.to_numeric)
#    zlist[ind] = np.asarray(zlist[ind])         
    
    z_gdp[ind] = z_gdp[ind].apply(pd.to_numeric)
#    z_gdp[ind] = np.asarray(z_gdp[ind])     
    
    z_job[ind] = z_job[ind].apply(pd.to_numeric)

    z_energy[ind] = z_energy[ind].apply(pd.to_numeric)
    
    z_0[ind] = z_0[ind].apply(pd.to_numeric)
    
""" Plane fitting """
A_3d = {}
Y_temp = {}
X_temp = {}
Z_3d_temp = {}
A_3d = {}
sol = {}
sol_gdp = {}
sol_job = {}
sol_energy = {}
sol_0 = {}
#errors = {}
#errors_gdp = {}
#residual = {}
#residual_gdp = {}


 
for ind in dx:
    Y_temp[ind] =np.array(Y[ind]).reshape((len(Y[ind]), 1))
    X_temp[ind]=np.array(X[ind]).reshape((len(X[ind]), 1))

    A_3d[ind] = np.hstack([X_temp[ind],Y_temp[ind],np.ones_like(X_temp[ind])])
    A_3d[ind] = np.matrix(A_3d[ind])
# sol from https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points 
#    https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    sol[ind] = (A_3d[ind].T * A_3d[ind]).I * A_3d[ind].T * np.matrix(Z_3d[ind]).T
    sol_gdp[ind] = (A_3d[ind].T * A_3d[ind]).I * A_3d[ind].T * np.matrix(Z_3d_gdp[ind]).T
    sol_job[ind] = (A_3d[ind].T * A_3d[ind]).I * A_3d[ind].T * np.matrix(Z_3d_job[ind]).T
    sol_energy[ind] = (A_3d[ind].T * A_3d[ind]).I * A_3d[ind].T * np.matrix(Z_3d_energy[ind]).T
    sol_0[ind] = (A_3d[ind].T * A_3d[ind]).I * A_3d[ind].T * np.matrix(Z_3d_0[ind]).T
#    errors[ind] =100*( np.matrix(Z_3d[ind]).T - A_3d[ind] *sol[ind])/np.matrix(Z_3d[ind]).T
#    errors[ind] = np.matrix(Z_3d[ind]).T - A_3d[ind] *sol[ind]
#    errors_gdp[ind] = np.matrix(Z_3d_gdp[ind]).T - A_3d[ind] *sol[ind]
#    residual[ind] = np.linalg.norm(errors[ind])
#    residual_gdp[ind] = np.linalg.norm(errors_gdp[ind])
    
#    print ind
#    print  'solution:'
#    print "%f x + %f y + %f = z" % (sol[ind][0], sol[ind][1], sol[ind][2])
#    print "%.0f x + %0.f y + %0.f = z" % (2, 3, 4)
#    print "errors:"
#    print errors[ind]
#    print 'residual:'
#    print residual[ind]
#    print ' '


"""from https://stackoverflow.com/questions/44473531/i-am-plotting-a-3d-plot-and-i-want-the-colours-to-be-less-distinct """
X_test = {}
Y_test = {}
Z_output = {}
Z_gdp = {}
Z_job = {}
Z_energy  = {}
Z_0 = {}
""" plotting surface for output """
""" this works"""
X_temp = np.arange(0,90,1)
Y_temp = np.arange(0,10,0.11)
for ind in dx:
    X_test[ind], Y_test[ind] = np.meshgrid(X_temp,Y_temp)
                    
    Z_output[ind] = np.multiply(sol[ind][0],X_test[ind]) +\
                    np.multiply(sol[ind][1].flatten(),Y_test[ind]) + sol[ind][2]
                    
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    # for cmap, use 'hot' or ' Spectral' or 'inferno'
    surface = ax.plot_surface(X_test[ind],
                           Y_test[ind],
                           Z_output[ind], cmap = 'Spectral')
    
    ax.scatter3D(X[ind], Y[ind], Z_3d[ind])
    fig.colorbar(surface, shrink=0.75, aspect=18, pad = 0.08)

# make empty handle, then change handle sizes to zeros
    r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
                                 visible=False)
    legend = ax.legend([r, r, r], ['Y: Percent increase in final demand for adaptively reused buildings',
                          'X: Percent increase in non-structural components reused',
                          'Z: Change in total output ($ x1000)'], 
                            loc='lower left', bbox_to_anchor=(0.7,-0.06),
                            handletextpad= 0, handlelength=0,prop={'size': 12})
    legend.set_title('Axis labels',
                     prop={'size':14, 'weight':'bold'})
    legend._legend_box.align = "left"
    # Chris said equation isnt necessary
#    legend2 = ax.legend([r],['%.2f x + %.2f y + %.2f = z' % (sol[ind][0], sol[ind][1], sol[ind][2])],
#                        loc = 'lower left', bbox_to_anchor = (-0.05,-0.06),
#                        handletextpad = 0, handlelength =0, prop ={'size':12})
#    legend2.set_title('Equation of Plane',
#                     prop={'size':16, 'weight':'bold'})
#    legend2._legend_box.align = "left"    
    
    ax.add_artist(legend)
#    ax.add_artist(legend2)
    
    plt.title('Change in total output for ' +region+ ' resulting from \nadaptive reuse of '+ind+' industry',
                  fontweight = 'bold', fontsize = 16, y =1.03)
    ax.set_ylabel('Y', fontweight = 'bold', fontsize = 16)
    ax.set_xlabel('X', fontweight = 'bold', fontsize = 16)
    ax.set_zlabel('Z', fontweight ='bold', fontsize = 16)
# adjusting distance from tick and labels
    ax.yaxis.labelpad=10
    ax.xaxis.labelpad=10
    ax.zaxis.labelpad=40
    ax.tick_params(axis='z', pad=16)
    
  # rotate views and save image.... takes very long for one set  
 # from https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib
    
 ### vvv ONLY RUN THIS SECTION OF CODE IF IMAGES DO NO EXISTS IN FOLDER vvvv###
#    if ind == 'Residential building construction':
#        for ii in xrange(0,180,1):
#            if ii >=0 and ii<90:
#                ax.view_init(elev=(90-ii), azim=270)
#                plt.savefig("movie/residential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/output_residential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#                
#    elif ind =='Non-residential building construction':
#        for ii in xrange(0,180,1):
#            if ii >=0 and ii<90:
#                ax.view_init(elev=(90-ii), azim=270)
#                plt.savefig("movie/nonresidential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/output_nonresidential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
            
 ### ^^^ONLY RUN THIS SECTION OF CODE IF IMAGES DO NO EXISTS IN FOLDER ^^^ ###

    plt.show()





""" surface plot for GDP """

for ind in dx:
    X_test[ind], Y_test[ind] = np.meshgrid(X_temp,Y_temp)
    Z_gdp[ind] = np.multiply(sol_gdp[ind][0],X_test[ind]) +\
                    np.multiply(sol_gdp[ind][1].flatten(),Y_test[ind]) + sol_gdp[ind][2]
    fig = plt.figure(figsize=(12,12))

    ax = fig.gca(projection='3d')
    # for cmap, use 'hot' or ' Spectral' or 'inferno'
    surface = ax.plot_surface(X_test[ind],
                           Y_test[ind],
                           Z_gdp[ind], cmap = 'Spectral')
    ax.scatter3D(X[ind], Y[ind], Z_3d_gdp[ind])
    fig.colorbar(surface, shrink=0.75, aspect=18, pad = 0.08)

# make empty handle, then change handle sizes to zeros
    r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
                                 visible=False)
    legend = ax.legend([r, r, r], ['Y: Percent increase in final demand for adaptively reused buildings',
                          'X: Percent increase in non-structural components reused',
                          'Z: Change in GDP ($ x1000)'], 
                            loc='lower left', bbox_to_anchor=(0.7,-0.06),
                            handletextpad= 0, handlelength=0,prop={'size': 12})
    legend.set_title('Axis labels',
                     prop={'size':14, 'weight':'bold'})
    legend._legend_box.align = "left"
    # Chris said equation isnt necessary
#    legend2 = ax.legend([r],['%.2f x + %.2f y + %.2f = z' % (sol_gdp[ind][0], sol_gdp[ind][1], sol_gdp[ind][2])],
#                        loc = 'lower left', bbox_to_anchor = (-0.05,-0.06),
#                        handletextpad = 0, handlelength =0, prop ={'size':12})
#    legend2.set_title('Equation of Plane',
#                     prop={'size':16, 'weight':'bold'})
#    legend2._legend_box.align = "left"    
    
    ax.add_artist(legend)
#    ax.add_artist(legend2)

    plt.title('Change in GDP for '+region+' resulting from \nadaptive reuse of '+ind+' industry',
                  fontweight = 'bold', fontsize = 16, y =1.03)
    ax.set_ylabel('Y', fontweight = 'bold', fontsize = 16)
    ax.set_xlabel('X', fontweight = 'bold', fontsize = 16)
    ax.set_zlabel('Z', fontweight ='bold', fontsize = 16)
# adjusting distance from tick and labels
    ax.yaxis.labelpad=10
    ax.xaxis.labelpad=10
    ax.zaxis.labelpad=40
    ax.tick_params(axis='z', pad=16)
    
#     # rotate views and save image.... takes very long for one set  
# # from https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib

 ### vvv ONLY RUN THIS SECTION OF CODE IF IMAGES DO NO EXISTS IN FOLDER vvvv###

#    if ind == 'Residential building construction':
#        for ii in xrange(0,180,1):
#            if ii >=0 and ii<90:
#                ax.view_init(elev=(90-ii), azim=270)
#                plt.savefig("movie/residential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/gdp_residential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#                
#    elif ind =='Non-residential building construction':
#        for ii in xrange(0,180,1):
#            if ii >=0 and ii<90:
#                ax.view_init(elev=(90-ii), azim=270)
#                plt.savefig("movie/nonresidential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/gdp_nonresidential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
            
 ### ^^^ONLY RUN THIS SECTION OF CODE IF IMAGES DO NO EXISTS IN FOLDER ^^^ ###

    plt.show()



""" surface plot for jobs """

for ind in dx:
    X_test[ind], Y_test[ind] = np.meshgrid(X_temp,Y_temp)
    Z_job[ind] = np.multiply(sol_job[ind][0],X_test[ind]) +\
                    np.multiply(sol_job[ind][1].flatten(),Y_test[ind]) + sol_job[ind][2]
    fig = plt.figure(figsize=(12,12))

    ax = fig.gca(projection='3d')
    # for cmap, use 'hot' or ' Spectral' or 'inferno'
    surface = ax.plot_surface(X_test[ind],
                           Y_test[ind],
                           Z_job[ind], cmap = 'Spectral')
    ax.scatter3D(X[ind], Y[ind], Z_3d_job[ind])
    fig.colorbar(surface, shrink=0.75, aspect=18, pad = 0.08)

# make empty handle, then change handle sizes to zeros
    r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
                                 visible=False)
    legend = ax.legend([r, r, r], ['Y: Percent increase in final demand for adaptively reused buildings',
                          'X: Percent increase in non-structural components reused',
                          'Z: Change in total jobs '], 
                            loc='lower left', bbox_to_anchor=(0.7,-0.06),
                            handletextpad= 0, handlelength=0,prop={'size': 12})
    legend.set_title('Axis labels',
                     prop={'size':14, 'weight':'bold'})
    legend._legend_box.align = "left"
    # Chris said equation isnt necessary
#    legend2 = ax.legend([r],['%.2f x + %.2f y + %.2f = z' % (sol_job[ind][0], sol_job[ind][1], sol_job[ind][2])],
#                        loc = 'lower left', bbox_to_anchor = (-0.05,-0.06),
#                        handletextpad = 0, handlelength =0, prop ={'size':12})
#    legend2.set_title('Equation of Plane',
#                     prop={'size':16, 'weight':'bold'})
#    legend2._legend_box.align = "left"    
    
    ax.add_artist(legend)
#    ax.add_artist(legend2)

    plt.title('Change in total jobs in '+region+' resulting from \nadaptive reuse of '+ind+' industry',
                  fontweight = 'bold', fontsize = 16, y =1.03)
    ax.set_ylabel('Y', fontweight = 'bold', fontsize = 16)
    ax.set_xlabel('X', fontweight = 'bold', fontsize = 16)
    ax.set_zlabel('Z', fontweight ='bold', fontsize = 16)
# adjusting distance from tick and labels
    ax.yaxis.labelpad=10
    ax.xaxis.labelpad=10
    ax.zaxis.labelpad=40
    ax.tick_params(axis='z', pad=16)
    
#     # rotate views and save image.... takes very long for one set  
# # from https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib

 ### vvv ONLY RUN THIS SECTION OF CODE IF IMAGES DO NO EXISTS IN FOLDER vvvv###

#    if ind == 'Residential building construction':
#        for ii in xrange(0,180,1):
#            if ii >=0 and ii<90:
#                ax.view_init(elev=(90-ii), azim=270)
#                plt.savefig("movie/residential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/gdp_residential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#                
#    elif ind =='Non-residential building construction':
#        for ii in xrange(0,180,1):
#            if ii >=0 and ii<90:
#                ax.view_init(elev=(90-ii), azim=270)
#                plt.savefig("movie/nonresidential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/gdp_nonresidential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
            
 ### ^^^ONLY RUN THIS SECTION OF CODE IF IMAGES DO NO EXISTS IN FOLDER ^^^ ###

    plt.show()
    
""" surface plot for energy """

for ind in dx:
    X_test[ind], Y_test[ind] = np.meshgrid(X_temp,Y_temp)
    Z_energy[ind] = np.multiply(sol_energy[ind][0],X_test[ind]) +\
                    np.multiply(sol_energy[ind][1].flatten(),Y_test[ind]) + sol_energy[ind][2]
    fig = plt.figure(figsize=(12,12))

    ax = fig.gca(projection='3d')
    # for cmap, use 'hot' or ' Spectral' or 'inferno'
    surface = ax.plot_surface(X_test[ind],
                           Y_test[ind],
                           Z_energy[ind], cmap = 'Spectral')
    ax.scatter3D(X[ind], Y[ind], Z_3d_energy[ind])
    fig.colorbar(surface, shrink=0.75, aspect=18, pad = 0.08)

# make empty handle, then change handle sizes to zeros
    r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
                                 visible=False)
    legend = ax.legend([r, r, r], ['Y: Percent increase in final demand for adaptively reused buildings',
                          'X: Percent increase in non-structural components reused',
                          'Z: Change in total energy consumption (TJ) '], 
                            loc='lower left', bbox_to_anchor=(0.7,-0.06),
                            handletextpad= 0, handlelength=0,prop={'size': 12})
    legend.set_title('Axis labels',
                     prop={'size':14, 'weight':'bold'})
    legend._legend_box.align = "left"
    # Chris said equation isnt necessary
#    legend2 = ax.legend([r],['%.2f x + %.2f y + %.2f = z' % (sol_job[ind][0], sol_job[ind][1], sol_job[ind][2])],
#                        loc = 'lower left', bbox_to_anchor = (-0.05,-0.06),
#                        handletextpad = 0, handlelength =0, prop ={'size':12})
#    legend2.set_title('Equation of Plane',
#                     prop={'size':16, 'weight':'bold'})
#    legend2._legend_box.align = "left"    
    
    ax.add_artist(legend)
#    ax.add_artist(legend2)

    plt.title('Change in total energy consumption in '+region+' resultign from \nadaptive reuse of '+ind+' industry',
                  fontweight = 'bold', fontsize = 16, y =1.03)
    ax.set_ylabel('Y', fontweight = 'bold', fontsize = 16)
    ax.set_xlabel('X', fontweight = 'bold', fontsize = 16)
    ax.set_zlabel('Z', fontweight ='bold', fontsize = 16)
# adjusting distance from tick and labels
    ax.yaxis.labelpad=10
    ax.xaxis.labelpad=10
    ax.zaxis.labelpad=40
    ax.tick_params(axis='z', pad=16)
    
#     # rotate views and save image.... takes very long for one set  
# # from https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib

 ### vvv ONLY RUN THIS SECTION OF CODE IF IMAGES DO NO EXISTS IN FOLDER vvvv###

#    if ind == 'Residential building construction':
#        for ii in xrange(0,180,1):
#            if ii >=0 and ii<90:
#                ax.view_init(elev=(90-ii), azim=270)
#                plt.savefig("movie/residential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/gdp_residential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#                
#    elif ind =='Non-residential building construction':
#        for ii in xrange(0,180,1):
#            if ii >=0 and ii<90:
#                ax.view_init(elev=(90-ii), azim=270)
#                plt.savefig("movie/nonresidential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/gdp_nonresidential/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
            
 ### ^^^ONLY RUN THIS SECTION OF CODE IF IMAGES DO NO EXISTS IN FOLDER ^^^ ###

    plt.show()
    

""" plotting region of Z<0 for GDP """
for ind in dx:
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca()

    contour = ax.contour(X_test[ind], Y_test[ind], Z_gdp[ind], 10, cmap="hsv", linestyles="solid")
    plt.clabel(contour,fmt = '%1.0f', colors = 'k', fontsize = '16')

    plt.title('Contour Plot: \n'+region+' adaptive reuse of '+ind+' \nindustry: change in GDP',
                  fontweight = 'bold', fontsize = 16, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 16)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 16)
    ax.tick_params(labelsize = 14)
    plt.show()

x_max = 90
x_min = 0
# equation of upper bound
y_max = {}
y_int = {}
x_int = {}
for ind in dx:
    
    y_max[ind] = (-sol_gdp[ind][2] - sol_gdp[ind][0]*x_max)/sol_gdp[ind][1]
    y_int[ind] = (-sol_gdp[ind][2] - sol_gdp[ind][0]*0)/sol_gdp[ind][1]  
    x_int[ind] = -sol_gdp[ind][2]/sol_job[ind][0]
    
        
    fig = plt.figure()
    ax = fig.gca()
    xlist = np.arange(x_min,x_max+1,1) # Create 1-D arrays for x,y dimensions

    ylist = np.arange(y_int[ind],y_max[ind]+1,0.2) 
    
# making x and y lines for plt.fill_between
    y_0 = 0+0*xlist
#    y_0 = float(y_int[ind])+0*xlist
    y_b = float(y_max[ind]) + 0*xlist
    x_0 = 0+0*ylist
    x_90 = 90+0*ylist
    F = ((-np.array(sol_gdp[ind][2]) - np.array(sol_gdp[ind][0])*xlist)/np.array(sol_gdp[ind][1])).T
    plt.plot(xlist,F, color = 'k' ,label ='%.2f x + %.2f y + %.2f = 0' % (sol_gdp[ind][0], sol_gdp[ind][1], sol_gdp[ind][2]))
    plt.plot(xlist,y_0,
            x_0, ylist,
         x_90, ylist, color = 'none')
#    plt.plot(xlist,y_0, color = 'C0')
    hatch = plt.fill_between(xlist, F.reshape(len(xlist),),y_0, facecolor="C0",alpha = 0.5, hatch='X', edgecolor="C0", linewidth=1)
    ax.set_xlim([0,x_max+1])
#    legend = plt.legend()

    if y_max[ind]< 0:
        ax.set_ylim([0,5])
    elif y_max[ind] >= 0:
        ax.set_ylim([0,y_max[ind]+y_max[ind]*0.1])

#    plt.grid()
    plt.title('Domain of negative change of GDP in '+region+': \nadaptive reuse of \n'+ind+' industry',
                  fontweight = 'bold', fontsize = 12, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)

    # Legend handles
#    hatch = Patch(fill=False, hatch='X', color = 'C0')
    white = Patch(fill=False, hatch=' ')
    plt.legend([hatch,white],['Domain of negative changes','Domain of positive changes'], prop={'size':10})

    plt.show()

""" plotting region of Z<0 for output """


for ind in dx:
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca()

    contour = ax.contour(X_test[ind], Y_test[ind], Z_job[ind], 10, cmap="hsv", linestyles="solid")
    plt.clabel(contour,fmt = '%1.0f', colors = 'k', fontsize = '16')

    plt.title('Contour Plot: \n'+region+' adaptive reuse of '+ind+' \nindustry: change in total output',
                  fontweight = 'bold', fontsize = 16, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 16)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 16)
    ax.tick_params(labelsize = 14)
    plt.show()

x_max = 90
x_min = 0
# equation of upper bound
y_max = {}
y_int = {}
x_int = {}
for ind in dx:
    
    y_max[ind] = (-sol[ind][2] - sol[ind][0]*x_max)/sol[ind][1]
    y_int[ind] = (-sol[ind][2] - sol[ind][0]*0)/sol[ind][1]  
    x_int[ind] = -sol[ind][2]/sol_job[ind][0]
    
        
    fig = plt.figure()
    ax = fig.gca()
    xlist = np.arange(x_min,x_max+1,1) # Create 1-D arrays for x,y dimensions

    ylist = np.arange(y_int[ind],y_max[ind]+1,0.2) 
    
# making x and y lines for plt.fill_between
    y_0 = 0+0*xlist
#    y_0 = float(y_int[ind])+0*xlist
    y_b = float(y_max[ind]) + 0*xlist
    x_0 = 0+0*ylist
    x_90 = 90+0*ylist
    F = ((-np.array(sol[ind][2]) - np.array(sol[ind][0])*xlist)/np.array(sol[ind][1])).T
    plt.plot(xlist,F, color = 'k' ,label ='%.2f x + %.2f y + %.2f = 0' % (sol[ind][0], sol[ind][1], sol[ind][2]))
    plt.plot(xlist,y_0,
            x_0, ylist,
         x_90, ylist, color = 'none')
#    plt.plot(xlist,y_0, color = 'C0')
    hatch = plt.fill_between(xlist, F.reshape(len(xlist),),y_0, facecolor="C0",alpha = 0.5, hatch='X', edgecolor="C0", linewidth=1)
    ax.set_xlim([0,x_max+1])
#    legend = plt.legend()

    if y_max[ind]< 0:
        ax.set_ylim([0,5])
    elif y_max[ind] >= 0:
        ax.set_ylim([0,y_max[ind]+y_max[ind]*0.1])

#    plt.grid()
    plt.title('Domain of negative change of total output for '+region+':\n adaptive reuse of \n'+ind+' industry',
                  fontweight = 'bold', fontsize = 12, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)
    
    # Legend handles
#    hatch = Patch(fill=False, hatch='X',color = 'C0')
    white = Patch(fill=False, hatch=' ')
    plt.legend([hatch,white],['Domain of negative changes','Domain of positive changes'], prop={'size':10})

    plt.show()

""" plotting region of Z<0 for jobs """
for ind in dx:
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca()

    contour = ax.contour(X_test[ind], Y_test[ind], Z_job[ind], 10, cmap="hsv", linestyles="solid")
    plt.clabel(contour,fmt = '%1.0f', colors = 'k', fontsize = '16')

    plt.title('Contour Plot: \n'+region+' adaptive reuse of '+ind+' \nindustry: change in total jobs',
                  fontweight = 'bold', fontsize = 16, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 16)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 16)
    ax.tick_params(labelsize = 14)
    plt.show()

x_max = 90
x_min = 0
# equation of upper bound
y_max = {}
y_int = {}
x_int = {}
for ind in dx:
    
    y_max[ind] = (-sol_job[ind][2] - sol_job[ind][0]*x_max)/sol_job[ind][1]
    y_int[ind] = (-sol_job[ind][2] - sol_job[ind][0]*0)/sol_job[ind][1]  
    x_int[ind] = -sol_job[ind][2]/sol_job[ind][0]
    
        
    fig = plt.figure()
    ax = fig.gca()
    xlist = np.arange(x_min,x_max+1,1) # Create 1-D arrays for x,y dimensions

    ylist = np.arange(y_int[ind],y_max[ind]+1,0.2) 
    
# making x and y lines for plt.fill_between
    y_0 = 0+0*xlist
#    y_0 = float(y_int[ind])+0*xlist
    y_b = float(y_max[ind]) + 0*xlist
    x_0 = 0+0*ylist
    x_90 = 90+0*ylist
    F = ((-np.array(sol_job[ind][2]) - np.array(sol_job[ind][0])*xlist)/np.array(sol_job[ind][1])).T
    plt.plot(xlist,F, color = 'k' ,label ='%.2f x + %.2f y + %.2f = 0' % (sol_job[ind][0], sol_job[ind][1], sol_job[ind][2]))
    plt.plot(xlist,y_0,
            x_0, ylist,
         x_90, ylist, color = 'none')
#    plt.plot(xlist,y_0, color = 'C0')
    hatch = plt.fill_between(xlist, F.reshape(len(xlist),),y_0, facecolor="C0",alpha = 0.5, hatch='X', edgecolor="C0", linewidth=1)
    ax.set_xlim([0,x_max+1])
#    legend = plt.legend()

    if y_max[ind]< 0:
        ax.set_ylim([0,5])
    elif y_max[ind] >= 0:
        ax.set_ylim([0,y_max[ind]+y_max[ind]*0.1])


#    plt.grid()
    plt.title('Domain of job losses in '+region+':\nadaptive reuse of \n'+ind+' industry',
                  fontweight = 'bold', fontsize = 12, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)
    ax.axhline(y=0, color='k', lw = 1)
  
    # Legend handles
#    hatch = Patch(fill=False, hatch='X',color = 'C0')
    white = Patch(fill=False, hatch=' ')
    plt.legend([hatch,white],['Domain of negative changes','Domain of positive changes'], prop={'size':10})
    
    plt.show()
    
    
""" plotting region of Z<0 for Energy """
for ind in dx:
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca()

    contour = ax.contour(X_test[ind], Y_test[ind], Z_energy[ind], 10, cmap="hsv", linestyles="solid")
    plt.clabel(contour,fmt = '%1.0f', colors = 'k', fontsize = '16')

    plt.title('Contour Plot: \n'+region+' adaptive reuse of '+ind+' \nindustry: change in total energy consumption',
                  fontweight = 'bold', fontsize = 16, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 16)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 16)
    ax.tick_params(labelsize = 14)
    plt.show()

x_max = 90
x_min = 0
# equation of upper bound
y_max = {}
y_int = {}
x_int = {}
for ind in dx:
    
    y_max[ind] = (-sol_energy[ind][2] - sol_energy[ind][0]*x_max)/sol_energy[ind][1]
    y_int[ind] = (-sol_energy[ind][2] - sol_energy[ind][0]*0)/sol_energy[ind][1]  
    x_int[ind] = -sol_energy[ind][2]/sol_job[ind][0]
    
        
    fig = plt.figure()
    ax = fig.gca()
    xlist = np.arange(x_min,x_max+1,1) # Create 1-D arrays for x,y dimensions

    ylist = np.arange(y_int[ind],y_max[ind]+1,0.2) 
    
# making x and y lines for plt.fill_between
    y_0 = 0+0*xlist
#    y_0 = float(y_int[ind])+0*xlist
    y_b = float(y_max[ind]) + 0*xlist
    x_0 = 0+0*ylist
    x_90 = 90+0*ylist
    F = ((-np.array(sol_energy[ind][2]) - np.array(sol_energy[ind][0])*xlist)/np.array(sol_energy[ind][1])).T
    plt.plot(xlist,F, color = 'k' ,label ='%.2f x + %.2f y + %.2f = 0' % (sol_energy[ind][0], sol_energy[ind][1], sol_energy[ind][2]))
    plt.plot(xlist,y_0,
            x_0, ylist,
         x_90, ylist, color = 'none')
#    plt.plot(xlist,y_0, color = 'C0')
    hatch = plt.fill_between(xlist, F.reshape(len(xlist),),y_0, facecolor="C0",alpha = 0.5, hatch='X', edgecolor="C0", linewidth=1)
    ax.set_xlim([0,x_max+1])
#    legend = plt.legend()

    if y_max[ind]< 0:
        ax.set_ylim([0,5])
    elif y_max[ind] >= 0:
        ax.set_ylim([0,y_max[ind]+y_max[ind]*0.1])

#    plt.grid()
    plt.title('Domain of negative change in energy use in '+region+': \nadaptive reuse of \n'+ind+' industry',
                  fontweight = 'bold', fontsize = 12, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)

    # Legend handles
#    hatch = Patch(fill=False, hatch='X',color = 'C0')
    white = Patch(fill=False, hatch=' ')
    plt.legend([hatch,white],['Domain of negative changes','Domain of positive changes'], prop={'size':10})

    plt.show()

""" IMPOSING ALL DOMAIN PLOTS """
x_domain = np.arange(0,91,1)
y_output = {}
y_gdp = {}
y_energy = {}
y_job = {}
for ind in dx:
    fig = plt.figure(figsize=(7,5))
    ax = fig.gca()

# lines to plot
    y_output[ind] =  (float(-sol[ind][2]) - float(sol[ind][0])*x_domain)/float(sol[ind][1])
    y_gdp[ind] =  (float(-sol_gdp[ind][2]) - float(sol_gdp[ind][0])*x_domain)/float(sol_gdp[ind][1])
    y_energy[ind] = (float(-sol_energy[ind][2]) - float(sol_energy[ind][0])*x_domain)/float(sol_energy[ind][1])
    y_job[ind] = (float(-sol_job[ind][2]) - float(sol_job[ind][0])*x_domain)/float(sol_job[ind][1])

    y_0 = 0+0*x_domain

    y_max[ind] = max(max(y_output[ind]),max(y_gdp[ind]),max(y_energy[ind]),max(y_job[ind]))+1
    
    output_line, = plt.plot(x_domain,y_output[ind], label = 'Output', lw=2)
    gdp_line, = plt.plot(x_domain, y_gdp[ind], label = 'GDP',lw =2)
    energy_line, = plt.plot(x_domain,y_energy[ind], label = 'Energy',lw=2)
    job_line, = plt.plot(x_domain,y_job[ind], label = 'Job',lw=2)
    hatch = plt.fill_between(x_domain,y_energy[ind],y_job[ind],facecolor="k", alpha = 0.3, hatch='///', edgecolor="grey", linewidth=1)
    white = Patch(fill=False, hatch=' ')

#    plt.fill_between(x_domain,y_energy[ind],y_0,facecolor="none",  alpha=0.5, hatch='X', edgecolor="C0", linewidth=1)
    legend = plt.legend([output_line,gdp_line,energy_line,job_line,hatch,white],
                        ['Output','GDP','Energy','Job','Desired domain','Undesired domain'],
                        loc = 'upper left',prop={'size':9})

    legend.set_title('Domain boundaries',
                     prop={'size':10, 'weight':'bold'})
    legend._legend_box.align = "left"

    ax.set_ylim([0,y_max[ind]])
    ax.set_xlim([0,91])

    plt.title('Desired domain: \n'+region+' adaptive reuse of \n'+ind+' industry',
                  fontweight = 'bold', fontsize = 12, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)
    plt.show()
"""
NEED TO PLOT MARKETSHARE , need to fix y19 and y20

"""
x_market = [20,40,60,80]

# delta_output: RoW
y19 = [-22007.768861231983,-44015.537722460096,-66023.30658,-88031.07544]
y20 = [-13122.303189193492,-26244.60637834717,-39366.90957,	-52489.21276]
slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(x_market,y19)
line19 = slope19*np.asarray(x_market)+intercept19
slope20, intercept20, r_value20, p_value20, std_err20 = stats.linregress(x_market,y20)
line20 = slope20*np.asarray(x_market)+intercept20

## delta_output: Rest of ON
#y19 = [57386.31485898938,114772.62971778959,172158.94457657484,229545.25943507883]
#y20 = [7744.454773614474,15488.909547152114,23233.364319741726,30977.819093368948]
#slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(x_market,y19)
#line19 = slope19*np.asarray(x_market)+intercept19
#slope20, intercept20, r_value20, p_value20, std_err20 = stats.linregress(x_market,y20)
#line20 = slope20*np.asarray(x_market)+intercept20

##gdp change; RoW
#y19 = [1084.6647731401026,2169.3295462839305,3253.9943194277585,4338.659092567861]
#y20 = [287.6736958436668,575.3473917096853,863.021087538451,1150.6947833821177]
#slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(x_market,y19)
#line19 = slope19*np.asarray(x_market)+intercept19
#slope20, intercept20, r_value20, p_value20, std_err20 = stats.linregress(x_market,y20)
#line20 = slope20*np.asarray(x_market)+intercept20

## gdp changes: Rest of ON 
#y19 = [89870.97631323338,179741.95262622833,269612.9289393425,359483.90525221825]
#y20 = [51087.34394919872,102174.68789827824,153262.03184700012,204349.0] 
#slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(x_market,y19)
#line19 = slope19*np.asarray(x_market)+intercept19
#slope20, intercept20, r_value20, p_value20, std_err20 = stats.linregress(x_market,y20)
#line20 = slope20*np.asarray(x_market)+intercept20

""" for RoW or Rest of ON Output change . MAKE SURE TO UPDATE PLOT TITLE """
f, axarr = plt.subplots(2,sharex=True)
axarr[0].plot( x_market, line19,marker = 's' , label= 'Residential building construction,' + ' y={:.2f}x+{:.2f}'.format(slope19,intercept19))[0]
plt.legend(bbox_to_anchor=(0.5,-0.65), loc='lower center', ncol=1,frameon=False)
axarr[0].set_title('RoW: Total output change vs percent \nmarket share of adpative reusable buildings', fontweight = 'bold',pad=25)
axarr[1].plot(x_market, line20,marker = 'o',c='C1', label='Non-residential building construction,'+' y={:.2f}x+{:.2f}'.format(slope20,intercept20))[0]
axarr[0].grid()
axarr[1].grid()
f.text(0.5, 0.01, 'Percentage of market', ha='center', fontweight = 'bold')
f.text(0.001, 0.5, 'Output, $CAD (x1000)', va='center', rotation='vertical',fontweight = 'bold')
axarr[0].legend(loc='lower center',bbox_to_anchor=(0.470,-1.85), frameon=False)
axarr[1].legend(loc='lower center',bbox_to_anchor=(0.485,-0.8), frameon=False)

""" for RoW GDP change """
#f, axarr = plt.subplots(2,sharex=True)
#axarr[0].plot( x_market, line19,marker = 's' , c ='C0', label= 'Residential building construction,' + ' y={:.2f}x+{:.2f}'.format(slope19,intercept19))[0]
##axarr[0].scatter( x_market, y19,marker = 's' , c='C0', label = 'Residential building construction')
#plt.legend(bbox_to_anchor=(0.5,-0.65), loc='lower center', ncol=1,frameon=False)
#axarr[0].set_title('RoW: Total GDP change vs percent \nmarket share of adpative reusable buildings', fontweight = 'bold',pad=25)
#axarr[1].plot( x_market, line20,marker = 's' , c='C1', label= 'Non-residential building construction,' + ' y={:.2f}x+{:.2f}'.format(slope20,intercept20))[0]
#
##axarr[1].scatter(x_market, y20,marker = 'o',c='C1' ,label = 'Non-residential building construction')
#axarr[0].grid()
#axarr[1].grid()
#f.text(0.5, 0.01, 'Percentage of market', ha='center', fontweight = 'bold')
#f.text(0.001, 0.5, '$CAD (x1000)', va='center', rotation='vertical',fontweight = 'bold')
#axarr[0].legend(loc='lower center',bbox_to_anchor=(0.470,-1.85), frameon=False)
#axarr[1].legend(loc='lower center',bbox_to_anchor=(0.485,-0.8), frameon=False)

""" for Rest of ON: GDP Change """

#f, axarr = plt.subplots(2,sharex=True)
#axarr[0].plot( x_market, line19,marker = 's' , label= 'Residential building construction,' + ' y={:.2f}x+{:.2f}'.format(slope19,intercept19))[0]
##axarr[0].scatter( x_market, y19,marker = 's' , c='C0', label = 'Residential building construction')
#plt.legend(bbox_to_anchor=(0.5,-0.65), loc='lower center', ncol=1,frameon=False)
#axarr[0].set_title('Rest of ON: Total GDP change vs percent \nmarket share of adpative reusable buildings', fontweight = 'bold',pad=25)
#axarr[1].plot(x_market, y20,marker = 'o', c='C1' ,label = 'Non-residential building construction'+ ' y={:.2f}x+{:.2f}'.format(slope19,intercept19))[0]
#axarr[0].grid()
#axarr[1].grid()
#f.text(0.5, 0.01, 'Percentage of market', ha='center', fontweight = 'bold')
#f.text(0.001, 0.5, '$CAD (x1000)', va='center', rotation='vertical',fontweight = 'bold')
#axarr[0].legend(loc='lower center',bbox_to_anchor=(0.470,-1.85), frameon=False)
#axarr[1].legend(loc='lower center',bbox_to_anchor=(0.485,-0.8), frameon=False)


""" making dataframe table for results chapter of thesis """
        
# commodities for base scenario
order_base = [  'Cement',
           'Ready-mixed concrete',
           'Concrete products',
           'Prefabricated metal buildings and components',
           'Fabricated steel plates and other fabricated structural metal',
           'Other architectural metal products',
           'Truck transportation services for general freight',
           'Truck transportation services for specialized freight',
           'Architectural, engineering and related services',
           'Household']
# commodities for all non-structural components and base scenario
order_scenario = [ 'Cement',
                   'Ready-mixed concrete',
                   'Concrete products',
                   'Prefabricated metal buildings and components',
                   'Fabricated steel plates and other fabricated structural metal',
                   'Other architectural metal products',
                   'Truck transportation services for general freight',
                   'Truck transportation services for specialized freight',
                   'Architectural, engineering and related services',
                   'Household',
                   'Textile products, n.e.c.',
                   'Hardwood lumber',
                   'Softwood lumber',
                   'Other sawmill products and treated wood products',
                   'Veneer and plywood',
                   'Wood trusses and engineered wood members',
                   'Reconstituted wood products',
                   'Wood windows and doors',
                   'Prefabricated wood and manufactured (mobile) buildings and components',
                   'Wood products, n.e.c.',
                   'Plastic and foam building and construction materials',
                   'Plastic products, n.e.c.',
                   'Rubber products, n.e.c.',
                   'Clay and ceramic products and refractories',
                   'Glass (including automotive), glass products and glass containers',
                   'Lime and gypsum products',
                   'Non-metallic mineral products, n.e.c.',
                   'Iron and steel basic shapes and ferro-alloy products',
                   'Iron and steel pipes and tubes (except castings)',
                   'Wire and other rolled and drawn steel products',
                   'Forged and stamped metal products',
                   'Metal windows and doors',
                   'Boilers, tanks and heavy gauge metal containers',
                   'Springs and wire products',
                   'Threaded metal fasteners and other turned metal products including automotive',
                   'Metal valves and pipe fittings',
                   'Fabricated metal products, n.e.c.',
                   'Heating and cooling equipment (except household refrigerators and freezers)',
                   'Other electronic components',
                   'Electric light bulbs and tubes',
                   'Lighting fixtures',
                   'Small electric appliances',
                   'Major appliances',
                   'Switchgear, switchboards, relays and industrial control apparatus',
                   'Wood kitchen cabinets and counter tops',
                   'Wholesale margins - building materials and supplies',
                   'Retail margins - furniture and home furnishings',
                   'Retail margins - building materials, garden equipment and supplies']



table = {}                                                                      # table with all commodities
table_base = {}                                                                 # table with base commodities
table_scenario = {}                                                             # table for commodies NOT in base scenario i.e. just non-structural components
for ind in dx:
    table[ind] = pd.DataFrame(index = order_scenario,
                         columns = ['Existing','Basic', '10%', '45%', '90%'])

    for s in ['Basic','10%','45%','90%']:
        for key in alpha_U:
            table[ind].loc[key,s] = U_ar[ind][s].loc[key,ind]
            table[ind].loc[key,'Existing'] = U.loc[key,ind]
            table[ind].loc['Household',s] = gva_ar_scrubbed[ind][s].loc['Household labour',ind]
            table[ind].loc['Household','Existing'] = gva_scrubbed.loc['Household labour',ind]
    table[ind] = table[ind].apply(pd.to_numeric)
    table_base[ind] =table[ind][table[ind].index.isin(order_base)]
    table_base[ind] =table_base[ind][table_base[ind].columns.intersection(['Existing','Basic'])]
    table_scenario[ind] = table[ind][~table[ind].index.isin(order_base)]

order_f = {'Residential building construction':['Residential construction'],
           'Non-residential building construction':['Industrial buildings',
                                                    'Office buildings',
                                                    'Shopping centers, plazas, malls and stores',
                                                    'Other commercial buildings',
                                                    'Schools, colleges, universities and other educational buildings',
                                                    'Health care buildings',
                                                    'Other institutional buildings']}
  
#table_f = {}
##loc = final_demand_ar['Residential building construction']['10%'].index.get_loc('Residential construction')
#for ind in dx:
#    table_f[ind] =  pd.DataFrame(index = order_f[ind],
#                         columns = ['Existing','f2.5_basic','f5_basic','f10_basic'])   # place holder name for simpler .loc
#    for s in ['f2.5_basic','f5_basic','f10_basic']:                                # recall, f10_base, f10_10, f10_45... etc has same increase in final demand
#        for key in alpha_f[ind]:
##            loc = final_demand_ar [ind][s].index.get_loc(key)
##            table_f[ind].loc[key,s] = g_ar[ind][s][loc]
#            table_f[ind].loc[key,s] = final_demand_ar[ind][s].loc[key,:].values.sum()
#            table_f[ind].loc[key,'Existing'] = final_demand.loc[key,:].values.sum()
#    table_f[ind] =table_f[ind].fillna(0)
#    table_f[ind] = table_f[ind].apply(pd.to_numeric)
##    table_f[ind].columns = ['Existing','f2.5_basic, f2.5_10, f2.5_45 and f2.5_90', 'f5_basic, f5_10, f5_45 and f5_90', 'f10_basic, f10_10, f10_45 and f10_90']
#    table_f[ind].columns = ['Existing','2.5% cost increase','5% cost increase', '10% cost increase']
table_f = {}
temp1 = copy.deepcopy(f_scrubbed)
for ind in ar:
    if ind == 'Existing':
        temp1[ind] = pd.DataFrame(data = temp1[ind], index = industry_hr)
    elif ind != 'Existing':
        for s in scenario:
            temp1[ind][s] = pd.DataFrame(data=temp1[ind][s],index = industry_hr)
for ind in dx:
    table_f[ind] =  pd.DataFrame(index = order_f,
                         columns = ['Existing','f2.5_basic','f5_basic','f10_basic'])   # place holder name for simpler .loc
    for s in ['f2.5_basic','f5_basic','f10_basic']:                                # recall, f10_base, f10_10, f10_45... etc has same increase in final demand
        table_f[ind].loc[ind,s] = temp1[ind][s].loc[ind,:].values.astype(int)[0]
        table_f[ind].loc[ind,'Existing'] = temp1['Existing'].loc[ind,:].values.astype(int)[0]
    table_f[ind] =table_f[ind].fillna(0)
    table_f[ind] = table_f[ind].apply(pd.to_numeric)
    table_f[ind].columns = ['Existing','2.5% cost increase','5% cost increase', '10% cost increase']

""" top 10 decreases and increases """
donut = ['Residential building construction',
      'Non-residential building construction']                                  # industry to apply adaptive reuse 
top10_dxr = {}
bot10_dxr = {}
top10_dxr_print = {}
bot10_dxr_print = {}
for ind in donut:
    top10_dxr[ind] = {}
    bot10_dxr[ind] = {}
    
    top10_dxr_print[ind] = {}
    bot10_dxr_print[ind] = {}
    for s in scenario:
        top10_dxr[ind][s]= \
                dx_r[ind][s]['RoW'].sort_values(
                        'Total Industry Output (%)',ascending=False).head(10) #sorting construction industries by highest technical coefficient
        bot10_dxr[ind][s] = \
                        dx_r[ind][s]['RoW'].sort_values(
                                'Total Industry Output (%)',ascending=True).head(10) #sorting construction industries by highest technical coefficient

        top10_dxr_print[ind][s] = \
            top10_dxr[ind][s][top10_dxr[ind][s].columns.intersection(
                    ['Total Industry Output (x1000)', 'Total Industry Output (%)'])]
        bot10_dxr_print[ind][s] = \
        bot10_dxr[ind][s][bot10_dxr[ind][s].columns.intersection(
                ['Total Industry Output (x1000)', 'Total Industry Output (%)'])]
    
""" printing results to excel tables, DO THIS WHEN NECESSARY, NEED TO REFORMAT EACH CODE RUN"""
## from https://stackoverflow.com/questions/20219254/how-to-write-to-an-existing-excel-file-without-overwriting-data-using-pandas
#import openpyxl
#from openpyxl import load_workbook
#from openpyxl.styles import Font
#book = load_workbook('Results-Top10-Bot10-RoW.xlsx')
#writer = pd.ExcelWriter('Results-Top10-Bot10-RoW.xlsx', engine = 'openpyxl')
#writer.book = book
#writer.sheets = dict((ws.title,ws) for ws in book.worksheets)
#
#for ind in donut:
#    for s in scenario:
#        top10_dxr_print['Residential building construction'][s].to_excel(writer,
#                   sheet_name = s,header= False,startrow = 17,startcol = 1)
#        top10_dxr_print['Non-residential building construction'][s].to_excel(writer,
#                   sheet_name = s,header= False,startrow = 17,startcol = 4)
#        
#        bot10_dxr_print['Residential building construction'][s].to_excel(writer,
#                   sheet_name = s,header= False,startrow = 4,startcol = 1)
#        bot10_dxr_print['Non-residential building construction'][s].to_excel(writer,
#                   sheet_name = s,header= False,startrow = 4,startcol = 4)
#
#        writer.save()


""" plotting difference between down-scaling output vs down-scaling household consumption """
agg_f = {}
z_hc = {}
z_hc['Existing'] = {}

    
for row in A_r_r['Existing']:
    z_hc['Existing'][row] = Z_row_col['Existing'][row]['RoW']['Household'] + Z_row_col['Existing'][row]['Rest of ON']['Household']
    z_hc['Existing'][row] = z_hc['Existing'][row].values.reshape((len(industry_hr),1))
    agg_f[row] = pd.DataFrame(index = ind_summ)
    agg_f[row]['Method 1 - Back calculating final demand'] = np.matrix(s_d) * np.matrix(z_hc['Existing'][row])
    agg_f[row]['Method 2 - Down-scaling final demand'] =  np.matrix(s_d) * np.matrix(f_rr2['Existing'][row])
for row in A_r_r['Existing']:
#    f = plt.figure(figsize = (15,5))
    agg_f[row].plot(y = ['Method 1 - Back calculating final demand', 
                                         'Method 2 - Down-scaling final demand'], 
                                    kind = 'bar',width = 0.7,figsize =(15,5)) 
    plt.xticks(rotation = 30 , ha = 'right')
    plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
    plt.ylabel('Final household consumption \n(CAD $ x1000)', fontsize = 10, fontweight ='bold')
    plt.title(row, fontsize = 14, fontweight = 'bold')
    plt.grid(linestyle = '--', linewidth = 0.6)
    plt.show()

""" plotting percentage of industry i output to total output """
""" including households"""
x_perc = pd.DataFrame(index = ind_summ, columns =['Ontario','RoW'])

x_perc['Ontario'] = 100*( np.matrix(s_d)*np.matrix(x_ar['Existing'])/x_ar['Existing'].sum() )
x_perc['RoW'] = 100*(np.matrix(s_d)*np.matrix(x_rr['Existing']['RoW'])/x_rr['Existing']['RoW'].sum())
x_perc['Ontario'].sum() # should be 100
x_perc['RoW'].sum() # should be 100

x_perc.plot(y = ['Ontario',
                 'RoW'], kind = 'bar', width = 0.7,figsize = (15,5))
plt.xticks(rotation = 30 , ha = 'right')
plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
plt.ylabel('Percent (%)', fontsize = 10, fontweight ='bold')
#plt.title('Breakdown of total output by industry by percentage \n(including households)', fontsize = 14, fontweight = 'bold')
plt.grid(linestyle = '--', linewidth = 0.6)
plt.show()

""" excluding households """
x_perc_nohh = pd.DataFrame(index = ind_summ, columns = ['Ontario','RoW'])
x_perc_nohh['Ontario']=np.matrix(s_d)*np.matrix(x_ar['Existing'])
x_perc_nohh['RoW']=np.matrix(s_d)*np.matrix(x_rr['Existing']['RoW'])
# drop household row
x_perc_nohh= x_perc_nohh.drop(['Household labour'])
x_perc_nohh['Ontario'] *= 100/x_perc_nohh['Ontario'].sum()
x_perc_nohh['RoW'] *= 100/x_perc_nohh['RoW'].sum()
x_perc['Ontario'].sum() # should be 100
x_perc['RoW'].sum() # should be 100
# plot
x_perc_nohh.plot(y = ['Ontario',
                 'RoW'], kind = 'bar', width = 0.7,figsize = (15,5))
plt.xticks(rotation = 30 , ha = 'right')
plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
plt.ylabel('Percent (%)', fontsize = 10, fontweight ='bold')
plt.title('Breakdown of total output by industry by percentage \n(excluding households)', fontsize = 14, fontweight = 'bold')
plt.grid(linestyle = '--', linewidth = 0.6)
plt.show()

""" Plotting output changes in $ """

""" including households """
#dx_r_abs = {}
#temp = {}
#for ind in dx:
#    dx_r_abs[ind] = pd.DataFrame(index = ind_summ, columns = ['Output change ($)'])
#    dx_r_abs[ind]['Output change ($)'] = agg_dx_r[ind]['Basic']['RoW']['Total Industry Output (x1000)']   
#    temp[ind] =pd.DataFrame(data = dx_r_abs[ind]['Output change ($)'].sum(),
#        index = ['Net'], columns = ['Output change ($)'])
#    dx_r_abs[ind] = dx_r_abs[ind].append(temp[ind])
#   
#    
#    ax = dx_r_abs[ind].plot(y=['Output change ($)'],kind = 'bar', width = 0.7,figsize = (15,5),legend = None)
#    highlight = 'Net'
#    pos = dx_r_abs[ind].index.get_loc(highlight)   
#    ax.patches[pos].set_facecolor('#aa3333')
#
#    plt.xticks(rotation = 30 , ha = 'right')
#    plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
#    plt.ylabel('Output changes ($)', fontsize = 10, fontweight ='bold')
#    plt.title('RoW Output changes for \'Basic\' scenario adaptive reuse of\n '+ind+' (including households)', fontsize = 14, fontweight = 'bold')
#    plt.grid(linestyle = '--', linewidth = 0.6)
#    plt.show()
#
#""" excluding households """
#dx_r_abs_nohh = {}
#temp = {}
#for ind in dx:
#    dx_r_abs_nohh[ind] = pd.DataFrame(index = ind_summ, columns = ['Output change ($)'])
#    dx_r_abs_nohh[ind]['Output change ($)'] = agg_dx_r[ind]['Basic']['RoW']['Total Industry Output (x1000)']   
#    # drop households
#    dx_r_abs_nohh[ind] =dx_r_abs_nohh[ind].drop(['Household labour'])
#    temp[ind] =pd.DataFrame(data = dx_r_abs_nohh[ind]['Output change ($)'].sum(),
#        index = ['Net'], columns = ['Output change ($)'])
#    dx_r_abs_nohh[ind] = dx_r_abs_nohh[ind].append(temp[ind])
#   
#    
#    ax = dx_r_abs_nohh[ind].plot(y=['Output change ($)'],kind = 'bar', width = 0.7,figsize = (15,5),legend = None)
#    highlight = 'Net'
#    pos = dx_r_abs_nohh[ind].index.get_loc(highlight)   
#    ax.patches[pos].set_facecolor('#aa3333')
#
#    plt.xticks(rotation = 30 , ha = 'right')
#    plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
#    plt.ylabel('Output changes ($)', fontsize = 10, fontweight ='bold')
#    plt.title('RoW Output changes for \'Basic\' scenario adaptive reuse of\n '+ind+' (including households)', fontsize = 14, fontweight = 'bold')
#    plt.grid(linestyle = '--', linewidth = 0.6)
#    plt.show()
#
""" testing domain plots """
""" AR of non-residential in Ontario scenario f10_90 was chosen because it was within desired domain and it is convenient """

##scenario to look at 
#scen = 'f10_90'
#
#
#x_r_abs = pd.DataFrame(index = ind_summ, columns = ['Base','Scenario'])
#x_r_abs['Base']= np.matrix(s_d)*np.matrix(x_rr['Existing']['RoW'])
#x_r_abs['Scenario'] = np.matrix(s_d)*np.matrix(x_rr['Non-residential building construction'][scen]['RoW'])
#temp = pd.DataFrame(data = [[x_r_abs['Scenario'].sum()-x_r_abs['Base'].sum(),0]],
#                    index = ['Net'],
#                    columns = ['Base','Scenario'])
#x_r_abs = x_r_abs.append(temp)
#
#ax = x_r_abs.plot(y=['Base','Scenario'],kind = 'bar', width = 0.7,figsize = (15,5))
#highlight = 'Net'
#pos = x_r_abs.index.get_loc(highlight)   
#ax.patches[pos].set_facecolor('#aa3333')
#
#plt.xticks(rotation = 30 , ha = 'right')
#plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
#plt.ylabel('Output ($)', fontsize = 10, fontweight ='bold')
#plt.title('RoW output for '+scen+' scenario adaptive reuse of \nNon-residential building construction (including households)', fontsize = 14, fontweight = 'bold')
#plt.grid(linestyle = '--', linewidth = 0.6)
#plt.show()
#del temp

""" plotting percent changes """
#output_energy_perc = copy.deepcopy(agg_dx_r['Non-residential building construction'][scen]['RoW'])
#output_energy_perc = output_energy_perc.drop(['Non-residential building construction'])
##agg_dx_r['Non-residential building construction'][scen]['RoW'].plot(
##        y=['Total Industry Output (%)','Total Industry Energy Use (%)'],kind = 'bar', width = 0.7,figsize = (15,5))
##
#output_energy_perc.plot(
#        y=['Total Industry Output (%)','Total Industry Energy Use (%)'],kind = 'bar', width = 0.7,figsize = (15,5))
#plt.xticks(rotation = 30 , ha = 'right')
#plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
#plt.ylabel('Percent change (%)', fontsize = 10, fontweight ='bold')
#plt.title('RoW '+scen+' scenario adaptive reuse of \nNon-residential building construction (including households)', fontsize = 14, fontweight = 'bold')
#plt.grid(linestyle = '--', linewidth = 0.6)
#plt.show()

""" energy intensities """
energy_int_summ = pd.DataFrame(data =  np.matrix(s_d)*np.matrix(energy_int),
                               index = ind_summ, columns = ['Aggregated energy intensities'])

energy_int_summ.plot(y=['Aggregated energy intensities'],kind = 'bar', width = 0.7,figsize = (15,5),legend=None)
plt.xticks(rotation = 30 , ha = 'right')
plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
plt.ylabel('Energy Intensities (TJ/$ output)', fontsize = 10, fontweight ='bold')
#plt.title('Energy intensities by sector', fontsize = 14, fontweight = 'bold')
plt.grid(linestyle = '--', linewidth = 0.6,alpha = 0.5)
#plt.savefig('energy intensity.pdf',bbox_inches='tight')
plt.show()

""" Job intensities """
# note the job intensities will be the same because of the assumption that regional output is proportional to employment
# recall employment data has its own aggregation
for row in A_r_r['Existing']:
    emp_int_r[row].plot(y=['Job intensity (jobs/$ output)'],kind = 'bar', width = 0.5,figsize = (8,5),legend=None)
    plt.xticks(rotation = 30 , ha = 'right')
    plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
    plt.ylabel('Job Intensities (jobs/$ output)', fontsize = 10, fontweight ='bold')
#    plt.title(row+' Job intensities by sector', fontsize = 14, fontweight = 'bold')
    plt.grid(linestyle = '--', linewidth = 0.6, alpha = 0.5)
#    plt.savefig(row+'job intensity.pdf',bbox_inches='tight')
    plt.show()



