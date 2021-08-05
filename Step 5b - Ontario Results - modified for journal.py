# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:00:43 2019

@author: jacky
"""

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

x_ar_energy = {}
for ind in ar:
    x_ar_energy[ind] = {}
    if ind != 'Existing':
        for s in scenario:
            x_ar_energy[ind][s] = np.diagflat(energy_int.values)*x_ar[ind][s]
    elif ind == 'Existing':
        x_ar_energy[ind]= np.diagflat(energy_int.values)*x_ar[ind]
dx_ont = {}
for ind in ar:
    if ind != 'Existing':  
        dx_ont[ind] = {}
        for s in scenario:
            dx_ont[ind][s] = pd.DataFrame(
                    index = industry_hr)
            dx_ont[ind][s]['Total Industry Output (x1000)'] =\
                                x_ar[ind][s] - x_ar['Existing']  
            dx_ont[ind][s]['Total Industry Output (%)'] =100*(x_ar[ind][s] - \
                                x_ar['Existing'] )/x_ar['Existing'] 
            dx_ont[ind][s]['Total Industry Energy Use (TJ)'] = \
            x_ar_energy[ind][s] - x_ar_energy['Existing']
            dx_ont[ind][s]['Total Industry Energy Use (%)'] =\
                                100*(x_ar_energy[ind][s] -\
                                x_ar_energy['Existing'])/\
                                     x_ar_energy['Existing']

""" CALCULATE AGGREGATE DIFFERENCES """         
# calculate aggregate 
s_d = pd.read_excel('summary_detail_concordance_matrix.xlsx', index_col=0)
s_d = s_d.fillna(0)
s_d = s_d.T                                                                     # concordance matrix (33x227)
ind_summ =pd.Series(pd.read_csv('Input_SUIC_S.csv').iloc[:,0])

x_ar_sum = {}
x_ar_energy_sum = {}
for ind in ar:
    x_ar_sum[ind] = {}
    x_ar_energy_sum[ind] = {}
    if ind == 'Existing':
        x_ar_sum[ind] = s_d.values * x_ar[ind]
        x_ar_energy_sum[ind] = s_d.values* np.matrix(x_ar_energy[ind])
    elif ind != 'Existing':
        for s in scenario:
            x_ar_sum[ind][s] = s_d.values * x_ar[ind][s]
            x_ar_energy_sum[ind][s] = s_d.values* np.matrix(x_ar_energy[ind][s])
agg_dx_ont = {}
for ind in ar:
    if ind != 'Existing':
        agg_dx_ont[ind] = {}
        for s in scenario:
            agg_dx_ont[ind][s] = pd.DataFrame(index = ind_summ)
            agg_dx_ont[ind][s]['Total Industry Output (x1000)'] = x_ar_sum[ind][s] - x_ar_sum['Existing']  
            agg_dx_ont[ind][s]['Total Industry Output (%)'] = 100*(x_ar_sum[ind][s] - x_ar_sum['Existing']  )/x_ar_sum['Existing']
            agg_dx_ont[ind][s]['Total Industry Energy Use (TJ)'] = x_ar_energy_sum[ind][s] - x_ar_energy_sum['Existing']
            agg_dx_ont[ind][s]['Total Industry Energy Use (%)'] = 100*(x_ar_energy_sum[ind][s] - x_ar_energy_sum['Existing'])/x_ar_energy_sum['Existing']

#""" calculate difference between RoW agg dx, and Ontario agg dx """
#
#agg_dx_ont_row = {}
#
#for ind in ar:
#    if ind != 'Existing':
#        agg_dx_ont_row[ind] = {}
            
#        for s in scenario:
#            agg_dx_ont_row[ind][s] = pd.DataFrame(index = ind_summ)
#            agg_dx_ont_row[ind][s]['Difference between RoW and Ont'] = agg_dx_ont[ind][s]['Total Industry Output (%)'] - agg_dx_r[ind][s]['RoW']['Total Industry Output (%)']

""" CALCULATE EMPLOYMENT CHANGES """
# employment data has their own aggregation of industries. make concordance matrix from excel file
e_a_d = pd. read_excel('emp_agg_detail_concordance_matrix.xlsx',index_col=0)
e_a_d = e_a_d.fillna(0)
e_a_d = e_a_d.T
ind_emp_agg = pd.Series(pd.read_csv('Employment_aggregated_industries.csv').iloc[:,0])

# irio output aggregated to the level of employment data
x_ar_emp = {}
x_ar_energy_emp = {}
for ind in ar:
    x_ar_emp[ind] = {}
    x_ar_energy_emp[ind] = {}
    if ind == 'Existing':
        x_ar_emp[ind] = e_a_d.values * x_ar[ind]
        x_ar_energy_emp[ind] = e_a_d.values* np.matrix(x_ar_energy[ind])
    elif ind != 'Existing':
        for s in scenario:
            x_ar_emp[ind][s] = e_a_d.values * x_ar[ind][s]
            x_ar_energy_emp[ind][s] = e_a_d.values * np.matrix(x_ar_energy[ind][s])

""" Calculate employment intensities and job changes"""
""" Updating employment intensity as per emails Oct 25th. #jobs/VA-labour has to be constant"""

""" percentage of aggregated construction that each detail construction occupies - manually done on excel"""
con_perc = {'Residential building construction':0.4131,
            'Non-residential building construction':0.1679,
            'Transportation engineering construction':0.0980,
            'Oil and gas engineering construction':0.0141,
            'Electric power engineering construction':0.0495,
            'Communication engineering construction':0.0280,
            'Other engineering construction':0.0654,
            'Repair construction':0.1490,
            'Other activities of the construction industry':0.0149}
temp = {}
emp_int = {}
for ind in ar:
    if ind == 'Existing':
        next
    elif ind != 'Existing':
        temp[ind] = copy.deepcopy(employ)
        # 0.2 is the increase in household labour from alpha_gva
        temp[ind].loc['Construction','ON'] *=(1+beta*0.2*con_perc[ind])
        temp[ind] = temp[ind].loc[:,'ON'].values.reshape((len(ind_emp_agg),1))          #
        # household intensity
        temp[ind][-1] = 0
        emp_int[ind] = np.divide(temp[ind],x_ar_emp['Existing'])
        #temp = {}

emp_dx_ont = {}                                                                     # dx_r aggregated to classification that employment is in
for ind in ar:
    if ind != 'Existing':
        emp_dx_ont[ind] = {}
        for s in scenario:
            emp_dx_ont[ind][s] = pd.DataFrame(index = ind_emp_agg)
            emp_dx_ont[ind][s]['Total Industry Output (x1000)'] = x_ar_emp[ind][s] - x_ar_emp['Existing']  
            emp_dx_ont[ind][s]['Total Industry Output (%)'] = 100*(x_ar_emp[ind][s] - x_ar_emp['Existing']  )/x_ar_emp['Existing']
            emp_dx_ont[ind][s]['Total Industry Energy Use (TJ)'] = x_ar_energy_emp[ind][s] - x_ar_energy_emp['Existing']


job_dx_ont = {}
job_dx_ont_tot = {}
for ind in ar:
    if ind != 'Existing':
        job_dx_ont[ind] = {}
        job_dx_ont_tot[ind]  = {}
        for s in scenario:
            job_dx_ont[ind][s] = pd.DataFrame(index = ind_emp_agg)
            job_dx_ont[ind][s]['Total Job Changes'] = 1000*np.multiply(emp_int[ind],
                     emp_dx_ont[ind][s]['Total Industry Output (x1000)'].values.reshape(len(ind_emp_agg),1))
            
            job_dx_ont_tot[ind][s]=  float(job_dx_ont[ind][s].values.sum(axis = 0))

""" industry output changes """
delta_output = {}
delta_energy = {}
#test = {}
delta_output_perc= {}
for ind in ar:
    if ind != 'Existing':
        delta_output[ind] = {}
        delta_energy[ind] = {}
#        test[ind] = {}
        delta_output_perc[ind]= {}
        for s in scenario:
            delta_output[ind][s] = dx_ont[ind][s]['Total Industry Output (x1000)'][:-1].sum(axis=0) 
            delta_energy[ind][s] = dx_ont[ind][s]['Total Industry Energy Use (TJ)'][:-1].sum(axis=0)
#                test[ind][s][row] = emp_dx_r[ind][s][row]['Total Industry Output (x1000)'].sum(axis=0) - delta_output[ind][s][row]
            delta_output_perc[ind][s] = delta_output[ind][s]/x_ar['Existing'][:-1].sum(axis=0)
# existing GDP
GDP_ont = {}
for ind in ar:
    GDP_ont[ind] = {}
    if ind != 'Existing':
        for s in scenario:
            GDP_ont[ind][s] = {}        
            GDP_ont[ind][s] = i_gva*(v_coeff[ind][s]*np.diagflat(x_ar[ind][s]))*i_hh
    elif ind == 'Existing':
        GDP_ont[ind] = i_gva*(v_coeff[ind]*np.diagflat(x_ar[ind]))*i_hh

GDP_change = {}

for ind in ar:
    GDP_change[ind] = {}
    
    if ind != 'Existing':
        for s in scenario:
            GDP_change[ind][s] = pd.DataFrame(index = ['GDP Change'],
                                              columns = ['Total Value','Percent'])
            GDP_change[ind][s]['Total Value'] = gdp[ind][s]-gdp['Existing']
            GDP_change[ind][s]['Percent'] = 100*(gdp[ind][s]-gdp['Existing'])/gdp['Existing']
    elif ind == 'Existing':
        GDP_change[ind] = pd.DataFrame(index = ['GDP Change'],
                                          columns = ['Total Value','Percent'])
        GDP_change[ind]['Total Value'] = gdp[ind]-gdp['Existing']
        GDP_change[ind]['Percent'] = 100*(gdp[ind]-gdp['Existing'])/gdp['Existing']
# %%
                
"""PLOTTING - aggregated data 
"""                 
x_axis = ['Basic', '10%','45%','90%']
x_fake = [0,10,45,90]
x_fd = [0,2.5,5,10]

for ind in dx_ont:
    for s in scenario:
        f = plt.figure(figsize=(15,5))
        agg_dx_ont[ind][s]['Total Industry Output (%)'].plot(kind ='bar',legend = False)
        plt.xticks(rotation = 30 , ha = 'right')
        plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
        plt.ylabel('Percent change, (%)', fontsize = 12, fontweight ='bold')
        plt.title('Ontario ' +ind +' : \nChanges in output for aggregated industries \n\''+s+'\' scenario', fontsize = 14, fontweight = 'bold')
        plt.grid(linestyle = '--', linewidth = 0.6)
        plt.show()
""" 3D PLOTS """

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
for ind in dx_ont:
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
    for s in scenario:
        if s == 'Basic':
            Y[ind].append(0)
            X[ind].append(0)
            zlist[ind].loc['0','0']=delta_output[ind][s]
            z_gdp[ind].loc['0','0']=GDP_change[ind][s].loc['GDP Change','Total Value']           
#            z_wage[ind].loc['0','0'] = dx_r[ind][s].loc['Household labour','Total Industry Output (%)']
            z_job[ind].loc['0','0'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['0','0']=delta_energy[ind][s]
        elif s == '10%':
            Y[ind].append(0)
            X[ind].append(10)
            zlist[ind].loc['0','10']= delta_output[ind][s]
            z_gdp[ind].loc['0','10']=GDP_change[ind][s].loc['GDP Change','Total Value']  
            z_job[ind].loc['0','10'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['0','10']=delta_energy[ind][s]
            
        elif s == '45%':
            Y[ind].append(0)
            X[ind].append(45)
            zlist[ind].loc['0','45']= delta_output[ind][s]
            z_gdp[ind].loc['0','45']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['0','45'] = job_dx_ont_tot[ind][s]            
            z_energy[ind].loc['0','45']=delta_energy[ind][s]

        elif s == '90%':
            Y[ind].append(0)
            X[ind].append(90)    
            zlist[ind].loc['0','90']= delta_output[ind][s]
            z_gdp[ind].loc['0','90']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['0','90'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['0','90']=delta_energy[ind][s]

        elif s == 'f2.5_basic':
            Y[ind].append(2.5)
            X[ind].append(0)      
            zlist[ind].loc['2.5','0']= delta_output[ind][s]
            z_gdp[ind].loc['2.5','0']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['2.5','0'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['2.5','0']=delta_energy[ind][s]

        elif s == 'f2.5_10':
            Y[ind].append(2.5)
            X[ind].append(10)
            zlist[ind].loc['2.5','10']= delta_output[ind][s]
            z_gdp[ind].loc['2.5','10']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['2.5','10'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['2.5','10']=delta_energy[ind][s]

        elif s == 'f2.5_45':
            Y[ind].append(2.5)
            X[ind].append(45)
            zlist[ind].loc['2.5','45']= delta_output[ind][s]
            z_gdp[ind].loc['2.5','45']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['2.5','45'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['2.5','45']=delta_energy[ind][s]

        elif s == 'f2.5_90':
            Y[ind].append(2.5)
            X[ind].append(90)
            zlist[ind].loc['2.5','90']= delta_output[ind][s]
            z_gdp[ind].loc['2.5','90']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['2.5','90'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['2.5','90']=delta_energy[ind][s]

        elif s == 'f5_basic':
            Y[ind].append(5)
            X[ind].append(0)
            zlist[ind].loc['5','0']= delta_output[ind][s]
            z_gdp[ind].loc['5','0']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['5','0'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['5','0']=delta_energy[ind][s]


        elif s == 'f5_10':
            Y[ind].append(5)
            X[ind].append(10)
            zlist[ind].loc['5','10']= delta_output[ind][s]
            z_gdp[ind].loc['5','10']=GDP_change[ind][s].loc['GDP Change','Total Value']     
            z_job[ind].loc['5','10'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['5','10']=delta_energy[ind][s]

        elif s == 'f5_45':
            Y[ind].append(5)
            X[ind].append(45)
            zlist[ind].loc['5','45']= delta_output[ind][s]
            z_gdp[ind].loc['5','45']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['5','45'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['5','45']=delta_energy[ind][s]

        elif s == 'f5_90':
            Y[ind].append(5)
            X[ind].append(90)
            zlist[ind].loc['5','90']= delta_output[ind][s]
            z_gdp[ind].loc['5','90']=GDP_change[ind][s].loc['GDP Change','Total Value']    
            z_job[ind].loc['5','90'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['5','90']=delta_energy[ind][s]

        elif s == 'f10_basic':
            Y[ind].append(10)
            X[ind].append(0)
            zlist[ind].loc['10','0']= delta_output[ind][s]
            z_gdp[ind].loc['10','0']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['10','0'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['10','0']=delta_energy[ind][s]

        elif s == 'f10_10':
            Y[ind].append(10)
            X[ind].append(10)
            zlist[ind].loc['10','10']= delta_output[ind][s]
            z_gdp[ind].loc['10','10']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['10','10'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['10','10']=delta_energy[ind][s]

        elif s == 'f10_45':
            Y[ind].append(10)
            X[ind].append(45)
            zlist[ind].loc['10','45']= delta_output[ind][s]
            z_gdp[ind].loc['10','45']=GDP_change[ind][s].loc['GDP Change','Total Value']  
            z_job[ind].loc['10','45'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['10','45']=delta_energy[ind][s]

        elif s == 'f10_90':
            Y[ind].append(10)
            X[ind].append(90)
            zlist[ind].loc['10','90']= delta_output[ind][s]
            z_gdp[ind].loc['10','90']=GDP_change[ind][s].loc['GDP Change','Total Value']      
            z_job[ind].loc['10','90'] = job_dx_ont_tot[ind][s]
            z_energy[ind].loc['10','90']=delta_energy[ind][s]

        Z_3d[ind].append(delta_output[ind][s])   
        Z_3d_gdp[ind].append(GDP_change[ind][s].loc['GDP Change','Total Value'])  
        Z_3d_job[ind].append(job_dx_ont_tot[ind][s])
        Z_3d_energy[ind].append(delta_energy[ind][s])
        
    zlist[ind] = zlist[ind].apply(pd.to_numeric)
#    zlist[ind] = np.asarray(zlist[ind])         
    
    z_gdp[ind] = z_gdp[ind].apply(pd.to_numeric)
#    z_gdp[ind] = np.asarray(z_gdp[ind])     
    
    z_job[ind] = z_job[ind].apply(pd.to_numeric)

    z_energy[ind] = z_energy[ind].apply(pd.to_numeric)
    
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

 
for ind in dx_ont:
    Y_temp[ind] =np.array(Y[ind]).reshape((len(Y[ind]), 1))
    X_temp[ind]=np.array(X[ind]).reshape((len(X[ind]), 1))

    A_3d[ind] = np.hstack([X_temp[ind],Y_temp[ind],np.ones_like(X_temp[ind])])
    A_3d[ind] = np.matrix(A_3d[ind])
# sol from https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points 
    sol[ind] = (A_3d[ind].T * A_3d[ind]).I * A_3d[ind].T * np.matrix(Z_3d[ind]).T
    sol_gdp[ind] = (A_3d[ind].T * A_3d[ind]).I * A_3d[ind].T * np.matrix(Z_3d_gdp[ind]).T
    sol_job[ind] = (A_3d[ind].T * A_3d[ind]).I * A_3d[ind].T * np.matrix(Z_3d_job[ind]).T
    sol_energy[ind] = (A_3d[ind].T * A_3d[ind]).I * A_3d[ind].T * np.matrix(Z_3d_energy[ind]).T

"""from https://stackoverflow.com/questions/44473531/i-am-plotting-a-3d-plot-and-i-want-the-colours-to-be-less-distinct """
X_test = {}
Y_test = {}
Z_output = {}
Z_gdp = {}
Z_job = {}
Z_energy  = {}
""" plotting surface for output """
""" this works"""
X_temp = np.arange(0,90,1)
Y_temp = np.arange(0,10,0.11)
for ind in dx_ont:
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
    
#    plt.title('Change in total output for Ontario resulting from \nAdaptive reuse of '+ind+' industry',
#                  fontweight = 'bold', fontsize = 16, y =1.03)
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
#                plt.savefig("movie/ontario_output_res/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/ontario_output_res/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#                
#    elif ind =='Non-residential building construction':
#        for ii in xrange(0,180,1):
#            if ii >=0 and ii<90:
#                ax.view_init(elev=(90-ii), azim=270)
#                plt.savefig("movie/ontario_output_nonres/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            elif ii>=90 and ii<=180:
#                ax.view_init(elev=0, azim=(270+(ii-90)))
#                plt.savefig("movie/ontario_output_nonres/movie%d.png" % ii)                                   # saves to 'movie' folder where this script is
#            
 ### ^^^ONLY RUN THIS SECTION OF CODE IF IMAGES DO NO EXISTS IN FOLDER ^^^ ###

#    plt.savefig(ind+' output plot.pdf',bbox_inches='tight')
    plt.show()
""" surface plot for GDP """

for ind in dx_ont:
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

#    plt.title('Change in GDP for Ontario resulting from \nadaptive reuse of '+ind+' industry',
#                  fontweight = 'bold', fontsize = 16, y =1.03)
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

#    plt.savefig(ind+' GDP plot.pdf',bbox_inches='tight')
    plt.show()



""" surface plot for jobs """

for ind in dx_ont:
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

#    plt.title('Change in total jobs in Ontario resulting from \nadaptive reuse of '+ind+' industry',
#                  fontweight = 'bold', fontsize = 16, y =1.03)
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

#    plt.savefig(ind+' job plot.pdf',bbox_inches='tight')
    plt.show()
    
""" surface plot for energy """

for ind in dx_ont:
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

#    plt.title('Change in total energy consumption in Ontario resulting from \nadaptive reuse of '+ind+' industry',
#                  fontweight = 'bold', fontsize = 16, y =1.03)
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
#    plt.savefig(ind+'energyplot.pdf',bbox_inches='tight')
    plt.show()
    

""" plotting region of Z<0 for GDP """
for ind in dx_ont:
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca()

    contour = ax.contour(X_test[ind], Y_test[ind], Z_gdp[ind], 10, cmap="Spectral", linestyles="solid")
    plt.clabel(contour,fmt = '%1.0f', colors = 'k', fontsize = '16')

    plt.title('Contour Plot: \n Ontario adaptive reuse of '+ind+' \nindustry: change in GDP',
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
for ind in dx_ont:
    
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
    plt.title('Domain of negative GDP change in Ontario: \nadaptive reuse of \n'+ind+' industry',
                  fontweight = 'bold', fontsize = 12, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)

    # Legend handles
#    hatch = Patch(fill=False, hatch='X',color = 'C0')
    white = Patch(fill=False, hatch=' ')
    plt.legend([hatch,white],['Domain of negative changes','Domain of positive changes'], prop={'size':10})

    plt.show()

""" plotting region of Z<0 for output """
for ind in dx_ont:
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca()

    contour = ax.contour(X_test[ind], Y_test[ind], Z_output[ind], 10, cmap = "Spectral", linestyles="solid")
    plt.clabel(contour,fmt = '%1.0f', colors = 'k', fontsize = '16')

    plt.title('Contour Plot: \n Ontario adaptive reuse of '+ind+' \nindustry: change in total output',
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
for ind in dx_ont:
    
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
    plt.title('Domain of negative change of total output for Ontario: \nadaptive reuse of \n'+ind+' industry',
                  fontweight = 'bold', fontsize = 12, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)

    # Legend handles
#    hatch = Patch(fill=False, hatch='X',color = 'C0')
    white = Patch(fill=False, hatch=' ')
    plt.legend([hatch,white],['Domain of negative changes','Domain of positive changes'], prop={'size':10})

    plt.show()

""" plotting region of Z<0 for jobs """
for ind in dx_ont:
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca()

    contour = ax.contour(X_test[ind], Y_test[ind], Z_job[ind], 10, cmap = "Spectral", linestyles="solid")
    plt.clabel(contour,fmt = '%1.0f', colors = 'k', fontsize = '16')

    plt.title('Contour Plot: \n Ontario adaptive reuse of '+ind+' \nindustry: change in total jobs',
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
for ind in dx_ont:
    
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
    plt.title('Domain of job losses in Ontario: \nadaptive reuse of \n'+ind+' industry',
                  fontweight = 'bold', fontsize = 12, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)

    # Legend handles
#    hatch = Patch(fill=False, hatch='X',color = 'C0')
    white = Patch(fill=False, hatch=' ')
    plt.legend([hatch,white],['Domain of negative changes','Domain of positive changes'], prop={'size':10})

    plt.show()
    
    
""" plotting region of Z<0 for Energy """
for ind in dx_ont:
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca()

    contour = ax.contour(X_test[ind], Y_test[ind], Z_energy[ind], 10, cmap = "Spectral", linestyles="solid")
    plt.clabel(contour,fmt = '%1.0f', colors = 'k', fontsize = '16')

    plt.title('Contour Plot: \n Ontario adaptive reuse of '+ind+' \nindustry: change in total energy consumption',
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
for ind in dx_ont:
    
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
    hatch= plt.fill_between(xlist, F.reshape(len(xlist),),y_0, facecolor="C0",alpha = 0.5,   hatch='X', edgecolor="C0", linewidth=1)
    ax.set_xlim([0,x_max+1])
#    legend = plt.legend()

    if y_max[ind]< 0:
        ax.set_ylim([0,5])
    elif y_max[ind] >= 0:
        ax.set_ylim([0,y_max[ind]+y_max[ind]*0.1])

#    plt.grid()
    plt.title('Domain of negative change in energy use in Ontario: \nadaptive reuse of \n'+ind+' industry',
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
y_output_ont = {}
y_gdp_ont = {}
y_energy_ont = {}
y_job_ont = {}
y_max[ind] = {}
for ind in dx_ont:
    fig = plt.figure(figsize = (7,5))
    ax = fig.gca()

# lines to plot
    y_output_ont[ind] =  (float(-sol[ind][2]) - float(sol[ind][0])*x_domain)/float(sol[ind][1])
    y_gdp_ont[ind] =  (float(-sol_gdp[ind][2]) - float(sol_gdp[ind][0])*x_domain)/float(sol_gdp[ind][1])
    y_energy_ont[ind] = (float(-sol_energy[ind][2]) - float(sol_energy[ind][0])*x_domain)/float(sol_energy[ind][1])
    y_job_ont[ind] = (float(-sol_job[ind][2]) - float(sol_job[ind][0])*x_domain)/float(sol_job[ind][1])

    y_max[ind] = max(max(y_output_ont[ind]),max(y_gdp_ont[ind]),max(y_energy_ont[ind]),max(y_job_ont[ind]))+1
    
    output_line_ont, = plt.plot(x_domain,y_output_ont[ind], label = 'Output', lw=2)
    gdp_line_ont, = plt.plot(x_domain, y_gdp_ont[ind], label = 'GDP',lw =2)
    energy_line_ont, = plt.plot(x_domain,y_energy_ont[ind], label = 'Energy',lw=2)
    job_line_ont, = plt.plot(x_domain,y_job_ont[ind], label = 'Jobs',lw=2)
    # scatter points of where output is positive, energy use is negative
    if ind == 'Non-residential building construction':
        plt.scatter(10,5,color = 'k')
        plt.scatter(45,10,color = 'k')
        plt.scatter(90,10,color = 'k')
        plt.scatter(10,10,color = 'k')
        
        plt.annotate('f5_10',(10,5),xytext = (11,5.5))
        plt.annotate('f10_45',(45,10),xytext = (46,10.5))
        plt.annotate('f10_90',(90,10),xytext = (80,10.5))
        plt.annotate('f10_10',(10,10),xytext=(11,10.5))
    elif ind == 'Residential building construction':
        plt.scatter(10,5,color = 'k')
        plt.scatter(45,10,color = 'k')
#        plt.scatter(90,10,color = 'k')
        plt.scatter(10,10,color = 'k')
        
        plt.annotate('f5_10',(10,5),xytext = (11,5.5))
        plt.annotate('f10_45',(45,10),xytext = (46,10.5))
#        plt.annotate('f10_90',(90,10),xytext = (80,10.5))
        plt.annotate('f10_10',(10,10),xytext=(11,10.5))        
        
    hatch = plt.fill_between(x_domain,y_energy_ont[ind],y_job_ont[ind],facecolor="dimgrey", alpha = 0.3, hatch='///', edgecolor="grey", linewidth=1)
    white = Patch(fill=False, hatch=' ')

    ax.set_ylim([0,y_max[ind]])
    ax.set_xlim([0,91])

    legend = plt.legend([output_line_ont,gdp_line_ont,energy_line_ont,job_line_ont,hatch,white],
                        ['Output','GDP','Energy','Jobs','Desired domain','Undesired domain'],
                        loc = 'upper left',prop={'size':9})
    legend.set_title('Domain boundaries',
                     prop={'size':10, 'weight':'bold'})
    legend._legend_box.align = "left"


#    plt.title('Desired domain: \n Ontario adaptive reuse of \n'+ind+' industry',
#                  fontweight = 'bold', fontsize = 12, y =1.03)
    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)

#    plt.savefig('figure'+ind+'.pdf')
    plt.savefig('figure'+ind+'.pdf', dpi = 400, format = 'pdf',bbox_inches='tight')
#    plt.savefig('fig'+ind+'.eps', format='eps')
#    plt.savefig('fig'+ ind + '.tiff', format ='tiff', dpi = 500)
    plt.show()
""" market share effect on Output """
x_market = [20,40,60,80]

# delta_output: Ontario
y19 = [-766684.1386087122,-1533368.2772174552,-2300052.415826303,-3066736.5544349663]
y20 = [-183215.13049762347,-366430.2609950759,-549645.3914923853,-732860.521990201]
# GDP
#y20 = [682080000.00,	682118000.00,	682156158.77,	682194231.70]
#y19 = [682133000.00	,682224000.00,	682314806.92	,682405762.56]
slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(x_market,y19)
line19 = slope19*np.asarray(x_market)+intercept19
slope20, intercept20, r_value20, p_value20, std_err20 = stats.linregress(x_market,y20)
line20 = slope20*np.asarray(x_market)+intercept20

""" Ontario output change vs market share """
f, axarr = plt.subplots(2,sharex=True)
axarr[0].plot( x_market, line19,marker = 's' , label= 'Residential building construction')[0]
plt.legend(bbox_to_anchor=(0.5,-0.65), loc='lower center', ncol=1,frameon=False)
#axarr[0].set_title('Ontario: Total output change vs percent \nmarket share of adpative reusable buildings', fontweight = 'bold',pad=25)
axarr[1].plot(x_market, line20,marker = 'o',c='C1', label='Non-residential building construction')[0]
axarr[0].grid()
axarr[1].grid()
f.text(0.5, 0.01, 'Percentage of market', ha='center', fontweight = 'bold')
f.text(-0.05, 0.5, 'Output $CAD (x1000)', va='center', rotation='vertical',fontweight = 'bold')
axarr[0].legend(loc='lower center',bbox_to_anchor=(0.455,-1.85), frameon=False)
axarr[1].legend(loc='lower center',bbox_to_anchor=(0.485,-0.8), frameon=False)
plt.savefig('marketshare.pdf',bbox_inches='tight')
""" top 10 decreases and increases """
donut = ['Residential building construction',
      'Non-residential building construction']                                  # industry to apply adaptive reuse 
top10_dxr_ont = {}
bot10_dxr_ont = {}
top10_dxr_ont_print = {}
bot10_dxr_ont_print = {}
for ind in donut:
    top10_dxr_ont[ind] = {}
    bot10_dxr_ont[ind] = {}
    
    top10_dxr_ont_print[ind] = {}
    bot10_dxr_ont_print[ind] = {}
    for s in scenario:
        top10_dxr_ont[ind][s]= \
                dx_ont[ind][s].sort_values(
                        'Total Industry Output (%)',ascending=False).head(10) #sorting construction industries by highest technical coefficient
        bot10_dxr_ont[ind][s] = \
                        dx_ont[ind][s].sort_values(
                                'Total Industry Output (%)',ascending=True).head(10) #sorting construction industries by highest technical coefficient

        top10_dxr_ont_print[ind][s] = \
            top10_dxr_ont[ind][s][top10_dxr_ont[ind][s].columns.intersection(
                    ['Total Industry Output (x1000)', 'Total Industry Output (%)'])]
        bot10_dxr_ont_print[ind][s] = \
        bot10_dxr_ont[ind][s][bot10_dxr_ont[ind][s].columns.intersection(
                ['Total Industry Output (x1000)', 'Total Industry Output (%)'])]
    
""" printing results to excel tables, DO THIS WHEN NECESSARY, NEED TO REFORMAT EACH CODE RUN"""
## from https://stackoverflow.com/questions/20219254/how-to-write-to-an-existing-excel-file-without-overwriting-data-using-pandas
#import openpyxl
#from openpyxl import load_workbook
#from openpyxl.styles import Font
#book = load_workbook('Results-Top10-Bot10-Ontario.xlsx')
#writer = pd.ExcelWriter('Results-Top10-Bot10-Ontario.xlsx', engine = 'openpyxl')
#writer.book = book
#writer.sheets = dict((ws.title,ws) for ws in book.worksheets)
#
#for ind in donut:
#    for s in scenario:
#        top10_dxr_ont_print['Residential building construction'][s].to_excel(writer,
#                   sheet_name = s,header= False,startrow = 17,startcol = 1)
#        top10_dxr_ont_print['Non-residential building construction'][s].to_excel(writer,
#                   sheet_name = s,header= False,startrow = 17,startcol = 4)
#        
#        bot10_dxr_ont_print['Residential building construction'][s].to_excel(writer,
#                   sheet_name = s,header= False,startrow = 4,startcol = 1)
#        bot10_dxr_ont_print['Non-residential building construction'][s].to_excel(writer,
#                   sheet_name = s,header= False,startrow = 4,startcol = 4)
#
#        writer.save()


""" Plotting output changes in $ """

""" including households """
x_ont_abs = {}
temp = {}
for ind in dx_ont:
    x_ont_abs[ind] = pd.DataFrame(index = ind_summ, columns = ['Output change ($)'])
    x_ont_abs[ind]['Output change ($)'] = x_ar_sum[ind]['f2.5_10'] - x_ar_sum['Existing']  
    temp[ind] =pd.DataFrame(data = x_ont_abs[ind]['Output change ($)'].sum(),
        index = ['Net'], columns = ['Output change ($)'])
    x_ont_abs[ind] = x_ont_abs[ind].append(temp[ind])
   
    
    ax = x_ont_abs[ind].plot(y=['Output change ($)'],kind = 'bar', width = 0.7,figsize = (15,5),legend = None)
    highlight = 'Net'
    pos = x_ont_abs[ind].index.get_loc(highlight)   
    ax.patches[pos].set_facecolor('#aa3333')

    plt.xticks(rotation = 30 , ha = 'right')
    plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
    plt.ylabel('Output changes ($)', fontsize = 10, fontweight ='bold')
    plt.title('Ontario output changes for \'Basic\' scenario adaptive reuse of \n'+ind+' (including households)', fontsize = 14, fontweight = 'bold')
    plt.grid(linestyle = '--', linewidth = 0.6)
    plt.show()
del temp
#""" excluding households """
#x_ont_abs_nohh = {}
#temp = {}
#for ind in dx_ont:
#    x_ont_abs_nohh[ind] = pd.DataFrame(index = ind_summ, columns = ['Output change ($)'])
#    x_ont_abs_nohh[ind]['Output change ($)'] = x_ar_sum[ind]['Basic'] - x_ar_sum['Existing']  
#   # drop households
#    x_ont_abs_nohh[ind] = x_ont_abs_nohh[ind].drop(['Household labour'])
#    temp[ind] =pd.DataFrame(data = x_ont_abs_nohh[ind]['Output change ($)'].sum(),
#        index = ['Net'], columns = ['Output change ($)'])
#    x_ont_abs_nohh[ind] = x_ont_abs_nohh[ind].append(temp[ind])
#   
#    
#    ax = x_ont_abs_nohh[ind].plot(y=['Output change ($)'],kind = 'bar', width = 0.7,figsize = (15,5),legend = None)
#    highlight = 'Net'
#    pos = x_ont_abs_nohh[ind].index.get_loc(highlight)   
#    ax.patches[pos].set_facecolor('#aa3333')
#
#    plt.xticks(rotation = 30 , ha = 'right')
#    plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
#    plt.ylabel('Output changes ($)', fontsize = 10, fontweight ='bold')
#    plt.title('Ontario output changes for \'Basic\' scenario adaptive reuse of \n'+ind+' (excluding households)', fontsize = 14, fontweight = 'bold')
#    plt.grid(linestyle = '--', linewidth = 0.6)
#    plt.show()
#del temp
#
""" testing domain plots """
""" AR of non-residential in Ontario scenario f10_90 was chosen because it was within desired domain and it is convenient """

#scenario to look at 
#scen = 'f10_90'
#
#
#x_abs = pd.DataFrame(index = ind_summ, columns = ['Base','Scenario'])
#x_abs['Base']= np.matrix(s_d)*np.matrix(x_ar['Existing'])
#x_abs['Scenario'] = np.matrix(s_d)*np.matrix(x_ar['Residential building construction'][scen])
#temp = pd.DataFrame(data = [[x_abs['Scenario'].sum()-x_abs['Base'].sum(),0]],
#                    index = ['Net'],
#                    columns = ['Base','Scenario'])
#x_abs = x_abs.append(temp)
#
#ax = x_abs.plot(y=['Base','Scenario'],kind = 'bar', width = 0.7,figsize = (15,5))
#highlight = 'Net'
#pos = x_abs.index.get_loc(highlight)   
#ax.patches[pos].set_facecolor('#aa3333')
#
#plt.xticks(rotation = 30 , ha = 'right')
#plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
#plt.ylabel('Output ($)', fontsize = 10, fontweight ='bold')
#plt.title('Ontario output for '+scen+' scenario adaptive reuse of \nNon-residential building construction (including households)', fontsize = 14, fontweight = 'bold')
#plt.grid(linestyle = '--', linewidth = 0.6)
#plt.show()
#del temp

#""" energy"""
#x_abs_energy = pd.DataFrame(index = ind_summ, columns = ['Base','Scenario'])
#x_abs_energy['Base']= np.matrix(s_d)*np.matrix(x_ar_energy['Existing'])
#x_abs_energy['Scenario'] = np.matrix(s_d)*np.matrix(x_ar_energy['Residential building construction'][scen])
#temp = pd.DataFrame(data = [[x_abs_energy['Scenario'].sum()-x_abs_energy['Base'].sum(),0]],
#                    index = ['Net'],
#                    columns = ['Base','Scenario'])
#x_abs_energy = x_abs_energy.append(temp)
#
#x_abs_energy.plot(y=['Base','Scenario'],kind = 'bar', width = 0.7,figsize = (15,5))
#highlight = 'Net'
#pos = x_abs_energy.index.get_loc(highlight)   
#ax.patches[pos].set_facecolor('#aa3333')
#
#plt.xticks(rotation = 30 , ha = 'right')
#plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
#plt.ylabel('Output ($)', fontsize = 10, fontweight ='bold')
#plt.title('Ontario energy use for '+scen+' scenario adaptive reuse of \nNon-residential building construction (including households)', fontsize = 14, fontweight = 'bold')
#plt.grid(linestyle = '--', linewidth = 0.6)
#plt.show()
#del temp

""" Job intensities plot """
emp_int_ont = pd.DataFrame(emp_int[ind], index = ind_emp_agg,columns =['Job intensity (jobs/$ output)'] )
emp_int_ont.plot(y=['Job intensity (jobs/$ output)'],kind = 'bar', width = 0.5,figsize = (8,5),legend=None)
plt.xticks(rotation = 30 , ha = 'right')
plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
plt.ylabel('Job Intensities (jobs/$ output)', fontsize = 10, fontweight ='bold')
plt.title(' Job intensities by sector', fontsize = 14, fontweight = 'bold')
plt.grid(linestyle = '--', linewidth = 0.6, alpha = 0.5)
#plt.savefig('job intensity.pdf',bbox_inches='tight')

plt.show()

""" Journal Paper Plot """
agg_dx_combine = {}
for row in dx_ont:
    agg_dx_combine[row] = pd.DataFrame(index = ind_summ, 
                                      columns = ['Basic','f2.5_basic','10%'])
    agg_dx_combine[row]['Basic'] = agg_dx_ont[row]['Basic']['Total Industry Output (%)']
    agg_dx_combine[row]['f2.5_basic'] = agg_dx_ont[row]['f2.5_basic']['Total Industry Output (%)']
    agg_dx_combine[row]['10%'] = agg_dx_ont[row]['10%']['Total Industry Output (%)']
    agg_dx_combine[row].plot(y=['Basic','10%','f2.5_basic'], kind = 'bar',width = 0.7,figsize =(15,5))    
            
    plt.xticks(rotation = 30 , ha = 'right')
    plt.xlabel('Aggregated industries', fontsize = 10, fontweight ='bold')
    plt.ylabel('Final household consumption \n(CAD $ x1000)', fontsize = 10, fontweight ='bold')
    plt.title(row, fontsize = 14, fontweight = 'bold')
    plt.grid(linestyle = '--', linewidth = 0.6)
    plt.show()


#### From https://stackoverflow.com/questions/13027147/histogram-with-breaking-axis-and-interlaced-colorbar
from matplotlib import gridspec
for row in dx_ont:
    agg_dx_combine[row] = pd.DataFrame(index = ind_summ, 
                                      columns = ['Basic','f2.5_basic','10%'])
    agg_dx_combine[row]['Basic'] = agg_dx_ont[row]['Basic']['Total Industry Output (%)']
    agg_dx_combine[row]['f2.5_basic'] = agg_dx_ont[row]['f2.5_basic']['Total Industry Output (%)']
    agg_dx_combine[row]['10%'] = agg_dx_ont[row]['10%']['Total Industry Output (%)']
    
    if row == 'Residential building construction':
        f, axis = plt.subplots(3,1,sharex = True,figsize = (17,5))
        gs  = gridspec.GridSpec(3, 1, height_ratios=[1,9 ,3])
        axis[0] = plt.subplot(gs[0])
        axis[1] = plt.subplot(gs[1])
        axis[2] = plt.subplot(gs[2])
        
        agg_dx_combine[row].plot( kind = 'bar',width = 0.8, ax = axis[0])    
        agg_dx_combine[row].plot( kind = 'bar',width = 0.8, ax = axis[1]) 
        agg_dx_combine[row].plot( kind = 'bar',width = 0.8, ax = axis[2])

        # setting y-axis breaks
        axis[0].set_ylim(0.45,0.51)        
        axis[1].set_ylim(-0.25,0.15)
        axis[2].set_ylim(-3.99,-0.5)
        
        #turning of legends
        axis[0].legend().set_visible(False)
        axis[2].legend().set_visible(False)
        axis[1].legend(loc = 'lower right', fontsize=12)
        
        # removing borders
        axis[0].spines['bottom'].set_visible(False)
        axis[1].spines['top'].set_visible(False)
        axis[1].spines['bottom'].set_visible(False)
        axis[2].spines['top'].set_visible(False)
        
        axis[0].xaxis.tick_top()
        axis[0].tick_params(labeltop = 'off')
        
        axis[1].tick_params(bottom = False)
        axis[1].tick_params(labelbottom = 'off')
        
        # turning off x axis label
        axis[0].xaxis.label.set_visible(False)
        axis[1].xaxis.label.set_visible(False)
        
######## from https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#a-simple-categorical-heatmap
        axis[2].set_xticklabels(ind_summ)
        plt.setp(axis[2].get_xticklabels(), rotation=30, ha="right",
         rotation_mode="anchor")
        
        axis[2].set_xlabel('Aggregated industries', fontsize = 12, fontweight ='bold')
        axis[1].set_ylabel('Percent change (%)',fontsize = 12, fontweight = 'bold')
########
  
#        f.suptitle('Ontario: Adaptive reuse of '+row+' industry',fontsize = 14, fontweight = 'bold')
      
        #grid lines
        axis[0].grid(linestyle = '--', linewidth = 0.6, alpha = 0.5)
        axis[1].grid(linestyle = '--', linewidth = 0.6, alpha = 0.5)
        axis[2].grid(linestyle = '--', linewidth = 0.6, alpha = 0.5)
        
        # adding break lines
        d = .005
        #adding break lines at bottom
        kwargs = dict(transform=axis[0].transAxes, color='k', clip_on=False)
        axis[0].plot((-d,+d),(-d,+d), **kwargs)
        axis[0].plot((1-d,1+d),(-d,+d), **kwargs)
        # adding break lines at top and bottom
        kwargs.update(transform=axis[1].transAxes)
        axis[1].plot((-d,+d),(1-d,1+d), **kwargs)
        axis[1].plot((1-d,1+d),(1-d,1+d), **kwargs)
        
        axis[1].plot((-d,+d),(-d,+d), **kwargs)
        axis[1].plot((1-d,1+d),(-d,+d), **kwargs)        
        # break lines at top
        kwargs.update(transform=axis[2].transAxes)
        axis[2].plot((-d,+d),(1-d,1+d), **kwargs)
        axis[2].plot((1-d,1+d),(1-d,1+d), **kwargs)

#        plt.savefig('3bar'+row+'.pdf',bbox_inches='tight')
        
    if row == 'Non-residential building construction':
        f, axis = plt.subplots(2,1,sharex = True,figsize = (17,5))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[1,8])
        axis[0] = plt.subplot(gs[0])
        axis[1] = plt.subplot(gs[1])
       
        agg_dx_combine[row].plot( kind = 'bar',width = 0.8, ax = axis[0])    
        agg_dx_combine[row].plot( kind = 'bar',width = 0.8, ax = axis[1]) 

        # setting y-axis breaks
        axis[0].set_ylim(0.45,0.51)        
        axis[1].set_ylim(-0.15,0.075)
        
        #turning of legends
        axis[0].legend().set_visible(False)
        axis[1].legend(loc = 'lower right', fontsize = 12)

        # removing borders
        axis[0].spines['bottom'].set_visible(False)
        axis[1].spines['top'].set_visible(False)

        axis[0].xaxis.tick_top()
        axis[0].tick_params(labeltop = 'off')
        # turning off x axis label
        axis[0].xaxis.label.set_visible(False)
        
        # labels
        axis[1].set_xticklabels(ind_summ)
        plt.setp(axis[1].get_xticklabels(), rotation=30, ha="right",
                                 rotation_mode="anchor")
        
        axis[1].set_xlabel('Aggregated industries', fontsize = 12, fontweight ='bold')
        axis[1].set_ylabel('Percent change (%)',fontsize = 12, fontweight = 'bold')
        
#        f.suptitle('Ontario: Adaptive reuse of '+row+' industry',fontsize = 14, fontweight = 'bold')

        #grid lines
        axis[0].grid(linestyle = '--', linewidth = 0.6, alpha = 0.5)
        axis[1].grid(linestyle = '--', linewidth = 0.6,alpha = 0.5)
 
        # adding break lines
        d = .005
        #adding break lines at bottom
        kwargs = dict(transform=axis[0].transAxes, color='k', clip_on=False)
        axis[0].plot((-d,+d),(-d,+d), **kwargs)
        axis[0].plot((1-d,1+d),(-d,+d), **kwargs)
        # adding break lines at top and bottom
        kwargs.update(transform=axis[1].transAxes)
        axis[1].plot((-d,+d),(1-d,1+d), **kwargs)
        axis[1].plot((1-d,1+d),(1-d,1+d), **kwargs)
#        plt.savefig('3bar'+row+'.pdf',bbox_inches='tight')
        
#""" Combining RoW and Ontario Desired domain """
#
#for ind in dx_ont:
#    fig = plt.figure(figsize=(7,5))
#    ax = fig.gca()
#
#    output_line, = plt.plot(x_domain,y_output[ind], label = 'Output', lw=2)
#    gdp_line, = plt.plot(x_domain, y_gdp[ind], label = 'GDP',lw =2)
#    energy_line, = plt.plot(x_domain,y_energy[ind], label = 'Energy',lw=2)
#    job_line, = plt.plot(x_domain,y_job[ind], label = 'Job',lw=2)
#    hatch = plt.fill_between(x_domain,y_energy[ind],y_job[ind],facecolor="k", alpha = 0.4, hatch='///', edgecolor="k", linewidth=1)
#    white = Patch(fill=False, hatch=' ')
#
#    y_output_ont[ind] =  (float(-sol[ind][2]) - float(sol[ind][0])*x_domain)/float(sol[ind][1])
#    y_gdp_ont[ind] =  (float(-sol_gdp[ind][2]) - float(sol_gdp[ind][0])*x_domain)/float(sol_gdp[ind][1])
#    y_energy_ont[ind] = (float(-sol_energy[ind][2]) - float(sol_energy[ind][0])*x_domain)/float(sol_energy[ind][1])
#    y_job_ont[ind] = (float(-sol_job[ind][2]) - float(sol_job[ind][0])*x_domain)/float(sol_job[ind][1])
#
#    y_max[ind] = max(max(y_output_ont[ind]),max(y_gdp_ont[ind]),max(y_energy_ont[ind]),max(y_job_ont[ind]),
#                     max(y_output[ind]),max(y_gdp[ind]),max(y_energy[ind]),max(y_job[ind]))+1
#    
#    output_line_ont, = plt.plot(x_domain,y_output_ont[ind], label = 'Output', lw=2)
#    gdp_line_ont, = plt.plot(x_domain, y_gdp_ont[ind], label = 'GDP',lw =2)
#    energy_line_ont, = plt.plot(x_domain,y_energy_ont[ind], label = 'Energy',lw=2)
#    job_line_ont, = plt.plot(x_domain,y_job_ont[ind], label = 'Job',lw=2)
#
#    hatch2 = plt.fill_between(x_domain,y_energy_ont[ind],y_job_ont[ind],facecolor="indigo", alpha = 0.5, hatch='\\', edgecolor="indigo", linewidth=1)
#    white2 = Patch(fill=False, hatch=' ')
#    
##    plt.fill_between(x_domain,y_energy[ind],y_0,facecolor="none",  alpha=0.5, hatch='X', edgecolor="C0", linewidth=1)
#    legend = plt.legend([hatch, hatch2, white],
#                        ['RoW: Desired domain', 'Ontario: Desired domain', 'Undesired domain'],
#                        loc = 'upper left',prop={'size':9})
#
#    legend.set_title('Domain boundaries',
#                     prop={'size':10, 'weight':'bold'})
#    legend._legend_box.align = "left"
#
#    ax.set_ylim([0,y_max[ind]])
#    ax.set_xlim([0,91])
#
#    plt.title('Desired domain: adaptive reuse of \n'+ind+' industry',
#                  fontweight = 'bold', fontsize = 12, y =1.03)
#    ax.set_ylabel('Percent increase in final demand for \nadaptively reused buildings', fontweight = 'bold', fontsize = 11)
#    ax.set_xlabel('Percent increase in non-structural components reused', fontweight = 'bold', fontsize = 11)
#    plt.show()