# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:51:40 2018

@author: jacky
"""
"""
2019-03-13 revision: added top 10 donut chart for Provincial level. Use B matrix
"""

""" 2019-05-17 revision: Hypothetical extraction for all industry . change donut chart to bar charts """

""" 2019-05-26 HEM for Ontario """
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt



# Create function to split two-region industry output into output for each region 
def IRIO_split(new_dict, irio_var) :
    new_dict['Rest of ON'] = irio_var.iloc[0:2*len(industry_hr)/len(A_r_r['Existing']),:]
    new_dict['RoW'] = irio_var.iloc[2*len(industry_hr)/len(A_r_r['Existing']):2*len(industry_hr),:]

""" Forward and backward linkages """
# direct backward linkages
i_closed = np.ones((len(industry_hr),1))
""" backward linkages """ 
""" t_bl_avg and d_bl_avg values > 1.0 indicate strong backward linkage"""
""" HYPOTHETICAL EXTRACTION LINKAGES ARE USED IN THESIS"""
d_bl = {}
d_bl_avg = {}
t_bl = {}
t_bl_avg = {}
backward_linkages = {}



for row in A_r_r['Existing']:
    for col in A_r_r['Existing']:
        if row != col:
            next
        elif row == col :
            # direct backward linkage
            d_bl[row] = np.matrix(i_closed.T)*np.matrix(A_row_col['Existing'][row][col].values)
            # total backward linkage
            t_bl[row] = np.matrix(i_closed.T)*np.linalg.inv(
                    np.identity(len(industry_hr))-A_row_col['Existing'][row][col].values
                                                    )
            # normalized vectors
            d_bl_avg[row] = len(industry_hr)*d_bl[row]/(d_bl[row]*i_closed)           
            t_bl_avg[row] = len(industry_hr)*t_bl[row]/(t_bl[row]*i_closed)
    # convert to DataFrames
    d_bl[row] = pd.DataFrame(data = d_bl[row],
        index = ['Direct Backward Linkage'],
        columns = industry_hc).T
    t_bl[row] = pd.DataFrame(data = t_bl[row],
        index = ['Total Backward Linkage'],
        columns = industry_hc).T
    d_bl_avg[row] = pd.DataFrame(data = d_bl_avg[row],
        index = ['Normalized Direct Backward Linkage'],
        columns = industry_hc).T
    t_bl_avg[row] = pd.DataFrame(data = t_bl_avg[row],
        index = ['Normalized Total Backward Linkage'],
        columns = industry_hc).T           

    backward_linkages[row] = pd.concat([d_bl[row],
                                      t_bl[row],
                                      d_bl_avg[row],
                                      t_bl_avg[row]],axis=1)

#d_bl_avg['RoW'].mean()                                                          # d_bl_avg and t_bl_avg should equal 1.0

""" forward linkage """
# calculate Ghosh's direct output coefficients - necessary for forward linkages
B_ghosh = {}
G_ghosh = {}
for row in A_r_r['Existing']:
    for col in A_r_r['Existing']:
        if row != col:
            next
        elif row == col:
            B_ghosh[row] = np.linalg.inv(np.diagflat(x_rr_i['Existing'][row].values))*\
                                        np.matrix(Z_row_col['Existing'][row][col].values)
            G_ghosh[row] = np.linalg.inv(
                    np.identity(len(industry_hr))-B_ghosh[row]
                                        )
            B_ghosh[row] = pd.DataFrame(data = B_ghosh[row],
                   index = industry_hr,
                   columns = industry_hc)
            G_ghosh[row] = pd.DataFrame(data = G_ghosh[row],
                   index = industry_hr,
                   columns = industry_hc)            
# for ontario forward linkage
B_ghosh_ont = np.linalg.inv(np.diagflat(x_ar['Existing'])) * np.matrix(z_scrubbed['Existing'])
G_ghosh_ont = np.linalg.inv(np.identity(len(industry_hr))-B_ghosh_ont)
B_ghosh_ont = pd.DataFrame(data = B_ghosh_ont,
       index = industry_hr,
       columns = industry_hc)
G_ghosh_ont= pd.DataFrame(data = G_ghosh_ont,
       index = industry_hr,
       columns = industry_hc) 

d_fl = {}
t_fl = {}
d_fl_avg = {}
t_fl_avg = {}
forward_linkages ={}
for row in A_r_r['Existing']:
    for col in A_r_r['Existing']:
        if row != col:
            next
        elif row == col:
            # direct forward linkage
            d_fl[row] = B_ghosh[row].values*np.matrix(i_closed)
            # direct backward linkage
            t_fl[row] = G_ghosh[row].values*np.matrix(i_closed)
            # normalized vectors
            d_fl_avg[row] = len(industry_hr)*d_fl[row]/(i_closed.T*d_fl[row])
            t_fl_avg[row] = len(industry_hr)*t_fl[row]/(i_closed.T*t_fl[row])

    # convert to DataFrames            
    d_fl[row] = pd.DataFrame(data = d_fl[row],
        index = industry_hr,
        columns = ['Direct Forward Linkage'])
    t_fl[row] = pd.DataFrame(data = t_fl[row],
        index = industry_hr,
        columns = ['Total Forward Linkage'])
    d_fl_avg[row] = pd.DataFrame(data = d_fl_avg[row],
        index = industry_hr,
        columns = ['Normalized Direct Forward Linkage'])
    t_fl_avg[row] = pd.DataFrame(data = t_fl_avg[row],
        index = industry_hr,
        columns = ['Normalized Total Forward Linkage']) 
    
    forward_linkages[row] = pd.concat([d_fl[row],
                                      t_fl[row],
                                      d_fl_avg[row],
                                      t_fl_avg[row]],axis=1)

#t_fl_avg['RoW'].mean()                                                          # d_fl_avg and t_fl_avg should equal 1.0


## testing if L_sr, L_rs = -A_sr and -A_rs inverse
#test1 = np.linalg.inv(I_irio-A_irio)
#test2 = test1[0:227,0:227]
#test3 = test1[227:454,0:227]
#test4 = test1[0:227,227:454]
#test5 = test1[227:454,227:454]
#
#aaaaa = test2 - np.linalg.inv(np.identity(len(industry_hr)) -A_row_col['Rest of ON']['Rest of ON'])
#aa = test4 - np.linalg.inv(A_row_col['Rest of ON']['RoW'])


""" Hypothetical extraction """

hypo_ex = ['Residential building construction',
           'Non-residential building construction']                             # could extend to all industries
hypo_ex = industry_hr.to_list()

# used for hypothetical extraction backward linkage
hypo_ex_hh = ['Residential building construction',
           'Non-residential building construction',
           'Household']                             # could extend to all industries

A_hypo_ex = {}
f_hypo_ex = {}
for ind in hypo_ex:
    A_hypo_ex[ind]=copy.deepcopy(A_row_col['Existing'])
    f_hypo_ex[ind]=copy.deepcopy(f_rr['Existing'])
    for row in A_r_r['Existing']:
        for col in A_r_r['Existing']:
            # delete row and columns of industry 
            A_hypo_ex[ind][row][col].loc[:,ind] *= 0
            A_hypo_ex[ind][row][col].loc[ind,:] *= 0
        f_hypo_ex[ind][row].loc[ind,:] *= 0
  
# reconcatenate for Matrix calculations
A_irio_temp = {}
A_irio_he = {}
f_rr_he = {}
x_irio_he = {}
x_irio_he_energy = {}
Tj = {}                                                                         # total decrease in economic activity in $
Tj_bar = {}                                                                     # total decrease in economic activity in % dollar
Tj_energy = {}                                                                  # total decrease , energy TJ
Tj_bar_energy = {}                                                              # total decrease, % energy

for ind in hypo_ex:
    A_irio_temp[ind] = {}
    for row in A_r_r['Existing']:
        A_irio_temp[ind][row] = {}
        #concatenate each row of giant A and I
        A_irio_temp[ind][row] = np.concatenate([A_hypo_ex[ind][row][col] for col in sorted(A_r_r['Existing'])],1) 
        # creating giant matrices/vectors
    
    A_irio_he[ind] = np.concatenate([A_irio_temp[ind][row] for row in sorted(A_r_r['Existing'])],0)
    
    f_rr_he[ind] = np.concatenate([f_hypo_ex[ind][row] for row in sorted(A_r_r['Existing'])],0)
    
    x_irio_he[ind] = np.matrix(np.linalg.inv(I_irio['Existing'] - A_irio_he[ind]))*np.matrix(f_rr_he[ind])
    
    x_irio_he_energy[ind] = np.diagflat(energy_int_irio)*x_irio_he[ind]
    # Total economic activity for Ontario in $
    Tj[ind] = x_r_r_irio['Existing'].sum(axis=0) - x_irio_he[ind].sum(axis=0)
    # total economic activity for ontario in %
    Tj_bar[ind] = \
    100*(x_r_r_irio['Existing'].sum(axis=0) - x_irio_he[ind].sum(axis=0) )\
                                                        /x_r_r_irio['Existing'].sum(axis=0)
    # total energy use in Ontario in TJ
    Tj_energy[ind] = \
    x_r_r_irio_energy['Existing'].values.sum(axis=0)- x_irio_he_energy[ind].sum(axis=0)
    # total energy use in Ontario in %
    Tj_bar_energy[ind] = \
    100*(x_r_r_irio_energy['Existing'].values.sum(axis=0)- x_irio_he_energy[ind].sum(axis=0))\
                                                  /x_r_r_irio_energy['Existing'].values.sum(axis=0)

    x_irio_he[ind] = pd.DataFrame(data=x_irio_he[ind],
             index = pd.concat([industry_hr, industry_hr], axis= 0),
             columns = ['Total Industry Output (x1000)'])
    
    x_irio_he_energy[ind] = pd.DataFrame(data=x_irio_he_energy[ind],
             index = pd.concat([industry_hr, industry_hr], axis= 0),
             columns = ['Total Industry Energy Use (TJ)'])

    Tj[ind] = pd.DataFrame(data = Tj[ind],
       index = ['Decrease in total economic activity'],
       columns = ['$'])
   
    Tj_bar[ind] = pd.DataFrame(data = Tj_bar[ind],
       index = ['Decrease in total economic activity'],
       columns = ['%'])
 
    Tj_energy[ind] = pd.DataFrame(data = Tj_energy[ind],
             index = ['Decrease in total energy use'],
             columns = ['TJ'])
    
    Tj_bar_energy[ind] = pd.DataFrame(data = Tj_bar_energy[ind],
             index = ['Decrease in total energy use'],
             columns = ['%'])

# split into regions   
x_r_he = {}
x_r_he_energy = {}
Tj_r = {}
Tj_bar_r = {}
Tj_r_energy = {}
Tj_bar_r_energy = {}
x_rr_i_energy = {}

IRIO_split(x_rr_i_energy,x_r_r_irio_energy['Existing'])

for ind in hypo_ex:
    x_r_he[ind] = {}
    x_r_he_energy[ind]={}
    IRIO_split(x_r_he[ind],x_irio_he[ind])
    IRIO_split(x_r_he_energy[ind],x_irio_he_energy[ind])
    
    Tj_r[ind] = {}
    Tj_bar_r[ind] = {}
    Tj_r_energy[ind] = {}
    Tj_bar_r_energy[ind] = {}
    for row in A_r_r['Existing']:
        #total economic activity for RoW and Rest of ON in $
        Tj_r[ind][row] = \
        x_rr_i['Existing'][row].values.sum(axis=0) - x_r_he[ind][row].values.sum(axis=0)
        #total economic activity for RoW and Rest of ON in %        
        Tj_bar_r[ind][row] = \
        100*(x_rr_i['Existing'][row].values.sum(axis=0) - x_r_he[ind][row].values.sum(axis=0))\
        /x_rr_i['Existing'][row].values.sum(axis=0)
        # total energy use for RoW and Rest of ON in TJ
        Tj_r_energy[ind][row] = \
        x_rr_i_energy[row].values.sum(axis=0) - x_r_he_energy[ind][row].values.sum(axis=0)
        # total energy use for RoW and Rest of ON in %        
        Tj_bar_r_energy[ind][row] = \
        100*(x_rr_i_energy[row].values.sum(axis=0) - x_r_he_energy[ind][row].values.sum(axis=0))\
        /x_rr_i_energy[row].values.sum(axis=0)

""" turn HEM reslts into dataframe to plot bar chart"""
# regions
a1 = {}
a2 = {}
a3 = {}
a4 = {}

a2_top10 = {}
a4_top10 = {}
for row in A_r_r['Existing']:
    a1[row] = pd.DataFrame(index = industry_hr, columns = ['Tj_r'])
    a2[row] = pd.DataFrame(index = industry_hr, columns = ['Tj_bar_r'])
    a3[row] = pd.DataFrame(index = industry_hr, columns = ['Tj_r_energy'])
    a4[row] = pd.DataFrame(index = industry_hr, columns = ['Tj_bar_r_energy'])
    
    for ind in hypo_ex:    
        a1[row].loc[ind,'Tj_r'] = float(Tj_r[ind][row] )
        a2[row].loc[ind,'Tj_bar_r'] = float(Tj_bar_r[ind][row] )
        a3[row].loc[ind,'Tj_r_energy'] = float(Tj_r_energy[ind][row] )
        a4[row].loc[ind,'Tj_bar_r_energy'] = float(Tj_bar_r_energy[ind][row] )
    a1[row] = a1[row].apply(pd.to_numeric)
    a2[row] = a2[row].apply(pd.to_numeric)
    a3[row] = a3[row].apply(pd.to_numeric)
    a4[row] = a4[row].apply(pd.to_numeric)
    
    # drop household
    a2[row] = a2[row].drop(['Household'])
    a4[row] = a4[row].drop(['Household'])
    
    a2_top10[row] = a2[row].sort_values('Tj_bar_r',ascending=False).head(20)
    a4_top10[row] = a4[row].sort_values('Tj_bar_r_energy',ascending=False).head(30)
for row in A_r_r['Existing']:
    ax = a2_top10[row].plot(y=['Tj_bar_r'],kind = 'bar',width = 0.7, figsize = (13,4), legend = False)      
###############################################################
# from https://stackoverflow.com/questions/20394091/pandas-matplotlib-make-one-color-in-barplot-stand-out
    for bar in ax.patches:
        bar.set_facecolor('#888888')
    highlight = 'Residential building construction'
    highlight2 = 'Non-residential building construction'
    pos = a2_top10[row].index.get_loc(highlight)   
    pos2= a2_top10[row].index.get_loc(highlight2)   
    ax.patches[pos].set_facecolor('#aa3333')
    ax.patches[pos2].set_facecolor('#aa3333')
################################################################3
    plt.xticks(rotation = 30 , ha = 'right')
    plt.xlabel('industries', fontsize = 10, fontweight ='bold')
    plt.ylabel('Total economic activity\n measured in output (%)', fontsize = 10, fontweight ='bold')
    plt.title(row, fontsize = 14, fontweight = 'bold')
    plt.grid(linestyle = '--', linewidth = 0.6)
    plt.show()

for row in A_r_r['Existing']:
    ax = a4_top10[row].plot(y=['Tj_bar_r_energy'],kind = 'bar',width = 0.6, figsize = (13,4), legend = False)      
###############################################################
# from https://stackoverflow.com/questions/20394091/pandas-matplotlib-make-one-color-in-barplot-stand-out
    for bar in ax.patches:
        bar.set_facecolor('#888888')
    highlight = 'Residential building construction'
    highlight2 = 'Non-residential building construction'
    pos = a4_top10[row].index.get_loc(highlight)   
    pos2= a4_top10[row].index.get_loc(highlight2)   
    ax.patches[pos].set_facecolor('#aa3333')
    ax.patches[pos2].set_facecolor('#aa3333')
################################################################3
    plt.xticks(rotation = 30 , ha = 'right')
    plt.xlabel('industries', fontsize = 10, fontweight ='bold')
    plt.ylabel('Total economic activity\n measured in energy (%)', fontsize = 10, fontweight ='bold')
    plt.title(row, fontsize = 14, fontweight = 'bold')
    plt.grid(linestyle = '--', linewidth = 0.6)
    plt.show()
   
""" Hypothetical Extraction for Ontario """
""" note that provincial HEM is done in above code """
""" turn Ontario HEM reslts into dataframe to plot bar chart"""
a1 = {}
a2 = {}
a3 = {}
a4 = {}

a2_top10 = {}
a4_top10 = {}

a1 = pd.DataFrame(index = industry_hr, columns = ['Tj'])
a2 = pd.DataFrame(index = industry_hr, columns = ['Tj_bar'])
a3 = pd.DataFrame(index = industry_hr, columns = ['Tj_energy'])
a4 = pd.DataFrame(index = industry_hr, columns = ['Tj_bar_energy'])
    
for ind in hypo_ex:    
    a1.loc[ind,'Tj'] = float(Tj[ind].values )
    a2.loc[ind,'Tj_bar'] = float(Tj_bar[ind].values)
    a3.loc[ind,'Tj_energy'] = float(Tj_energy[ind].values)
    a4.loc[ind,'Tj_bar_energy'] = float(Tj_bar_energy[ind].values )
a1 = a1.apply(pd.to_numeric)
a2 = a2.apply(pd.to_numeric)
a3 = a3.apply(pd.to_numeric)
a4 = a4.apply(pd.to_numeric)

# drop households
a2 = a2.drop(['Household'])
a4 = a4.drop(['Household'])
a2_top10 = a2.sort_values('Tj_bar',ascending=False).head(20)
a4_top10 = a4.sort_values('Tj_bar_energy',ascending=False).head(30)

ax = a2_top10.plot(y=['Tj_bar'],kind = 'bar',width = 0.7, figsize = (13,4), legend = False)      
###############################################################
# from https://stackoverflow.com/questions/20394091/pandas-matplotlib-make-one-color-in-barplot-stand-out
for bar in ax.patches:
    bar.set_facecolor('#888888')
highlight = 'Residential building construction'
highlight2 = 'Non-residential building construction'
pos = a2_top10.index.get_loc(highlight)   
pos2= a2_top10.index.get_loc(highlight2)   
ax.patches[pos].set_facecolor('#aa3333')
ax.patches[pos2].set_facecolor('#aa3333')
################################################################3
plt.xticks(rotation = 30 , ha = 'right')
plt.xlabel('industries', fontsize = 10, fontweight ='bold')
plt.ylabel('Total economic activity\n measured in output (%)', fontsize = 10, fontweight ='bold')
plt.title('Ontario', fontsize = 14, fontweight = 'bold')
plt.grid(linestyle = '--', linewidth = 0.6)
plt.show()

ax = a4_top10.plot(y=['Tj_bar_energy'],kind = 'bar',width = 0.6, figsize = (13,4), legend = False)      
###############################################################
# from https://stackoverflow.com/questions/20394091/pandas-matplotlib-make-one-color-in-barplot-stand-out
for bar in ax.patches:
    bar.set_facecolor('#888888')
highlight = 'Residential building construction'
highlight2 = 'Non-residential building construction'
pos = a4_top10.index.get_loc(highlight)   
pos2= a4_top10.index.get_loc(highlight2)   
ax.patches[pos].set_facecolor('#aa3333')
ax.patches[pos2].set_facecolor('#aa3333')
################################################################3
plt.xticks(rotation = 30 , ha = 'right')
plt.xlabel('industries', fontsize = 10, fontweight ='bold')
plt.ylabel('Total economic activity \nmeasured in energy (%)', fontsize = 10, fontweight ='bold')
plt.title('Ontario', fontsize = 14, fontweight = 'bold')
plt.grid(linestyle = '--', linewidth = 0.6)
plt.show()

""" Hypothetical Extraction Backward Linkage - Regions """
A_he_bl = {}
#f_he_bl = {}
for ind in hypo_ex_hh:
    A_he_bl[ind]=copy.deepcopy(A_row_col['Existing'])
#    f_he_bl[ind]=copy.deepcopy(f_rr['Existing'])
    for row in A_r_r['Existing']:
        for col in A_r_r['Existing']:
            # delete columns of industry 
            A_he_bl[ind][row][col].loc[:,ind] *= 0

A_irio_temp = {}
A_irio_he_bl = {}
for ind in hypo_ex_hh:
    A_irio_temp[ind] = {}
    for row in A_r_r['Existing']:
        A_irio_temp[ind][row] = {}
        #concatenate each row of giant A and I
        A_irio_temp[ind][row] = np.concatenate([A_he_bl[ind][row][col] for col in sorted(A_r_r['Existing'])],1) 
    A_irio_he_bl[ind] = np.concatenate([A_irio_temp[ind][row] for row in sorted(A_r_r['Existing'])],0)
    
x_irio_he_bl = {}
he_bl = {}                                                                      # backward linkage in irio dimensions (454x1)
he_bl_r = {}                                                                    # backward linkage by region
he_bl_r_tot = {}                                                                # total backward linkage by region
for ind in hypo_ex_hh:
    x_irio_he_bl[ind] = np.linalg.inv(I_irio['Existing']-A_irio_he_bl[ind])*f_rr_irio['Existing']
    
    he_bl[ind] = (x_r_r_irio['Existing'] - x_irio_he_bl[ind])
    
    he_bl[ind] = pd.DataFrame(data = he_bl[ind],
         index = pd.concat([industry_hr,industry_hr],axis=0),
         columns = ['Hypothetical Extraction - Backward Linkage'])
    
    he_bl_r[ind] = {}
    he_bl_r_tot[ind] = {}
    # split hypothetical backward linkages into the two regions
    IRIO_split(he_bl_r[ind],he_bl[ind])
   
    for row in A_r_r['Existing']:
        he_bl_r_tot[ind][row] = 100* he_bl_r[ind][row].values.sum(axis=0)/x_rr_i['Existing'][row].loc[ind,:].values.sum(axis=0)
#        he_bl_r_tot[ind][row] =  (x_r_r_irio['Existing'].sum(axis=0) - x_irio_he_bl[ind].sum(axis=0))\
#        /x_rr_i['Existing'][row].loc[ind,:].values.sum(axis=0)

        he_bl_r[ind][row] =100* he_bl_r[ind][row]/x_rr_i['Existing'][row].loc[ind,:].values

""" Hypothetical Extraction Backward Linkage - Ontario """
A_he_bl_ont = {}
temp = {}
#f_he_bl = {}
for ind in hypo_ex_hh:
    A_he_bl_ont[ind]= pd.DataFrame(data = copy.deepcopy(A_ixi_scrub['Existing']), 
                               index = industry_hr, columns = industry_hc)

    temp[ind] = pd.DataFrame(data = copy.deepcopy(x_ar['Existing']),
        index = industry_hr)
    # delete columns of industry 
    A_he_bl_ont[ind].loc[:,ind] *= 0

x_he_bl_ont = {}
he_bl_ont = {}                                                              # total backward linkage by region
he_bl_ont_tot = {}
for ind in hypo_ex_hh:
    x_he_bl_ont[ind] = np.linalg.inv(np.identity(len(industry_hr)) - A_he_bl_ont[ind])*f_scrubbed['Existing']
    
    he_bl_ont[ind] = (x_ar['Existing'] - x_he_bl_ont[ind])
    
    he_bl_ont[ind] = pd.DataFrame(data = he_bl_ont[ind],
         index = industry_hr,
         columns = ['Hypothetical Extraction - Backward Linkage'])
  
    he_bl_ont_tot[ind] = 100*he_bl_ont[ind].values.sum(axis=0)/temp[ind].loc[ind,:].values.sum(axis=0)
    
    he_bl_ont[ind] = 100* he_bl_ont[ind]/temp[ind].loc[ind,:].values
    



""" Hypothetical Extraction Forward Linkage - Regions """

""" is expected to be vector of zeroes since the residential and non residential
 industry rows are all zeroes """

B_he_fl = {}
v_he_fl = {}
x_prime_ghosh = {}
x_he_prime_ghosh = {}
he_fl_r = {}
he_fl_r_tot = {}
for ind in hypo_ex_hh:
    B_he_fl[ind] = copy.deepcopy(B_ghosh)
    v_he_fl[ind] = copy.deepcopy(w_rr['Existing'])
    x_he_prime_ghosh[ind] = {}
    he_fl_r[ind] = {}
    he_fl_r_tot[ind] = {}
    for row in A_r_r['Existing']:
        B_he_fl[ind][row].loc[ind,:] *=0
        x_prime_ghosh[row] = w_rr['Existing'][row].values*\
        np.matrix(np.linalg.inv(np.identity(len(industry_hr))-B_ghosh[row].values))
        
        x_he_prime_ghosh[ind][row] = w_rr['Existing'][row].values*\
        np.matrix(np.linalg.inv(np.identity(len(industry_hr))-B_he_fl[ind][row].values))
        he_fl_r[ind][row] = \
        100*(x_prime_ghosh[row] - x_he_prime_ghosh[ind][row])/x_rr_i['Existing'][row].loc[ind,:].values
       
        he_fl_r[ind][row] = pd.DataFrame(data = he_fl_r[ind][row],
               columns = industry_hc,
               index = ['Hypothetical Extraction - Forward Linkage'])
        he_fl_r[ind][row] = he_fl_r[ind][row].T
        he_fl_r_tot[ind][row] = \
            100*(x_prime_ghosh[row] - x_he_prime_ghosh[ind][row]).sum(axis=1)\
                        /x_rr_i['Existing'][row].loc[ind,:].values.sum(axis=0)

#        he_fl_r_tot[ind][row] = \
#            (x_prime_ghosh[row].sum(axis=1) - x_he_prime_ghosh[ind][row].sum(axis=1))\
#                        /x_rr_i['Existing'][row].loc[ind,:].values.sum(axis=0)
                        
""" Hypothetical Extraction Forward Linkage - ONtario """                   
B_he_fl_ont = {}
v_he_fl_ont = {}
x_prime_ghosh_ont = {}
x_he_prime_ghosh_ont = {}
he_fl_ont = {}
he_fl_ont_tot = {}
for ind in hypo_ex_hh:
    B_he_fl_ont[ind] = copy.deepcopy(B_ghosh_ont)
    v_he_fl_ont[ind] = pd.DataFrame(data = copy.deepcopy(v_L_hh['Existing']),
                       index = ['Total other value added'],
                       columns = industry_hc)      
    B_he_fl_ont[ind].loc[ind,:] *= 0         
    x_prime_ghosh_ont = v_L_hh['Existing']*\
            np.matrix(np.linalg.inv(np.identity(len(industry_hr))-B_ghosh_ont))
    x_he_prime_ghosh_ont[ind] = v_L_hh['Existing']*\
            np.matrix(np.linalg.inv(np.identity(len(industry_hr))-B_he_fl_ont[ind].values))
    he_fl_ont[ind] = 100*(x_prime_ghosh_ont - x_he_prime_ghosh_ont[ind])/temp[ind].loc[ind,:].values
    he_fl_ont[ind] = pd.DataFrame(data = he_fl_ont[ind],
       columns = industry_hc,
       index = ['Hypothetical Extraction - Forward Linkage'])
    he_fl_ont[ind] = he_fl_ont[ind].T
    he_fl_ont_tot[ind] =100* ( x_prime_ghosh_ont - x_he_prime_ghosh_ont[ind]).sum(axis=1)\
                        /temp[ind].loc[ind,:].values.sum(axis=0)
    
""" Graphics """
##ontario_total_output = x_r_r_irio.sum(axis=0)
## Pie chart for Tj
#labels = [row for row in sorted(Tj)]                                            # labels for residential and non-residential
#labels.append('Rest of Industries')                                             # append third label for pie chart
#size = [np.asscalar(Tj[row].values) for row in sorted(Tj)]                      # values for residential and non-residential
#size.append(
#        np.asscalar(x_r_r_irio['Existing'].sum(axis=0)-\
#            sum(Tj[row].values for row in sorted(Tj))
#                  ))                                                            # append rest of industry values
#explode = (0.1,0.1,0)                                                           # explode first two 
#fig1, ax1 = plt.subplots()
#ax1.pie(size, explode = explode, autopct='%1.1f%%',pctdistance=0.8)
#plt.legend(labels, loc=(-0.05, -0.2), shadow=True)
#my_circle=plt.Circle( (0,0), 0.6, color='white')
#p=plt.gcf()
#p.gca().add_artist(my_circle)
#kwargs = dict(size=15, fontweight='bold', va='center')
#ax1.text(0, 0, '$1.65B\nCAD', ha='center', **kwargs)
#
#ax1.axis('equal')                                                             # Equal aspect ratio ensures that pie is drawn as a circle.
#ax1.set_title('Ontario Total Economic Output', fontweight ='bold')
#plt.show()
#
#RoW_total_output = x_rr_i['Existing']['RoW'].sum(axis=0)
#size_RoW = [np.asscalar(Tj_r[row]['RoW']) for row in sorted(Tj)]
#size_RoW.append(
#        np.asscalar(x_rr_i['Existing']['RoW'].sum(axis=0)-\
#                    sum(Tj_r[row]['RoW'] for row in sorted(Tj))
#                    ))
#fig2, ax2 = plt.subplots()
#ax2.pie(size_RoW, explode = explode, autopct='%1.1f%%',pctdistance=0.8)
#plt.legend(labels, loc=(-0.05, -0.2), shadow=True)
#my_circle=plt.Circle( (0,0), 0.6, color='white')
#p=plt.gcf()
#p.gca().add_artist(my_circle)
#kwargs = dict(size=15, fontweight='bold', va='center')
#ax2.text(0, 0, '$71.2M\nCAD', ha='center', **kwargs)
#
#ax2.axis('equal')                                                             # Equal aspect ratio ensures that pie is drawn as a circle.
#ax2.set_title('Region of Waterloo Total Economic Output', fontweight ='bold')
#plt.show()
#
## bar chart for hypothetical extraction
#top_heblr = {}
#for ind in hypo_ex:
#    top_heblr[ind] ={}
#    for row in A_r_r['Existing']:
#        top_heblr[ind][row] = he_bl_r[ind][row].sort_values(
#                'Hypothetical Extraction - Backward Linkage',
#                ascending=False).head(10)
#
## Plotting comparisons
# 
#    f = plt.figure(figsize=(0.5,0.5))
#    top_heblr[ind]['RoW'].plot(kind ='bar',legend = False)
#    plt.xticks(rotation = 30 , ha = 'right')
#    plt.xlabel('Industries', fontsize = 10, fontweight ='bold')
#    plt.ylabel('%', fontsize = 12, fontweight ='bold')
#    plt.title('Top Backward linkages for \n' + ind, fontsize = 14, fontweight = 'bold')
#    plt.show()
#
#
#
""" 
RoW: Donut chart of top 10 technical coefficients (Using A matrix)
"""
donut = ['Residential building construction',
      'Non-residential building construction']                                  # industry to apply adaptive reuse 
top_aij = {}
for ind in donut:
    top_aij[ind] = {}
    for row in A_r_r['Existing']:
        top_aij[ind][row]= {}
        for col in A_r_r['Existing']:
            if row != col:
                next
            elif row == col:
                top_aij[ind][row][col]= \
                A_row_col['Existing'][row][col].sort_values(ind,ascending=False).head(10).loc[:,ind] #sorting construction industries by highest technical coefficient

def autopct_generator(limit):
    def inner_autopct(pct):
        return ('$%.2f' % pct) if pct > limit else ''
    return inner_autopct

labels = {}
size = {}
labels_1 = {}                                                                   # labels for values greater than 10
#test2 = {}
normsize = {}
labels_2 = {}                                                                   # labels used in legend. labels are the ones not shown in pie chart
for ind in donut:
    labels[ind] = top_aij[ind]['RoW']['RoW'].index.tolist()
#    labels[ind].extend(['Value-added other \nthan household labour','Other industries, and imports'])
    labels[ind].extend(['Value-added other \nthan household labour \nand imports','imports','Other industries'])

    size[ind] = top_aij[ind]['RoW']['RoW'].values.tolist()
#    size[ind].append(np.asscalar(gva_aij['Existing']['RoW'].loc[:,ind].values))
    size[ind].append(np.asscalar(gva_rr_aij['Existing']['RoW'].loc['Other value added',ind]))
    size[ind].append(np.asscalar(gva_rr_aij['Existing']['RoW'].loc['Imports',ind]))

    size[ind].append(1-sum(size[ind]))
#
#test = np.multiply(w_rr['Existing']['RoW'],v_prop['Existing'][1,:])
#test2 = np.divide(test,x_rr_j[ind][row].T.values)
### function from https://stackoverflow.com/questions/34035427/conditional-removal-of-labels-in-matplotlib-pie-chart
    
    labels_1[ind] = [n if v > sum(size[ind]) * 0.1 else ''
              for n, v in zip(labels[ind], size[ind])]                       

#    test2[ind] = [n if v < sum(size[ind]) * 0.1 else ''
#              for n, v in zip(labels[ind], size[ind])] 
### 
    
    explode = (0.1,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.1,0.1,0.1)                                                           # explode first two 
    fig1, ax1 = plt.subplots()
### code from https://stackoverflow.com/questions/46692643/how-to-show-filtered-legend-labels-in-pyplot-pie-chart-based-on-the-values-of-co   
    normsize[ind] = [ item / sum(size[ind])*100 for item in size[ind]]                 
### code from https://stackoverflow.com/questions/44076203/getting-percentages-in-legend-from-pie-matplotlib-pie-chart
    labels_2[ind] =[
            '$%1.2f, %s ' % (s, l) for s, l in zip( normsize[ind],labels[ind])
                    ]
###    
    p,t,a = ax1.pie(size[ind],
                    explode = explode,
                    autopct=autopct_generator(10),
                    pctdistance=0.75,\
                    labels = labels_1[ind])
    
    h,l = zip(*[(h,lab) for h,lab,ii in zip(p,labels_2[ind],normsize[ind]) if ii < 10])
    legend = plt.legend(h,l, loc=(-0.05, -0.8), shadow=True)
    legend.set_title('Top 10 Industry Recipes - \$ input per $100 output',
                     prop={'size':10, 'weight':'bold'})
    legend._legend_box.align = "left"
###    
    my_circle=plt.Circle( (0,0), 0.6, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    kwargs = dict(size= 10, fontweight='bold', va='center')
    ax1.text(0, 0, '$ \ninput per \n $100 \noutput', ha='center', **kwargs)
    ax1.axis('equal')                                                             # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title('Region of Waterloo\n' + ind + ' \n\'Industry recipe\'' , fontweight ='bold',fontsize =14)
#    ax1.set_title('Region of Waterloo:\n' + ind , fontweight ='bold',fontsize =14)

    plt.show()



""" 
Provincial: Donut chart of top 10 technical coefficients (Using B matrix)
"""
# need to turn B matrix into dataframe for simplicity of code
B_df = pd.DataFrame(data = B['Existing'], index = product, columns = industry)
# need to append HH labour coefficient to B matrix for visual purposes
temp = np.divide(gva.loc['Household labour',:].values  , x.T)
temp2 = pd.DataFrame(temp, index = ['Household labour'], columns = industry)
B_df = B_df.append(temp2)

# need to find coefficients for value-added (excluding wages and salaries)
temp3 = np.divide(gva.loc['Other value added',:].values , x.T)
temp4 = pd.DataFrame(data = temp3, index = ['Other value added'], columns = industry)
temp5 = np.divide(gva_scrubbed.loc['Imports',:].values,x.T)
temp6 = pd.DataFrame(data = temp5, index = ['Imports'],columns = industry)
#del temp

top_bij = {}
for ind in donut:
    top_bij[ind] = B_df.sort_values(ind,ascending=False).head(10).loc[:,ind] #sorting construction industries by highest technical coefficient

labels = {}
size = {}
labels_1 = {}                                                                   # labels for values greater than 10
#test2 = {}
normsize = {}
labels_2 = {}                                                                   # labels used in legend. labels are the ones not shown in pie chart
for ind in donut:
    labels[ind] = top_bij[ind].index.tolist()
    labels[ind].extend(['Value-added other \nthan household labour \nand imports','imports','Other commodities'])
    size[ind] = top_bij[ind].values.tolist()
    size[ind].append(np.asscalar(temp4.loc[:,ind].values))
    size[ind].append(np.asscalar(temp6.loc[:,ind].values))
    size[ind].append(1-sum(size[ind]))

### function from https://stackoverflow.com/questions/34035427/conditional-removal-of-labels-in-matplotlib-pie-chart
    
    labels_1[ind] = [n if v > sum(size[ind]) * 0.1 else ''
              for n, v in zip(labels[ind], size[ind])]                       

#    test2[ind] = [n if v < sum(size[ind]) * 0.1 else ''
#              for n, v in zip(labels[ind], size[ind])] 
### 
    
    explode = (0.1,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.1,0.1,0.1)                                                           # explode first two 
    fig1, ax1 = plt.subplots()
### code from https://stackoverflow.com/questions/46692643/how-to-show-filtered-legend-labels-in-pyplot-pie-chart-based-on-the-values-of-co   
    normsize[ind] = [ item / sum(size[ind])*100 for item in size[ind]]                 
### code from https://stackoverflow.com/questions/44076203/getting-percentages-in-legend-from-pie-matplotlib-pie-chart
    labels_2[ind] =[
            '$%1.2f, %s ' % (s, l) for s, l in zip( normsize[ind],labels[ind])
                    ]
###    
    p,t,a = ax1.pie(size[ind],
                    explode = explode,
                    autopct=autopct_generator(10),
                    pctdistance=0.75,\
                    labels = labels_1[ind])
    
    h,l = zip(*[(h,lab) for h,lab,ii in zip(p,labels_2[ind],normsize[ind]) if ii < 10])
    legend = plt.legend(h,l, loc=(-0.05, -0.8), shadow=True)
    legend.set_title('Top 10 Commodity Recipes - \$ input per $100 output',
                     prop={'size':10, 'weight':'bold'})
    legend._legend_box.align = "left"
###    
    my_circle=plt.Circle( (0,0), 0.6, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    kwargs = dict(size= 10, fontweight='bold', va='center')
    ax1.text(0, 0, '$ \ncommodity \ninput per \n $100 \noutput', ha='center', **kwargs)
    ax1.axis('equal')                                                             # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title('Ontario\n' + ind + ' \n\'Commodity recipe\'' , fontweight ='bold',fontsize =14)
#    ax1.set_title('Ontario:\n' + ind  , fontweight ='bold',fontsize =14)

    plt.show()
del temp,temp2,temp3,temp4, temp5, temp6
