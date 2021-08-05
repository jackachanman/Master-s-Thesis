# -*- coding: utf-8 -*-
"""
Created on Tue Feb 05 13:13:59 2019

@author: jacky
"""

import numpy as np
import pandas as pd

"""
THIS MODEL APPLIES METHOD 2 OF IMPORT SCRUBBING (EXPORT INCLUDED IN FINAL DEMAND)
THIS MODEL SCRUBS THE USE MATRIX
"""

"""
this code separates Existing case from AR cases

"""

""" 2019-02-13 revision: update calculation of xj in closing model.
    added IRIO_vector_split and IRIO_matrix_split for scenarios and existing
"""

""" 2019-02-18 revision: adjusted scenarios to include changing final demand """


""" march 07 2019 update changed lines 831 and 524 to properly split x_rr_i 
prior: x_rr_i was found by splitting x_i_irio << which is not final x_i. lines 513 and 820 is hashtagged (taken out of code use)"""

""" march 03-13 update removes the codes that pertains to RAS-ing aka 'Two-region logic with more than two regions' calculations of f_rr and w_rr
 were incorrect in previous versions. code has been simplified to just 'Two-regions' and calculations of f_rr and w_rr and x_rr_i were corrected """

"""2019-03-18 update: travel expenditure by non-residents and residents added to
household final consumption expenditure, from ONTARIO model, wages and salaries, and self employment 
is added to Z matrix """

""" 2019-04-23 update: extended v_prop to include hh column """

""" 2019-05-23 deleted unused code """
def IRIO_concat_existing(A_row_col,I_row_col,Z_row_col,f_rr, x_r_r,w_rr,
                         A_irio_temp,I_irio_temp,Z_irio_temp):
    # re-concatenate
    for row in A_r_r[ind]:
        A_irio_temp[row] = {}
        I_irio_temp[row] = {}
        Z_irio_temp[row] = {}
        #concatenate each row of giant A and I
        A_irio_temp[row] = np.concatenate([A_row_col[row][col] for col in sorted(A_r_r[ind])],1) 
        I_irio_temp[row] = np.concatenate([I_row_col[row][col] for col in sorted(A_r_r[ind])],1)
        Z_irio_temp[row] = np.concatenate([Z_row_col[row][col] for col in sorted(A_r_r[ind])],1)
    
    # creating giant matrices/vectors
    global A_irio, I_irio, Z_irio, f_rr_irio, x_r_r_irio, w_rr_irio
    
    A_irio[ind] = np.concatenate([A_irio_temp[row] for row in sorted(A_r_r[ind])],0)
    I_irio[ind] = np.concatenate([I_irio_temp[row] for row in sorted(A_r_r[ind])],0)
    Z_irio[ind] = np.concatenate([Z_irio_temp[row] for row in sorted(A_r_r[ind])],0)
    f_rr_irio[ind] = np.concatenate([f_rr[row] for row in sorted(A_r_r[ind])],0)
    x_r_r_irio[ind] = np.concatenate([x_r_r[row] for row in sorted(A_r_r[ind])],0)
    w_rr_irio[ind] = np.concatenate([w_rr[row] for row in sorted(A_r_r[ind])],1)

""" IRIO concatenation function is needed for scenarios because it has different
dictionary structure than 'Existing' """
    
def IRIO_concat_scenario(A_row_col,I_row_col,Z_row_col,f_rr, x_r_r,w_rr,
                         A_irio_temp,I_irio_temp,Z_irio_temp):
    # re-concatenate
    for row in A_r_r[ind][s]:
        A_irio_temp[row] = {}
        I_irio_temp[row] = {}
        Z_irio_temp[row] = {}
        #concatenate each row of giant A and I
        A_irio_temp[row] = np.concatenate([A_row_col[row][col] for col in sorted(A_r_r[ind][s])],1) 
        I_irio_temp[row] = np.concatenate([I_row_col[row][col] for col in sorted(A_r_r[ind][s])],1)
        Z_irio_temp[row] = np.concatenate([Z_row_col[row][col] for col in sorted(A_r_r[ind][s])],1)
    
    # creating giant matrices/vectors
    global A_irio, I_irio, Z_irio, f_rr_irio, x_r_r_irio, w_rr_irio
    
    A_irio[ind][s] = np.concatenate([A_irio_temp[row] for row in sorted(A_r_r[ind][s])],0)
    I_irio[ind][s] = np.concatenate([I_irio_temp[row] for row in sorted(A_r_r[ind][s])],0)
    Z_irio[ind][s] = np.concatenate([Z_irio_temp[row] for row in sorted(A_r_r[ind][s])],0)
    f_rr_irio[ind][s] = np.concatenate([f_rr[row] for row in sorted(A_r_r[ind][s])],0)
    x_r_r_irio[ind][s] = np.concatenate([x_r_r[row] for row in sorted(A_r_r[ind][s])],0)
    w_rr_irio[ind][s] = np.concatenate([w_rr[row] for row in sorted(A_r_r[ind][s])],1)   


"""Create function to split two-region industry output into output for each region.
IRIO conatenation function needed for 'Existing' scenario because it has
different dictionary structure"""

def IRIO_vector_split_existing(new_dict, irio_var) :
    new_dict['Rest of ON'] = irio_var.iloc[0:2*len(industry_hr)/len(A_r_r[ind]),:]
    new_dict['RoW'] = irio_var.iloc[2*len(industry_hr)/len(A_r_r[ind]):2*len(industry_hr),:]
def IRIO_matrix_split_existing(new_dict, irio_var):
    new_dict['Rest of ON']['Rest of ON'] = \
                    irio_var.iloc[0:2*len(industry_hr)/len(A_r_r[ind]),
                                  0:2*len(industry_hr)/len(A_r_r[ind])]
    new_dict['Rest of ON']['RoW'] = \
                    irio_var.iloc[0:2*len(industry_hr)/len(A_r_r[ind]),
                                  2*len(industry_hr)/len(A_r_r[ind]):2*len(industry_hr)]
                    
    new_dict['RoW']['Rest of ON'] = irio_var.iloc[2*len(industry_hr)/len(A_r_r[ind]):2*len(industry_hr),
                                    0:2*len(industry_hr)/len(A_r_r[ind])]
    new_dict['RoW']['RoW'] = irio_var.iloc[2*len(industry_hr)/len(A_r_r[ind]):2*len(industry_hr),
                                           2*len(industry_hr)/len(A_r_r[ind]):2*len(industry_hr)]

"""Create function to split two-region industry output into output for each region.
IRIO concatenation function is needed for scenarios because it has different
dictionary structure than 'Existing'"""   

def IRIO_vector_split_scenario(new_dict, irio_var) :
    new_dict['Rest of ON'] = irio_var.iloc[0:2*len(industry_hr)/len(A_r_r[ind][s]),:]
    new_dict['RoW'] = irio_var.iloc[2*len(industry_hr)/len(A_r_r[ind][s]):2*len(industry_hr),:]
def IRIO_matrix_split_scenario(new_dict, irio_var):
    new_dict['Rest of ON']['Rest of ON'] = \
                    irio_var.iloc[0:2*len(industry_hr)/len(A_r_r[ind][s]),
                                  0:2*len(industry_hr)/len(A_r_r[ind][s])]
    new_dict['Rest of ON']['RoW'] = \
                    irio_var.iloc[0:2*len(industry_hr)/len(A_r_r[ind][s]),
                                  2*len(industry_hr)/len(A_r_r[ind][s]):2*len(industry_hr)]
                    
    new_dict['RoW']['Rest of ON'] = irio_var.iloc[2*len(industry_hr)/len(A_r_r[ind][s]):2*len(industry_hr),
                                    0:2*len(industry_hr)/len(A_r_r[ind][s])]
    new_dict['RoW']['RoW'] = irio_var.iloc[2*len(industry_hr)/len(A_r_r[ind][s]):2*len(industry_hr),
                                           2*len(industry_hr)/len(A_r_r[ind][s]):2*len(industry_hr)]

 
""" Regionalizing Ontario Model for Region of Waterloo """
# METHOD 1: back calculating f_r
x_r = {}
Z_r = {}
A_r = {}

A_r_r = {}                                                                      # creating new A, Z, and x specific for RAS of IRIO
A_sr_r = {}
A_sr_sr = {}
A_r_sr = {}
Z_r_r = {}
Z_sr_r = {}
Z_sr_sr = {}
Z_r_sr = {}
x_r_r = {}
x_sr = {}

Z_row_col = {}
Z_row_marg = {}                                                                 # creating column and row margins to re-introduce into RAS
Z_row_marg_sum = {}
Z_col_marg = {}
Z_col_marg_sum = {}
#creating RAS function for interregional flows
r_ras = {}
s_ras = {}
# create dictionaries for RAS function
rows = {}
row_sum ={}
u_r_sr = {}

columns = {}
col_sum = {}
v_sr_r = {}

err_s = {}
err_r = {}
n_vs_err_s = {}
n_vs_err_r = {}

""" recalculate final demand and A matrices """
x_rr = {}
f_rr = {}
w_rr = {}
A_row_col = {}
I_row_col = {}

A_irio_temp = {}
Z_irio_temp = {}
I_irio_temp = {}

A_irio = {}
Z_irio = {}
I_irio = {}
L_irio = {}

f_rr_irio = {}
x_r_r_irio = {}
w_rr_irio = {}

x_i_irio = {}
x_j_irio = {}

""" Closing wrt HH """
""" splitting f_rr into con_ex, cap_form, export """
#D_f_export_prop = {}
D_f_con_ex_rr = {}
D_f_cap_form_rr = {}
D_f_export_rr = {}

#test = sorted(list(employ))
""" concatenating appropriately """
z_hc = {}
z_hc2={}
z_hr = {}
x_rr_i = {}
x_rr_j = {}

""" calculate gross value added technical coefficients"""
gva_aij = {}

""" Energy Intensities IRIO dimensions"""
##censored data not estimated
#energy_int = pd.read_csv('Energy_intensities_ordered.csv').set_index('SUIC')  

##estimated censored data
#energy_int = pd.read_csv('Energy_intensities_ordered - estimated censored industries.csv').set_index('SUIC')  

energy_int_irio = np.concatenate([energy_int,energy_int],0)                     # concatenate energy intensities into irio dimensions (454x1)
x_r_r_irio_energy = {}

""" create vector of ones for irio """
i_irio = np.matrix(np.ones((2*len(industry_hr),1)))
i_hh = np.matrix(np.ones((len(industry_hr),1)))

test = {}
A_irio2 = {}
test2 = {}
test3 = {}

hh_diff = {}

v_prop_sum = {}

#hh_prop = {}

# proportion of w_rr to v_L
w_rr_prop = {}

# proportion of wage for each region
wage_rr_prop = {}

# proportion of f_rr to f_scrubbed
f_rr_prop = {}

# proportion of hh for each region
hh_rr_prop = {}


z_hr_prop = {}
hh_rr_diff = {}

# down scaling final demand by population
f_rr2 = {}
diff = {}

# updating v_prop and calculating value added coefficients for plots
gva_rr = {}
gva_rr_aij = {}

for ind in ar:
    x_r[ind] = {}
    A_r[ind] = {}
    Z_r[ind] = {}
    
    A_r_r[ind] = {}                                                                      # creating new A, Z, and x specific for RAS of IRIO
    A_sr_r[ind] = {}
    A_sr_sr[ind] = {}
    A_r_sr[ind] = {}
    Z_r_r[ind] = {}
    Z_sr_r[ind] = {}
    Z_sr_sr[ind] = {}
    Z_r_sr[ind] = {}
    x_r_r[ind] = {}
    x_sr[ind] = {}
    
    Z_row_col[ind] = {}
    Z_row_marg[ind] = {}                                                                 # creating column and row margins to re-introduce into RAS
    Z_row_marg_sum[ind] = {}
    Z_col_marg[ind] = {}
    Z_col_marg_sum[ind] = {}
    
    r_ras[ind] = {}
    s_ras[ind] = {}    
    err_s[ind] = {}
    err_r[ind] = {}
    
    rows[ind] = {}
    row_sum[ind] ={}
    u_r_sr[ind] = {}
    
    columns[ind] = {}
    col_sum[ind] = {}
    v_sr_r[ind] = {}
    
    n_vs_err_s[ind] = {}
    n_vs_err_r[ind] = {}
    
    x_rr[ind] ={}
    f_rr[ind] = {}
    w_rr[ind] = {}
    A_row_col[ind] = {}
    I_row_col[ind] = {}
    
    A_irio_temp[ind] = {}
    Z_irio_temp[ind] = {}
    I_irio_temp[ind] = {}
    
    A_irio[ind] = {}
    Z_irio[ind] = {}
    I_irio[ind] = {}
    L_irio[ind] = {}

    f_rr_irio[ind] = {}
    x_r_r_irio[ind] = {}
    w_rr_irio[ind] = {}
    x_i_irio[ind] = {}
    x_j_irio[ind] = {}
    
    z_hc[ind] = {}
    z_hc2[ind] = {}
    z_hr[ind] = {}
    x_rr_i[ind] = {}
    x_rr_j[ind] = {}
    
    gva_aij[ind] = {}
    
    D_f_con_ex_rr[ind] = {}
    D_f_cap_form_rr[ind] = {}
    D_f_export_rr[ind] = {}
        
    x_r_r_irio_energy[ind] = {}
    
    test[ind] = {}
    test2[ind] = {}
    test3[ind] = {}
    hh_diff[ind] = {}
    v_prop_sum[ind] = {}
#    hh_prop[ind] = {}
    
    w_rr_prop[ind] ={}
    wage_rr_prop[ind] ={}
    
    f_rr_prop[ind] = {}
    hh_rr_prop[ind] = {}
    
    z_hr_prop[ind] = {}
    hh_rr_diff[ind] ={}
    
    
    f_rr2[ind] = {}
    diff[ind] = {}
    
    gva_rr[ind] = {}
    gva_rr_aij[ind] = {}

    if ind == 'Existing':
        for j in list(employ):    
            x_r[ind][j] = np.diag(agg[j].values.flatten())*x_ar[ind]                                 # regionalized by employment
            A_r[ind][j] = np.multiply(agg_CILQ[j] , A_ixi_scrub[ind])                             # calculate A_r 
            Z_r[ind][j] = A_r[ind][j]*np.diagflat(x_r[ind][j])
    
            # down scale final demand (just the household portion)
#            f_rr2[ind][j] = pop_2014.loc[j,'Proportion']*h_con_ex[ind] +\
#            np.diag(agg[j].values.flatten())*h_cap_form[ind] +\
#                   np.diag(agg[j].values.flatten())*h_export[ind]
            f_rr2[ind][j] = pop_2014.loc[j,'Proportion']*hh_scrubbed[ind]
            # concatenate 0 for comparisons
            f_rr2[ind][j] = np.vstack([f_rr2[ind][j],[0]])
         
        for j in list(employ):
        # step 1 & 2
            A_r_r[ind][j] = A_r[ind][j]
            A_sr_r[ind][j] = np.subtract(A_ixi_scrub[ind], A_r[ind][j])
            A_sr_sr[ind][j] =  np.multiply(agg_CILQ_rest[j], A_ixi_scrub[ind]) 
            A_r_sr[ind][j] = np.subtract(A_ixi_scrub[ind], A_sr_sr[ind][j])
            # step 3
            Z_r_r[ind][j] = Z_r[ind][j]
            x_r_r[ind][j] = copy.deepcopy(x_r[ind][j])
            x_sr[ind][j] = x_ar[ind] - x_r_r[ind][j]                                                       
            Z_sr_r[ind][j] = A_sr_r[ind][j] * np.diagflat(x_r[ind][j])
            Z_sr_sr[ind][j] = A_sr_sr[ind][j] * np.diagflat(x_sr[ind][j])
            Z_r_sr[ind][j] = A_r_sr[ind][j] * np.diagflat(x_sr[ind][j])

        #remove ON and Rest of ON in A_r_r, Z_r_r etc. for simplicity of RAS            # should i delete or just pass it in loops?
        del A_r_r[ind]['ON'] #,A_r_r['Rest of ON']
        del A_r_sr[ind]['ON'] #,A_r_sr['Rest of ON']
        del A_sr_r[ind]['ON']# ,A_sr_r['Rest of ON']  
        del A_sr_sr[ind]['ON']# ,A_sr_sr['Rest of ON']
        del Z_r_r[ind]['ON'] #,Z_r_r['Rest of ON']
        del Z_r_sr[ind]['ON']#,Z_r_sr['Rest of ON']
        del Z_sr_r[ind]['ON']# ,Z_sr_r['Rest of ON']  
        del Z_sr_sr[ind]['ON']# ,Z_sr_sr['Rest of ON']
        del x_r_r[ind]['ON'] #, x_r_r['Rest of ON']
        del x_sr[ind]['ON'] #, x_sr['Rest of ON']

        del f_rr2[ind]['ON']    
    
        for j in A_r_r[ind]:            
            if j != 'Rest of ON':                                               # j = 'Rest of ON' is used for consistant formating for conseuqent codes, A_r_r is technical coefficients for Rest of ON 
                next
            elif j == 'Rest of ON':                                                      
            
                f_rr[ind]['Rest of ON'] = x_r_r[ind][j]- (Z_r_r[ind][j] + Z_r_sr[ind][j])*i_hh  
                f_rr[ind]['RoW'] =  x_sr[ind][j]- (Z_sr_r[ind][j] + Z_sr_sr[ind][j])*i_hh 
                
                x_rr[ind]['Rest of ON'] = x_r_r[ind][j]
                x_rr[ind]['RoW'] = x_sr[ind][j]
                
                w_rr[ind]['Rest of ON'] = x_r_r[ind][j].T - i_hh.T*(Z_r_r[ind][j]+Z_sr_r[ind][j])
                w_rr[ind]['RoW'] =x_sr[ind][j].T - i_hh.T*(Z_r_sr[ind][j]+Z_sr_sr[ind][j])
                
                Z_row_col[ind]['Rest of ON'] = {}
                Z_row_col[ind]['Rest of ON']['Rest of ON'] = Z_r_r[ind][j]
                Z_row_col[ind]['Rest of ON']['RoW'] = Z_r_sr[ind][j]
                Z_row_col[ind]['RoW'] = {}
                Z_row_col[ind]['RoW']['Rest of ON'] = Z_sr_r[ind][j]
                Z_row_col[ind]['RoW']['RoW'] = Z_sr_sr[ind][j]
                
                A_row_col[ind]['Rest of ON'] = {}
                A_row_col[ind]['Rest of ON']['Rest of ON'] = A_r_r[ind][j]
                A_row_col[ind]['Rest of ON']['RoW'] = A_r_sr[ind][j]
                A_row_col[ind]['RoW'] = {}
                A_row_col[ind]['RoW']['Rest of ON'] = A_sr_r[ind][j]
                A_row_col[ind]['RoW']['RoW'] = A_sr_sr[ind][j]      
                
                I_row_col[ind]['Rest of ON'] = {}
                I_row_col[ind]['Rest of ON']['Rest of ON'] = np.identity(len(industry_hr))
                I_row_col[ind]['Rest of ON']['RoW'] = np.zeros_like(L[ind])
                I_row_col[ind]['RoW'] = {}
                I_row_col[ind]['RoW']['Rest of ON'] = np.zeros_like(L[ind])
                I_row_col[ind]['RoW']['RoW'] = np.identity(len(industry_hr))

#           test[ind][s] = f_rr[ind]['Rest of ON'] + f_rr[ind]'RoW']  -f_scrubbed[ind]  # check if sum to provincial
#           test[ind][s] = w_rr[ind]['Rest of ON'] + w_rr[ind]['RoW'] - v_L_hh[ind]       # check if sum to provincial
                
          
        IRIO_concat_existing(A_row_col[ind],I_row_col[ind],Z_row_col[ind],
                             f_rr[ind], x_rr[ind], w_rr[ind],
                             A_irio_temp[ind],I_irio_temp[ind],Z_irio_temp[ind])

        for row in A_r_r['Existing']:
            gva_rr[ind][row] = v_prop_hh[ind]*np.diagflat(w_rr[ind][row])       # w_rr does not have labour now (Model is closed at provincial level). using v-prop is wrong

        A_irio[ind] = pd.DataFrame(data = A_irio[ind],
                    index = pd.concat([industry_hr,industry_hr],axis=0),
                    columns = pd.concat([industry_hc,industry_hc],axis=0))
        x_i_irio[ind] = pd.DataFrame(data = x_r_r_irio[ind],
                index = pd.concat([industry_hr,industry_hr],axis=0),
                columns = ['Total Industry Output'])
        x_j_irio[ind] = pd.DataFrame(data = x_r_r_irio[ind].T,
                index = ['Total Industry Input'],
                columns = pd.concat([industry_hc,industry_hc],axis=0) )

        #""" 
        #need to split x_i_irio into x_rr_i
        #need to split x_j_irio into x_rr_j
        #need to split A_irio into A_row_col
        #"""
        IRIO_vector_split_existing(x_rr_i[ind],x_i_irio[ind])
        IRIO_vector_split_existing(x_rr_j[ind],x_j_irio[ind].T)        
        IRIO_matrix_split_existing(A_row_col[ind],A_irio[ind])
#        # concatenate row of zeroes for the new z_hr row
        for row in A_r_r[ind]:
            f_rr[ind][row] = pd.DataFrame(data = f_rr[ind][row],
                index = industry_hr,
                columns = ['Total Final Demand'])
            w_rr[ind][row] = pd.DataFrame(data = w_rr[ind][row],
                index = ['Total Gross Value Added'],
                columns = industry_hc)
            gva_aij[ind][row] = pd.DataFrame(
                data = np.divide(w_rr[ind][row].values,x_rr_j[ind][row].T.values),
                   index = ['Gross Value Added Technical Coefficient'],
                   columns = industry_hc)    

            for col in A_r_r[ind]:
                Z_row_col[ind][row][col] = pd.DataFrame(data=Z_row_col[ind][row][col],
                         index = industry_hr,
                         columns = industry_hc)

        for row in A_r_r[ind]:
#        #2nd row is for imports. value added for household only consists of imports.   recall household is moved into transaction matrix 
            gva_rr_aij[ind][row] = gva_rr[ind][row] * np.linalg.inv(np.diagflat(x_rr_j[ind][row].T.values))  
            gva_rr_aij[ind][row] = pd.DataFrame(data = gva_rr_aij[ind][row], 
                      index = ['Other value added','Imports'],
                      columns = industry_hc)
                
        x_r_r_irio_energy[ind] = np.diagflat(energy_int_irio)*x_r_r_irio[ind]                     # find energy use for x_irio
        x_r_r_irio_energy[ind] = pd.DataFrame(data = x_r_r_irio_energy[ind],
                                  index = pd.concat([industry_hc, industry_hc],axis=0),
                                  columns = ['Total Industry Energy Output (TJ)'])
    
    elif ind != 'Existing':
        for s in scenario:
            x_r[ind][s] = {}
            Z_r[ind][s] = {}
            A_r[ind][s] = {}
            
            A_r_r[ind][s] = {}                                                                      # creating new A, Z, and x specific for RAS of IRIO
            A_sr_r[ind][s] = {}
            A_sr_sr[ind][s] = {}
            A_r_sr[ind][s] = {}
            Z_r_r[ind][s] = {}
            Z_sr_r[ind][s] = {}
            Z_sr_sr[ind][s] = {}
            Z_r_sr[ind][s] = {}
            x_r_r[ind][s] = {}
            x_sr[ind][s] = {}
            
            Z_row_col[ind][s] = {}
            
            Z_row_marg[ind][s] = {}                                                                 # creating column and row margins to re-introduce into RAS
            Z_row_marg_sum[ind][s] = {}
            Z_col_marg[ind][s] = {}
            Z_col_marg_sum[ind][s] = {}
    
            r_ras[ind][s] = {}
            s_ras[ind][s] = {}
            
            rows[ind][s] = {}
            row_sum[ind][s] ={}
            u_r_sr[ind][s] = {}
            
            columns[ind][s] = {}
            col_sum[ind][s] = {}
            v_sr_r[ind][s] = {}
            
            err_s[ind][s] = {}
            err_r[ind][s] = {}
            n_vs_err_s[ind][s] = {}
            n_vs_err_r[ind][s] = {}     
        
            x_rr[ind][s] = {}
            f_rr[ind][s] = {}
            w_rr[ind][s] = {}
            A_row_col[ind][s] = {}
            I_row_col[ind][s] = {}
            
            A_irio_temp[ind][s] = {}
            Z_irio_temp[ind][s] = {}
            I_irio_temp[ind][s] = {}
            
            A_irio[ind][s] = {}
            Z_irio[ind][s] = {}
            I_irio[ind][s] = {}
            L_irio[ind][s] = {}
            
            f_rr_irio[ind][s] = {}
            x_r_r_irio[ind][s] = {}
            w_rr_irio[ind][s] = {}
            x_i_irio[ind][s] = {}
            x_j_irio[ind][s] = {}
        
            z_hc[ind][s] = {}
            z_hc2[ind][s]= {}
            z_hr[ind][s] = {}
            x_rr_i[ind][s] = {}
            x_rr_j[ind][s] = {}
            
            gva_aij[ind][s] = {}
            
            hh_diff[ind][s] = {}
            test3[ind][s] = {}
                        
#            hh_prop[ind][s] = {}
            
            hh_rr_diff[ind][s] = {}
            
            w_rr_prop[ind][s] ={}
            wage_rr_prop[ind][s] ={}
            
            f_rr_prop[ind][s] = {}
            hh_rr_prop[ind][s] = {}
            
            z_hr_prop[ind][s] = {}

            f_rr2[ind][s] = {}
            diff[ind][s] = {}

            gva_rr[ind][s] = {}
            gva_rr_aij[ind][s] = {}


            for j in list(employ):                                                          
                x_r[ind][s][j] = np.diag(agg[j].values.flatten())*x_ar[ind][s]                                 # regionalized by employment
                A_r[ind][s][j] = np.multiply(agg_CILQ[j] , A_ixi_scrub[ind][s])                             # calculate A_r 
                Z_r[ind][s][j] = A_r[ind][s][j]*np.diagflat(x_r[ind][s][j])
                
                # down scaling final demand (household consumption) by population
                f_rr2[ind][s][j] = pop_2014.loc[j,'Proportion']*hh_scrubbed[ind][s]
#                f_rr2[ind][s][j] = pop_2014.loc[j,'Proportion']*h_con_ex[ind][s] +\
#                        np.diag(agg[j].values.flatten())*h_cap_form[ind][s] +\
#                        np.diag(agg[j].values.flatten())*h_export[ind][s]

                # concatenate 0 to f_rr2 for comparisons
                f_rr2[ind][s][j] = np.vstack([f_rr2[ind][s][j],[0]])
                
            for j in list(employ):
                # step 1 & 2
                A_r_r[ind][s][j] = A_r[ind][s][j]
                A_sr_r[ind][s][j] = np.subtract(A_ixi_scrub[ind][s], A_r[ind][s][j])
                A_sr_sr[ind][s][j] =  np.multiply(agg_CILQ_rest[j], A_ixi_scrub[ind][s]) 
                A_r_sr[ind][s][j] = np.subtract(A_ixi_scrub[ind][s], A_sr_sr[ind][s][j])
                # step 3
                Z_r_r[ind][s][j] = Z_r[ind][s][j]
                x_r_r[ind][s][j] = copy.deepcopy(x_r[ind][s][j])
                x_sr[ind][s][j] = x_ar[ind][s] - x_r_r[ind][s][j]                                                       
                Z_sr_r[ind][s][j] = A_sr_r[ind][s][j] * np.diagflat(x_r[ind][s][j])
                Z_sr_sr[ind][s][j] = A_sr_sr[ind][s][j] * np.diagflat(x_sr[ind][s][j])
                Z_r_sr[ind][s][j] = A_r_sr[ind][s][j] * np.diagflat(x_sr[ind][s][j])


            #remove ON and Rest of ON in A_r_r, Z_r_r etc. for simplicity of RAS            # should i delete or just pass it in loops?
            del A_r_r[ind][s]['ON'] #,A_r_r['Rest of ON']
            del A_r_sr[ind][s]['ON'] #,A_r_sr['Rest of ON']
            del A_sr_r[ind][s]['ON']# ,A_sr_r['Rest of ON']  
            del A_sr_sr[ind][s]['ON']# ,A_sr_sr['Rest of ON']
            del Z_r_r[ind][s]['ON'] #,Z_r_r['Rest of ON']
            del Z_r_sr[ind][s]['ON']#,Z_r_sr['Rest of ON']
            del Z_sr_r[ind][s]['ON']# ,Z_sr_r['Rest of ON']  
            del Z_sr_sr[ind][s]['ON']# ,Z_sr_sr['Rest of ON']
            del x_r_r[ind][s]['ON'] #, x_r_r['Rest of ON']
            del x_sr[ind][s]['ON'] #, x_sr['Rest of ON']
            
            del f_rr2[ind][s]['ON']
            
            for j in A_r_r[ind][s]:            
                if j != 'Rest of ON':                                               # j = 'Rest of ON' is used for consistant formating for conseuqent codes
                    next
                elif j == 'Rest of ON':                                                      
                
                    f_rr[ind][s]['Rest of ON'] = x_r_r[ind][s][j]- (Z_r_r[ind][s][j] + Z_r_sr[ind][s][j])*i_hh  
                    f_rr[ind][s]['RoW'] =  x_sr[ind][s][j]- (Z_sr_r[ind][s][j] + Z_sr_sr[ind][s][j])*i_hh 
                    
                    x_rr[ind][s]['Rest of ON'] = x_r_r[ind][s][j]
                    x_rr[ind][s]['RoW'] = x_sr[ind][s][j]
                    
                    w_rr[ind][s]['Rest of ON'] = x_r_r[ind][s][j].T - i_hh.T*(Z_r_r[ind][s][j]+Z_sr_r[ind][s][j])
                    w_rr[ind][s]['RoW'] =x_sr[ind][s][j].T - i_hh.T*(Z_r_sr[ind][s][j]+Z_sr_sr[ind][s][j])
                    
                    
                    Z_row_col[ind][s]['Rest of ON'] = {}
                    Z_row_col[ind][s]['Rest of ON']['Rest of ON'] = Z_r_r[ind][s][j]
                    Z_row_col[ind][s]['Rest of ON']['RoW'] = Z_r_sr[ind][s][j]
                    Z_row_col[ind][s]['RoW'] = {}
                    Z_row_col[ind][s]['RoW']['Rest of ON'] = Z_sr_r[ind][s][j]
                    Z_row_col[ind][s]['RoW']['RoW'] = Z_sr_sr[ind][s][j]
                    
                    
                    A_row_col[ind][s]['Rest of ON'] = {}
                    A_row_col[ind][s]['Rest of ON']['Rest of ON'] = A_r_r[ind][s][j]
                    A_row_col[ind][s]['Rest of ON']['RoW'] = A_r_sr[ind][s][j]
                    A_row_col[ind][s]['RoW'] = {}
                    A_row_col[ind][s]['RoW']['Rest of ON'] = A_sr_r[ind][s][j]
                    A_row_col[ind][s]['RoW']['RoW'] = A_sr_sr[ind][s][j]      
                    
                    
                    I_row_col[ind][s]['Rest of ON'] = {}
                    I_row_col[ind][s]['Rest of ON']['Rest of ON'] = np.identity(len(industry_hr))
                    I_row_col[ind][s]['Rest of ON']['RoW'] = np.zeros_like(L[ind][s])
                    I_row_col[ind][s]['RoW'] = {}
                    I_row_col[ind][s]['RoW']['Rest of ON'] = np.zeros_like(L[ind][s])
                    I_row_col[ind][s]['RoW']['RoW'] = np.identity(len(industry_hr))   
                    
#                test[ind][s] = f_rr[ind][s]['Rest of ON'] + f_rr[ind][s]['RoW']  -f_scrubbed[ind][s]    # check if sum to provincial
#                test[ind][s] = w_rr[ind][s]['Rest of ON'] + w_rr[ind][s]['RoW'] - v_L_hh[ind][s]          # check if sum to provincial
  
            IRIO_concat_scenario(A_row_col[ind][s],I_row_col[ind][s],Z_row_col[ind][s],
                                 f_rr[ind][s], x_rr[ind][s], w_rr[ind][s],
                                 A_irio_temp[ind][s],I_irio_temp[ind][s],Z_irio_temp[ind][s])

            for row in A_r_r[ind][s]:
                gva_rr[ind][s][row] = v_prop_hh[ind][s]*np.diagflat(w_rr[ind][s][row])
            A_irio[ind][s] = pd.DataFrame(data = A_irio[ind][s],
                        index = pd.concat([industry_hr,industry_hr],axis=0),
                        columns = pd.concat([industry_hc,industry_hc],axis=0))
            x_i_irio[ind][s] = pd.DataFrame(data = x_r_r_irio[ind][s],
                    index = pd.concat([industry_hr,industry_hr],axis=0),
                    columns = ['Total Industry Output'])
            x_j_irio[ind][s] = pd.DataFrame(data = x_r_r_irio[ind][s].T,
                    index = ['Total Industry Input'],
                    columns = pd.concat([industry_hc,industry_hc],axis=0) )

            #""" 
            #need to split x_i_irio into x_rr_i
            #need to split x_j_irio into x_rr_j
            #need to split A_irio into A_row_col
            #"""
            IRIO_vector_split_scenario(x_rr_i[ind][s],x_i_irio[ind][s])         # check x = x_rr_i['Existing']['RoW'] +  x_rr_i['Existing']['Rest of ON']
            IRIO_vector_split_scenario(x_rr_j[ind][s],x_j_irio[ind][s].T)
            IRIO_matrix_split_scenario(A_row_col[ind][s],A_irio[ind][s])
#  
            for row in A_r_r[ind][s]:
                f_rr[ind][s][row] = pd.DataFrame(data = f_rr[ind][s][row],
                    index = industry_hr,
                    columns = ['Total Final Demand'])
                w_rr[ind][s][row] = pd.DataFrame(data = w_rr[ind][s][row],
                    index = ['Total Gross Value Added'],
                    columns = industry_hc)
                gva_aij[ind][s][row] = pd.DataFrame(
                        data = np.divide(w_rr[ind][s][row].values,x_rr_j[ind][s][row].T.values),
                       index = ['Gross Value Added Technical Coefficient'],
                       columns = industry_hc)    

                for col in A_r_r[ind][s]:
                    Z_row_col[ind][s][row][col] = pd.DataFrame(
                            data=Z_row_col[ind][s][row][col],
                             index = industry_hr,
                             columns = industry_hc)
            for row in A_r_r[ind][s]:
            #2nd row is for imports. value added for household only consists of imports. household labour is moved into transaction    
                gva_rr_aij[ind][s][row] = gva_rr[ind][s][row] * np.linalg.inv(np.diagflat(x_rr_j[ind][s][row].T.values))
                gva_rr_aij[ind][s][row] = pd.DataFrame(data = gva_rr_aij[ind][s][row], 
                      index = ['Other value added','Imports'],
                      columns = industry_hc)                

    
            x_r_r_irio_energy[ind][s] = np.diagflat(energy_int_irio)*x_r_r_irio[ind][s]                     # find energy use for x_irio
            x_r_r_irio_energy[ind][s] = pd.DataFrame(data = x_r_r_irio_energy[ind][s],
                                      index = pd.concat([industry_hc, industry_hc],axis=0),
                                      columns = ['Total Industry Energy Output (TJ)'])

