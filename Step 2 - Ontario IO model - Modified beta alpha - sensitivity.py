# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:55:12 2018

@author: jacky
"""
"""
THIS MODEL APPLIES METHOD 2 OF IMPORT SCRUBBING (EXPORT INCLUDED IN FINAL DEMAND)
THIS MODEL SCRUBS THE USE MATRIX
"""

"""
this code separates Existing case from AR cases

"""

""" 2019-02-13 revision: fixed v_prop in existing and ar situations
    for existing case, vprop's denominator changed to gva
    for scenario case, vprop's denominator changed to gva_ar
"""

""" 2019-02-18 revision: adjusted scenarios to include changing final demand """

""" 2019-03-18 revision : as per email from Stats Can,  a portion of the variable 
for grossed mixed income (income of the self-employed) would also be used as 
household final consumption expenditures. Gross mixed income is split into two parts,
one for self employment, and the remaining. portion for self employment is calculated
based on accounting difference of household final consumpetion expenditure and wages and salaries

wages and salaries includes travel expenditures by residents and non-residents  """

""" 2019-03-22  fix v_prop calculations """

""" 2019-04-10 do not allocate hh_scrubbed_diff to labour. hh_diff represents the imports
that goes to household expenditure. refer to Miller and Blair example of 
endogenousing household """

""" 2019-04-19 changed final demand scenarios to avoid confusion. 10,40,90 to 2.5,5,10
'Base' scenario to 'Basic' scenario """

""" 2019-05-09 
1.trying to fix regional final demand for AR_no_price_change scenario by manipulating r. 
    theres another way to get same results import vector should decrease by amount of delta Use that is imports.
    eg if import is 50/50 of Use, and Use decreases by 2, then imports decrease by 50% of 2.
2. price adjustments (final demand changse) are made at the model stage, not data stage."""

""" 2019-05-17 closing model at provincial level before AR changes """
                
""" 2019-05-23 mirroring method in CHris' toy model """

""" 2020-01-6 attempting to modify recipe change for adaptive reuse of non-residential buildings into residential 
    Material intensities from Marinova et al., (2019) and CMHC 2011 data base are used to estimate the modified recipe change """
    
import pandas as pd
import numpy as np
import copy

""" Data """
V = pd.read_excel('Supply.xlsx',index_col = 0)                                                # read copy-paste Supply matrix from excel and convert to dataframe
V = V.apply(pd.to_numeric, errors='coerce')                                     # convert the periods to nan
V = V.fillna(0)                                                                 # convert nan to zeroes
U = pd.read_excel('Use.xlsx',index_col = 0)
U = U.apply(pd.to_numeric, errors='coerce')                                     
U = U.fillna(0) 
gva = pd.read_excel('gva.xlsx', index_col = 0)
#gva= pd.read_excel('gva2.xlsx',index_col = 0)                                                 # 'taxes on production' and 'subsidies on products' removed to maintain balance with final demand (excluding taxes)
gva = gva.apply(pd.to_numeric, errors='coerce')                                 
gva = gva.fillna(0) 
imports = pd.read_excel('imports.xlsx',index_col = 0)
imports = imports.apply(pd.to_numeric, errors='coerce')
imports = imports.fillna(0) 
f_con_ex = pd.read_excel('f_con_ex.xlsx',index_col = 0)
f_con_ex = f_con_ex.apply(pd.to_numeric, errors='coerce')                                     
f_con_ex = f_con_ex.fillna(0) 
f_cap_form = pd.read_excel('f_cap_form.xlsx',index_col = 0)
f_cap_form = f_cap_form.apply(pd.to_numeric, errors='coerce')                                     
f_cap_form = f_cap_form.fillna(0) 
f_export = pd.read_excel('f_export.xlsx',index_col = 0)
f_export = f_export.apply(pd.to_numeric, errors='coerce')                                     
f_export = f_export.fillna(0) 
f_con_ex_tax = pd.read_excel('f_con_ex_tax.xlsx',index_col = 0)
f_con_ex_tax = f_con_ex.apply(pd.to_numeric, errors='coerce')                                     
f_con_ex_tax = f_con_ex.fillna(0) 
f_cap_form_tax = pd.read_excel('f_cap_form_tax.xlsx',index_col = 0)
f_cap_form_tax = f_cap_form.apply(pd.to_numeric, errors='coerce')                                     
f_cap_form_tax = f_cap_form.fillna(0) 
f_export_tax = pd.read_excel('f_export_tax.xlsx',index_col = 0)
f_export_tax = f_export.apply(pd.to_numeric, errors='coerce')                                     
f_export_tax = f_export.fillna(0) 
imports_tax = pd.read_excel('imports_tax.xlsx',index_col = 0)
imports_tax = imports.apply(pd.to_numeric, errors='coerce')
imports_tax = imports.fillna(0) 


industry=pd.Series(pd.read_csv('Input_SUIC_D.csv').iloc[:,0])
product=pd.Series(pd.read_csv('Input_SUPC_D.csv').iloc[:,0])

industry_hr = industry.copy()
industry_hc = industry.copy()
industry_hc[len(industry_hc)+1] = 'Household'
industry_hr[len(industry_hr)+1] = str('Household')

compensation = pd.read_excel('Total compensation for all jobs.xlsx',index_col=0)
compensation = compensation.apply(pd.to_numeric, errors='coerce')                                 
compensation = compensation.fillna(0) 
compensation = compensation.T


""" MODEL BUILDING """

""" NOTE: the Make Dataframe is NOT transposed"""
# manipulating final demand matrix for correct usage 
imports_neg = imports*-1                                                        # Use matrix already includes imports, Imports of final demand matrix must be negative entrance
#e = pd.concat([imports_neg,f_con_ex,f_cap_form,f_export],axis=1)
e = pd.concat([f_con_ex,f_cap_form,f_export],axis=1)
final_demand = pd.concat([f_con_ex, f_cap_form, f_export], axis=1)
# vector of ones, vector length of industry
i = np.ones((len(industry),1))
i_hh = np.ones((len(industry_hr),1))
i_product = np.ones((len(product),1))
# total industry output
x = V.T.fillna(0).values*np.matrix(np.ones((len(product),1)))                   # creating total industry output vector
x[x==0]=10**-10
q = V.fillna(0).values*np.matrix(np.ones((len(industry),1))) 
g = final_demand.fillna(0).values*\
            np.matrix(np.ones((len(f_con_ex.columns)+len(f_cap_form.columns)+\
                               len(f_export.columns),1)))

# portion of final demand for imports, imports in dimensions of (industry)x(import)
m = imports_neg.fillna(0).values*np.matrix(np.ones((len(imports.columns),1)))
# part of final demand that is export
exp = f_export.fillna(0).values*np.matrix(np.ones((len(f_export.columns),1)))

# total commodity output
q= V.fillna(0).values*np.matrix(i)
q[q==0] = 10**-10
# D Matrix - Commodity output proportion (market shares matrix), column sum = 1
D = V.T.fillna(0).values*np.matrix(np.linalg.inv(np.diag(q.flat)))
#test = D*q

""" portion of gross mixed income that is slef employment is found by
subtracting total compensation for all jobs by wages and salaries, and employment
social contribution"""

gva.loc['Self employment'] = compensation.loc['Total compensation for all jobs']\
 - gva.loc['Wages and salaries'] - gva.loc['Employers\' social contributions']
hh_col_sum = (f_con_ex.iloc[:,0:100].sum(axis=1)).sum(axis=0)
""" reallocate error proportionately """
hh_diff = hh_col_sum - sum(gva.loc['Wages and salaries'] + gva.loc['Self employment'])
gva.loc['error'] = hh_diff/sum(gva.loc['Wages and salaries'])*gva.loc['Wages and salaries']
gva.loc['Gross mixed income'] -= (gva.loc['Self employment']+gva.loc['error'])

# combine appropriately to make household labour row
gva.loc['Household labour'] = gva.loc['error'] +gva.loc['Wages and salaries'] + gva.loc['Self employment']
# combine remaining
#gva.loc['Other value added'] = gva.loc['Subsidies on production'] +\
#                                gva.loc['Taxes on production']+\
#                                gva.loc['Employers\' social contributions']+\
#                                gva.loc['Gross mixed income']+\
#                                gva.loc['Gross operating surplus']
#
gva.loc['Other value added'] = gva.loc['Subsidies on production'] +\
                                gva.loc['Taxes on production']+\
                                gva.loc['Employers\' social contributions']+\
                                gva.loc['Gross mixed income']+\
                                gva.loc['Gross operating surplus']+\
                                gva.loc['Taxes on products']+\
                                gva.loc['Subsidies on products']
# delete rows used for combination                               
#gva = gva.drop(['Wages and salaries','Self employment','error',
#         'Subsidies on production','Taxes on production','Employers\' social contributions',
#         'Gross mixed income','Gross operating surplus'])
gva = gva.drop(['Wages and salaries','Self employment','error',
         'Subsidies on production','Taxes on production','Employers\' social contributions',
         'Gross mixed income','Gross operating surplus', 'Taxes on products', 'Subsidies on products'])

hh_row_sum = sum(gva.loc['Household labour'])
i_gva = np.matrix(np.ones((1,len(gva)))) 
#test = hh_col_sum - hh_row_sum
#test =  i_gva*gva.values*i  - (g.sum(axis=0) + m.sum(axis=0))                   # accounting is off about 6 million in SUT tables >> accounting isnt off if i include taxes on products and subsides on products
#test = i_gva*gva.values*i                                                       # GDP

"""
Changing Use matrix recipe
"""
######### MARKET SHARE
beta = 0.2                                                               # percent of market that is adaptive reuse
######################

#### RECIPE CHANGE FOR CONCRETE STEEL WOOD
fkthis = 0.8
ar = ['Residential building construction',
       'Non-residential building construction',
       'Existing']

scenario = {#'Existing': (0,0),
            'Basic': (beta,0,0),
            '10%': (beta,0.1,0),
            '45%': (beta,0.45,0),
            '90%': (beta,0.9,0),
            'f2.5_basic': (beta,0,0.025),
            'f2.5_10': (beta,0.1,0.025),
            'f2.5_45': (beta,0.45,0.025),
            'f2.5_90': (beta,0.9,0.025),
            'f5_basic': (beta,0,0.05),
            'f5_10': (beta,0.1,0.05),
            'f5_45': (beta,0.45,0.05),
            'f5_90': (beta,0.9,0.05),                                                  #first number is beta, second is s
            'f10_basic': (beta,0,0.1),
            'f10_10': (beta,0.1,0.1),
            'f10_45': (beta,0.45,0.1),
            'f10_90': (beta,0.9,0.1)                                                   #first number is beta, second is s
            }                                                   #first number is beta, second is s

""" Other value added must increase by the value 
    equal to total value of recipe change  . if household la bour goes up, other value added must go down  """
alpha_gva = {'Household labour':0.2}

""" calculate value added coefficients """
gva_x = i_gva*gva.values * np.linalg.inv(np.diagflat(x))
""" update value added coefficient to match closed model dimensions """
# concatenate zero to the end. there is no value added for household, so concatenate zero
gva_x_hh = np.hstack([gva_x,np.zeros((1,1))])
#

# need to create dictionaries for every variable that is subject to scenarios
u = {}
r = {}
gva_ar = {}
U_ar = {}
U_scrubbed = {}
h = {}
f_scrubbed = {}
B = {}
A_ixi_scrub = {}
L = {}
x_ar = {}
z_scrubbed = {}
v_L = {}
v_import = {}
v_prop = {}
v_prop2= {}
h_con_ex = {}
h_cap_form = {}
h_export = {}
test = {}
test2 = {}
D_f_con_ex = {}
D_f_cap_form = {}
D_f_export = {}

D_f_con_ex_prop = {}
D_f_cap_form_prop = {}
D_f_export_prop = {}
D_f_con_ex_prop_no_hh = {}

hh_prop = {}
#hh_row_sum = {}
hh_scrubbed = {}
hh_lab = {}
hh_scrubbed_diff = {}
gva_scrubbed = copy.deepcopy(gva)
gva_ar_scrubbed = {}
U_delta = {}
m_ar = {}

x_j_ar = {}

x_i = {}
x_j = {}

v_prop_hh = {}

gva_ar_scrubbed_hh = {}
U_imports = {}
v_import2 = {}
gdp = {}
tot_import_change = {}
gva_increase = {}
v_coeff = {}
import_coeff = {}
v_scrubbed = {}
import_scrubbed = {}
v_L_hh = {}
v_added = {}
max_lab_increase = {}
gva_lab_increase = {}
gdp2 = {}
f_change = {}
hhc_change = {}
for ind in ar:
    if ind == 'Existing':
        u[ind] = U.fillna(0).values*np.matrix(i) 
        r[ind] = np.ones_like(q)                                                             # note that r will be negative
        r[ind] = np.divide(m, (u[ind]+g), out=np.zeros_like(r[ind]), 
                                                         where=(u[ind]+g)!=0)
        U_scrubbed[ind] = U.values + np.matrix(np.diag(r[ind].flat))*U.values
        h[ind] = g + np.diag(r[ind].flat)*g
        f_scrubbed[ind] = h[ind]
        B[ind] = U_scrubbed[ind]*np.matrix(np.linalg.inv(np.diag(x.flat)))           # need to replace nan with 0, or matrix multiplcation doesnt work
        A_ixi_scrub[ind] = D*B[ind]
        L[ind] = np.linalg.inv(np.identity(len(industry)) - (A_ixi_scrub)[ind])
        f_scrubbed[ind] = D*f_scrubbed[ind]
######## another way to calculate v_imports . this was is better. ##############
        U_imports[ind] = np.matrix(np.diag(r[ind].flat))*U.values               # note this term is used in calculting scrubbed Use matrix
        U_imports[ind] *= -1                                                    # recall r has negative values
        v_import[ind] = np.matrix(i_product.T)* U_imports[ind]
#############################################################################

######## calculate value added and import coefficients

        v_coeff[ind] = np.matrix(gva.values) * np.linalg.inv(np.diagflat(x))
        import_coeff[ind] = np.matrix(v_import[ind])* np.linalg.inv(np.diagflat(x))
######## calculate new x, and z, and value added and imports
        x_ar[ind] = L[ind]*f_scrubbed[ind]                                      # named x_ar for coding convenience in two-region model
        z_scrubbed[ind]= A_ixi_scrub[ind]*np.diag(x_ar[ind].flat)
        v_added[ind] = v_coeff[ind]*np.diagflat(x_ar[ind])
        v_import[ind] = import_coeff[ind] * np.diagflat(x_ar[ind])
        v_L[ind] = i_gva*v_added[ind] + v_import[ind]
#        test[ind] = x_ar[ind].T - i.T * z_scrubbed[ind] - v_L[ind]              # should equal 0

        gva_scrubbed = gva_scrubbed.append(pd.DataFrame(data = v_import[ind],index = ['Imports'],
                columns = industry))
        v_prop[ind] = gva_scrubbed.values*np.matrix(
                np.linalg.inv(np.diagflat(np.matrix(np.ones((1,len(gva_scrubbed))))*gva_scrubbed.values)))

        #### CLOSING WRT TO HOUSEHOLDS ####

       # finding household proportions
        D_f_con_ex[ind] = D*np.diag((1+r[ind]).flat)* np.matrix(f_con_ex)            # industry final demand for further manipulation, need to scrub                                             
        D_f_cap_form[ind] = D*np.diag((1+r[ind]).flat)* np.matrix(f_cap_form) 
        D_f_export[ind] = D* np.diag((1+r[ind]).flat) * np.matrix(f_export)                              # recall our import scrubbing nethod excludes exports
        # Calculate proportions of scrubbed ontario final demands
        
        D_f_con_ex_prop[ind] = np.matrix(
                np.linalg.inv(np.diagflat(f_scrubbed[ind]))) * D_f_con_ex[ind]
        D_f_cap_form_prop[ind] = np.matrix(
                np.linalg.inv(np.diagflat(f_scrubbed[ind]))) * D_f_cap_form[ind]
        D_f_export_prop[ind] = np.matrix(
                np.linalg.inv(np.diagflat(f_scrubbed[ind]))) * D_f_export[ind]

        D_f_con_ex_prop_no_hh[ind] = D_f_con_ex_prop[ind][:,100:]
        
        hh_prop[ind] =D_f_con_ex_prop[ind][:,0:100].sum(axis=1)  
        hh_scrubbed[ind] = np.multiply(hh_prop[ind],f_scrubbed[ind])
        hh_lab[ind] = np.matrix(gva_scrubbed.loc['Household labour'].values.T)



        test[ind] = gva_scrubbed.values.sum(axis=0).T - v_L[ind]               # should be zero
#        test[ind] = x_ar[ind] - x                                              # equals 0
#        test[ind] =np.multiply (x_ar[ind].T,i_gva * v_prop2[ind]) - v_L[ind]
#        test[ind] = v_L[ind]*i - i.T*f_scrubbed[ind]                           # should equal, they do equal

        
        # remove hh from final demand
        f_scrubbed[ind] -= hh_scrubbed[ind]
        # remove hh from value added 
        gva_scrubbed_hh = copy.deepcopy(gva_scrubbed)
        gva_scrubbed_hh = gva_scrubbed_hh.drop(['Household labour'])
#        v_prop_hh[ind] = gva_scrubbed_hh.values*np.linalg.inv(np.diagflat(v_L_hh[ind]))

        v_prop_hh[ind] = gva_scrubbed_hh.values*np.matrix(
                            np.linalg.inv(np.diagflat(np.matrix(np.ones((1,len(gva_scrubbed_hh))))*gva_scrubbed_hh.values)))        
#        test[ind] = i_gva*v_prop_hh[ind]                                       # must equal 1
     
######## difference between household row and household column is imports        
        if hh_lab[ind]*i > i.T*hh_scrubbed[ind]:
            hh_scrubbed_diff[ind] = hh_lab[ind]*i - i.T*hh_scrubbed[ind]
            
            v_import[ind] = np.hstack([v_import[ind],hh_scrubbed_diff[ind]])                
            
            x_ar[ind] = np.vstack([x_ar[ind],(i.T*hh_scrubbed[ind] + hh_scrubbed_diff[ind])])
    
            
            f_scrubbed[ind]= np.vstack([f_scrubbed[ind],[0]])
           
        elif  hh_scrubbed[ind]*i > i.T*hh_lab[ind]:
            hh_scrubbed_diff[ind] = hh_scrubbed[ind]*i - i.T*hh_lab[ind]
            f_scrubbed[ind] = np.append(f_scrubbed[ind],
                np.array(hh_scrubbed_diff[ind]), axis=0)
            
            x_ar[ind] = np.vstack([x_ar[ind],(hh_lab[ind]*i+ hh_scrubbed_diff[ind])])
            
            v_import[ind]= np.hstack([v_import[ind], np.zeros((1,1))])

        # concatenate 0 to the other value added
        v_added[ind] = np.hstack([v_added[ind],[[0],[0]]])
        hh_lab[ind] = np.hstack([hh_lab[ind],[[0]]])
        
        v_L_hh[ind] = v_import[ind] + i_gva*v_added[ind] - hh_lab[ind]                      # need to include import now 
        
        # update value added and import coefficients
        v_coeff[ind] = v_added[ind] * np.linalg.inv(np.diagflat(x_ar[ind])   )         # recall value added and household labour does not have value for hh consumption. so coefficients will be 0/x
        import_coeff[ind] = v_import[ind] *np.linalg.inv(np.diagflat(x_ar[ind]) )      

        # add hh labour and consumption into transaction matrix
        z_scrubbed[ind] = np.hstack([z_scrubbed[ind],hh_scrubbed[ind]])
        z_scrubbed[ind] = np.vstack([z_scrubbed[ind],hh_lab[ind]])
        
        # calculate new inputs and outputs
        
        x_i[ind] = z_scrubbed[ind] * i_hh + f_scrubbed[ind]
        x_j[ind] = i_hh.T*z_scrubbed[ind] + v_L_hh[ind]
#        test[ind] = x_i[ind] - x_j[ind].T                                      # must equal 0
        
        A_ixi_scrub[ind] = z_scrubbed[ind]*np.linalg.inv(np.diagflat(x_i[ind]))
        L[ind] =np.linalg.inv(np.identity(len(industry_hr)) - (A_ixi_scrub)[ind])
        x_ar[ind] = L[ind] * f_scrubbed[ind]
#        test[ind] = x_ar[ind] - x_i[ind]                                       # must equal 0
        z_scrubbed[ind] = A_ixi_scrub[ind] * np.diagflat(x_ar[ind])
        v_L_hh[ind] = x_ar[ind].T - i_hh.T*z_scrubbed[ind]
        
        v_added[ind] = v_coeff[ind]*np.diagflat(x_ar[ind])
        v_import[ind] = import_coeff[ind] * np.diagflat(x_ar[ind])
      
        gdp[ind] = i_gva*v_added[ind]*i_hh

#        gdp[ind] = gva_x_hh*x_ar[ind]   
#        test[ind] = gva_x_hh*x_ar[ind]   - i_gva*v_added[ind]*i_hh
        
      # for down allocating final demand by population
      # need to update ??
        h_con_ex[ind] = np.diag((1+r[ind]).flat)*\
                    np.matrix(f_con_ex)*\
                    np.matrix(np.ones((len(f_con_ex.columns),1)))
        h_con_ex[ind] = D*h_con_ex[ind]
        h_cap_form[ind] = np.diag((1+r[ind]).flat)*\
                    np.matrix(f_cap_form)*\
                    np.matrix(np.ones((len(f_cap_form.columns),1)))
        h_cap_form[ind] = D*h_cap_form[ind]
        h_export[ind] = np.diag((1+r[ind]).flat)*\
                    np.matrix(f_export)*\
                    np.matrix(np.ones((len(f_export.columns),1)))
        h_export[ind] = D*h_export[ind]

###### update v_prop
        v_prop_hh[ind] = np.hstack([v_prop_hh[ind],[[0],[1]]])                  # v_prop dropped households in Ontario code
        
#test = (gva_scrubbed.sum(axis=1)).sum(axis=0) - f_scrubbed['Existing'].sum(axis=0)
    elif ind != 'Existing':
        next

for ind in ar:
    if ind == 'Existing':
        next        
    elif ind != 'Existing':
        u[ind] = {}
        r[ind] = {}
        gva_ar[ind] = {}
        U_ar[ind] = {}
        U_scrubbed[ind] = {}
        h[ind] = {}
        f_scrubbed[ind] = {}
        B[ind] = {}
        A_ixi_scrub[ind]= {}
        L[ind] = {}
        x_ar[ind] = {}
        z_scrubbed[ind] = {}
        v_L[ind] = {}
        v_prop[ind] = {}
        h_con_ex[ind] = {}
        h_cap_form[ind] = {}
        h_export[ind] = {}
        test2[ind] = {}
        v_import[ind] = {}
        D_f_con_ex[ind] = {}
        D_f_cap_form[ind] = {}
        D_f_export[ind] = {}
        
        D_f_con_ex_prop[ind] = {}
        D_f_cap_form_prop[ind] = {}
        D_f_export_prop[ind] = {}
        D_f_con_ex_prop_no_hh[ind] = {}
        
        hh_prop[ind] = {}
        hh_scrubbed[ind] = {}
        hh_scrubbed_diff[ind] = {}
#        hh_row_sum[ind] = {}
        hh_lab[ind] = {}

        gva_ar_scrubbed[ind] = {}
        U_delta[ind] = {}
        m_ar[ind] = {}
        
        x_j_ar[ind] = {}
        
        x_i[ind] ={}
        x_j[ind] = {}
        
        gva_ar_scrubbed_hh[ind] = {}
        v_prop_hh[ind] = {}
        
        U_imports[ind] = {}
        v_import2[ind] = {}
        gdp[ind] = {}
        gva_increase[ind] = {}
        tot_import_change[ind] = {}
        v_coeff[ind] = {}
        import_coeff[ind] = {}
        v_scrubbed[ind] = {}
        import_scrubbed[ind] = {}
        v_L_hh[ind] = {}
        v_added[ind] = {}
        max_lab_increase[ind] = {}
        gva_lab_increase[ind] = {}
        gdp2[ind] = {}
        f_change[ind] = {}
        hhc_change[ind] = {}
        for s in scenario:
            U_ar[ind][s] = copy.deepcopy(U)
            gva_ar[ind][s] = copy.deepcopy(gva)


            # need to update: Started Jan-6-2020
            alpha_U = {'Cement' :fkthis,
                           'Ready-mixed concrete':fkthis,
                           'Concrete products' : fkthis,
                           'Prefabricated metal buildings and components':fkthis,
                           'Fabricated steel plates and other fabricated structural metal':fkthis,
                           'Other architectural metal products':fkthis,
                           'Truck transportation services for general freight':0.2,
                           'Truck transportation services for specialized freight':0.2,
            
                           'Textile products, n.e.c.':scenario[s][1],
                           'Hardwood lumber':scenario[s][1],
                           'Softwood lumber':scenario[s][1],
                           'Other sawmill products and treated wood products':scenario[s][1],
                           'Veneer and plywood':scenario[s][1],
                           'Wood trusses and engineered wood members':scenario[s][1],
                           'Reconstituted wood products':scenario[s][1],
                           'Wood windows and doors':scenario[s][1],
                           'Prefabricated wood and manufactured (mobile) buildings and components':scenario[s][1],
                           'Wood products, n.e.c.':scenario[s][1],
                           'Plastic and foam building and construction materials':scenario[s][1],
                           'Plastic products, n.e.c.':scenario[s][1],
                           'Rubber products, n.e.c.':scenario[s][1],
                           'Clay and ceramic products and refractories':scenario[s][1],
                           'Glass (including automotive), glass products and glass containers':scenario[s][1],
                           'Lime and gypsum products':scenario[s][1],
                           'Non-metallic mineral products, n.e.c.':scenario[s][1],
                           'Iron and steel basic shapes and ferro-alloy products':scenario[s][1],
                           'Iron and steel pipes and tubes (except castings)':scenario[s][1],
                           'Wire and other rolled and drawn steel products':scenario[s][1],
                           'Forged and stamped metal products':scenario[s][1],
                           'Metal windows and doors':scenario[s][1],
                           'Boilers, tanks and heavy gauge metal containers':scenario[s][1],
                           'Springs and wire products':scenario[s][1],
                           'Threaded metal fasteners and other turned metal products including automotive':scenario[s][1],
                           'Metal valves and pipe fittings':scenario[s][1],
                           'Fabricated metal products, n.e.c.':scenario[s][1],
                           'Heating and cooling equipment (except household refrigerators and freezers)':scenario[s][1],
                           'Other electronic components':scenario[s][1],
                           'Electric light bulbs and tubes':scenario[s][1],
                           'Lighting fixtures':scenario[s][1],
                           'Small electric appliances':scenario[s][1],
                           'Major appliances':scenario[s][1],
                           'Switchgear, switchboards, relays and industrial control apparatus':scenario[s][1],
                           'Wood kitchen cabinets and counter tops':scenario[s][1],
                           'Wholesale margins - building materials and supplies':scenario[s][1],
                           'Retail margins - furniture and home furnishings':scenario[s][1],
                           'Retail margins - building materials, garden equipment and supplies':scenario[s][1],
                           'Architectural, engineering and related services': -0.2}
            # values hard calculated in excel. individual recipes calculated then summed to get equivalent
            alpha_U_res = {'Cement' :fkthis,
                           'Ready-mixed concrete':fkthis,
                           'Concrete products' : fkthis,
                           'Prefabricated metal buildings and components':fkthis,
                           'Fabricated steel plates and other fabricated structural metal':fkthis,
                           'Other architectural metal products':fkthis,
                           'Truck transportation services for general freight':0.2,
                           'Truck transportation services for specialized freight':0.2,

                           'Hardwood lumber':fkthis,
                           'Softwood lumber':fkthis,
                           'Other sawmill products and treated wood products':fkthis,
                           'Veneer and plywood':fkthis,
                           'Wood trusses and engineered wood members':fkthis,
                           'Reconstituted wood products':fkthis,
                           'Wood windows and doors':fkthis,
                           'Prefabricated wood and manufactured (mobile) buildings and components':fkthis,
                           'Wood products, n.e.c.':fkthis,


            
                           'Textile products, n.e.c.':scenario[s][1],
                           'Plastic and foam building and construction materials':scenario[s][1],
                           'Plastic products, n.e.c.':scenario[s][1],
                           'Rubber products, n.e.c.':scenario[s][1],
                           'Clay and ceramic products and refractories':scenario[s][1],
                           'Glass (including automotive), glass products and glass containers':scenario[s][1],
                           'Lime and gypsum products':scenario[s][1],
                           'Non-metallic mineral products, n.e.c.':scenario[s][1],
                           'Iron and steel basic shapes and ferro-alloy products':scenario[s][1],
                           'Iron and steel pipes and tubes (except castings)':scenario[s][1],
                           'Wire and other rolled and drawn steel products':scenario[s][1],
                           'Forged and stamped metal products':scenario[s][1],
                           'Metal windows and doors':scenario[s][1],
                           'Boilers, tanks and heavy gauge metal containers':scenario[s][1],
                           'Springs and wire products':scenario[s][1],
                           'Threaded metal fasteners and other turned metal products including automotive':scenario[s][1],
                           'Metal valves and pipe fittings':scenario[s][1],
                           'Fabricated metal products, n.e.c.':scenario[s][1],
                           'Heating and cooling equipment (except household refrigerators and freezers)':scenario[s][1],
                           'Other electronic components':scenario[s][1],
                           'Electric light bulbs and tubes':scenario[s][1],
                           'Lighting fixtures':scenario[s][1],
                           'Small electric appliances':scenario[s][1],
                           'Major appliances':scenario[s][1],
                           'Switchgear, switchboards, relays and industrial control apparatus':scenario[s][1],
                           'Wood kitchen cabinets and counter tops':scenario[s][1],
                           'Wholesale margins - building materials and supplies':scenario[s][1],
                           'Retail margins - furniture and home furnishings':scenario[s][1],
                           'Retail margins - building materials, garden equipment and supplies':scenario[s][1],
                           'Architectural, engineering and related services': -0.2}        
        
            for ingredient in alpha_U:
                if ind == 'Non-residential building construction':
                    U_ar[ind][s].loc[ingredient,ind] *=\
                                             (1-scenario[s][0]*alpha_U[ingredient])
                elif ind == 'Residential building construction':            
                    U_ar[ind][s].loc[ingredient,ind] *=\
                                             (1-scenario[s][0]*alpha_U_res[ingredient])                                           
            #""" import scrub """ 
            # row sum of U matrix 
            u[ind][s] = U_ar[ind][s].fillna(0).values*np.matrix(i) 
            
            # scaling factor, assume that exports are not made up of imports. this method is different from Miller & Blair (2009)
            r[ind][s] = np.ones_like(q)                                         # note that r will be negative
            U_delta[ind][s] = U - U_ar[ind][s]
#            test2[ind][s] = np.matrix(np.ones((1,len(product))))*U_delta[ind][s].values
            m_ar[ind][s] = m - np.multiply(U_delta[ind][s].values*np.matrix(i) , r['Existing'])
            r[ind][s] = np.divide(m_ar[ind][s], (u[ind][s]+g), 
                         out=np.zeros_like(r[ind][s]), where=(u[ind][s]+g)!=0)  # divide m by (u+g) where (u+g) not equal zero
#            test2[ind][s] = r[ind][s] - r['Existing']                           # if removed imports properly from table , r should equal
            h[ind][s] = g + np.diag(r[ind][s].flat)*g
            # scrub Use Matrix
            U_scrubbed[ind][s] = U_ar[ind][s].values +\
                        np.matrix(np.diag(r[ind][s].flat))*U_ar[ind][s].values
                
            f_scrubbed[ind][s] = h[ind][s] #+ exp
############ add total reduction in commodity value to value added.  #######            

############ if household labour goes up, other value added has to go down ####################
            

            for ingredient in alpha_gva:
                max_lab_increase[ind][s] = gva.loc['Other value added',ind]/gva.loc[ingredient,ind]
                print 'max increase to household labour is: ' +str( 100*max_lab_increase[ind][s] ) +'%'

                if scenario[s][0]*alpha_gva[ingredient] < max_lab_increase[ind][s]:
                    gva_ar[ind][s].loc[ingredient,ind] *=\
                                           (1+scenario[s][0]*alpha_gva[ingredient])
                elif scenario[s][0]*alpha_gva[ingredient] > max_lab_increase[ind][s]:
                    next
                                     
            # check if household labour increase is less than other value added.
                if gva_ar[ind][s].loc[ingredient,ind] -gva.loc[ingredient,ind] >= gva.loc['Other value added',ind]:
                    print 'Increases to household labour is NOT okay'
                elif gva_ar[ind][s].loc[ingredient,ind] -gva.loc[ingredient,ind] < gva.loc['Other value added',ind]:
                    print 'Increases to Household labour is okay '
####################################################################################
            
            gva_lab_increase[ind][s] = gva_ar[ind][s].loc['Household labour',:].values - gva.loc['Household labour',:].values
            gva_lab_increase[ind][s] = gva_lab_increase[ind][s].reshape((1,len(industry)))
            
            gva_increase[ind][s] = np.matrix(i_product.T)*U_delta[ind][s].values
            
            # savings in recipe goes to other value added - Chris 
            gva_ar[ind][s].loc['Other value added'] = gva_ar[ind][s].loc['Other value added'].values + gva_increase[ind][s] - gva_lab_increase[ind][s]
            # if increase household labour, other value added has to decrease
            

            # calculate new input
            x_j_ar[ind][s] = (np.matrix(np.ones((1,len(product))))*U_ar[ind][s].values) + (i_gva*np.matrix(gva_ar[ind][s].values)) 
           
#            test2[ind][s] = x_j_ar[ind][s] - x.T                               #should equal. if not, check that delta value added = delta recipe change, or check taxes
############ another way to calculate v_imports . this was is better. ##############
            U_imports[ind][s] = np.matrix(np.diag(r[ind][s].flat))*U_ar[ind][s].values               # note this term is used in calculting scrubbed Use matrix
            U_imports[ind][s] *= -1                                                    # recall r has negative values
            v_import[ind][s] = np.matrix(i_product.T)* U_imports[ind][s]
#            v_import[ind][s] = pd.DataFrame(data = v_import2[ind][s],index = ['Imports'],
#                    columns = industry)
#############################################################################
        
            # Calculate new value added and import coefficients
            v_coeff[ind][s] = np.matrix(gva_ar[ind][s].values) * np.linalg.inv(np.diagflat(x_j_ar[ind][s]))
            import_coeff[ind][s] = np.matrix(v_import[ind][s])* np.linalg.inv(np.diagflat(x_j_ar[ind][s]))
            # B matrix - parallel to ordinary technical coefficient aij
############ if x_j_ar = x, use either.
#            B[ind][s] = U_scrubbed[ind][s]*np.matrix(np.linalg.inv(np.diag(x.flat)))           # need to replace nan with 0, or matrix multiplcation doesnt work
            B[ind][s] = U_scrubbed[ind][s]*np.matrix(np.linalg.inv(np.diag(x_j_ar[ind][s].flat)))           # need to replace nan with 0, or matrix multiplcation doesnt work
#            test2[ind][s] = i_product.T*B[ind][s] + i_gva*v_coeff[ind][s] + import_coeff[ind][s] # must equal 1
            
        #""" leontief Matrices based on industry techonology: industry-by-industry"""
            # industry techonology : industry-by-industry
            A_ixi_scrub[ind][s] = D*B[ind][s]
            # Leontief Matrix (industry-by-industry total requirement matrix under industry technology)
            L[ind][s] = np.linalg.inv(np.identity(len(industry)) -\
                                                         (A_ixi_scrub)[ind][s])
            # leontief f vector from Make and Use final demand
            f_scrubbed[ind][s] = D*f_scrubbed[ind][s]#
            
            #""" need to calculate new x """
               
            x_ar[ind][s] = L[ind][s]*f_scrubbed[ind][s]
#            test2[ind][s] =  x_ar[ind][s] - x_j_ar[ind][s]                    # will be different now.
############ calculate new transactions, imports and value added    
            v_added[ind][s] =  v_coeff[ind][s] * np.diagflat(x_ar[ind][s])    # this is value added not leontief valeu added ( Leontief value added includes imports)
            v_import[ind][s] = import_coeff[ind][s]*np.diagflat(x_ar[ind][s])
            z_scrubbed[ind][s]= A_ixi_scrub[ind][s]*np.diag(x_ar[ind][s].flat) #z_scrubbed is actually the new model transaction matrix
            v_L[ind][s] = v_import[ind][s] + i_gva*v_added[ind][s]
##############################################################################
#            test2[ind][s] = ( x_ar[ind][s].T - i.T*z_scrubbed[ind][s] )- v_L[ind][s] # must equal 0

############# check if gdp change = import change
#            gdp[ind][s] = i_gva*v_added[ind][s]*i 
#            tot_import_change[ind][s] = np.matrix(v_import[ind][s]- v_import['Existing'][:,:-1])*i
#            test2[ind][s] = gdp[ind][s] - gdp['Existing'] + tot_import_change[ind][s] # should equal 0 for non price change scenarios



########### DataFrames
            gva_ar_scrubbed[ind][s] = pd.DataFrame(data = v_added[ind][s],index = ['Household labour','Other value added'],columns = industry)
            gva_ar_scrubbed[ind][s] = gva_ar_scrubbed[ind][s].append(pd.DataFrame(data = v_import[ind][s],index = ['Imports'],
                    columns = industry))

            v_prop[ind][s] = gva_ar_scrubbed[ind][s].values*np.matrix(
                    np.linalg.inv(np.diagflat(np.matrix(np.ones((1,len(gva_ar_scrubbed[ind][s]))))*gva_ar_scrubbed[ind][s].values)))
  

            #### CLOSING MODEL WRT HOUSEHOLDS ####

          # finding household proportions
            D_f_con_ex[ind][s] = D*np.diag((1+r[ind][s]).flat)* np.matrix(f_con_ex)            # industry final demand for further manipulation, need to scrub                                             
            D_f_cap_form[ind][s] = D*np.diag((1+r[ind][s]).flat)* np.matrix(f_cap_form) 
            D_f_export[ind][s] = D* np.diag((1+r[ind][s]).flat) * np.matrix(f_export)                              # recall our import scrubbing nethod excludes exports
            # Calculate proportions of scrubbed ontario final demands
            
            D_f_con_ex_prop[ind][s] = np.matrix(
                    np.linalg.inv(np.diagflat(f_scrubbed[ind][s]))) * D_f_con_ex[ind][s]
            D_f_cap_form_prop[ind][s] = np.matrix(
                    np.linalg.inv(np.diagflat(f_scrubbed[ind][s]))) * D_f_cap_form[ind][s]
            D_f_export_prop[ind][s] = np.matrix(
                    np.linalg.inv(np.diagflat(f_scrubbed[ind][s]))) * D_f_export[ind][s]
#            hh_row_sum[ind][s] = sum(gva_ar[ind][s].loc['Household labour'])
            
            hh_prop[ind][s] =D_f_con_ex_prop[ind][s][:,0:100].sum(axis=1)  
            D_f_con_ex_prop_no_hh[ind][s] = D_f_con_ex_prop[ind][s][:,100:]
            
            hh_scrubbed[ind][s] = np.multiply(hh_prop[ind][s],f_scrubbed[ind][s])
            hh_lab[ind][s] = np.matrix(gva_ar_scrubbed[ind][s].loc['Household labour'].values.T)

            # remove hh from final demand
            f_scrubbed[ind][s] -= hh_scrubbed[ind][s]
            # remove hh from value added 
            gva_ar_scrubbed_hh[ind][s] = copy.deepcopy(gva_ar_scrubbed[ind][s])
            gva_ar_scrubbed_hh[ind][s] = gva_ar_scrubbed_hh[ind][s].drop(['Household labour'])
            v_prop_hh[ind][s] = gva_ar_scrubbed_hh[ind][s].values*np.matrix(
                    np.linalg.inv(np.diagflat(np.matrix(np.ones((1,len(gva_ar_scrubbed_hh[ind][s]))))*gva_ar_scrubbed_hh[ind][s].values)))
            #test2[ind][s] = i_gva*v_prop_hh[ind][s]                            must equal 1
            if hh_lab[ind][s]*i > i.T*hh_scrubbed[ind][s]:
                hh_scrubbed_diff[ind][s] = hh_lab[ind][s]*i - i.T*hh_scrubbed[ind][s]

                v_import[ind][s] = np.hstack([v_import[ind][s],hh_scrubbed_diff[ind][s]])                
                
                x_ar[ind][s] = np.vstack([x_ar[ind][s],(i.T*hh_scrubbed[ind][s] + hh_scrubbed_diff[ind][s])])

                f_scrubbed[ind][s]= np.vstack([f_scrubbed[ind][s],[0]])
                             
            elif  hh_scrubbed[ind][s]*i > i.T*hh_lab[ind][s]:
                hh_scrubbed_diff[ind][s] = hh_scrubbed[ind][s]*i - i.T*hh_lab[ind][s]
                f_scrubbed[ind][s] = np.append(f_scrubbed[ind][s],
                    np.array(hh_scrubbed_diff[ind][s]), axis=1)
                
                x_ar[ind][s] = np.vstack([x_ar[ind][s],(hh_lab[ind][s]*i+ hh_scrubbed_diff[ind][s])])
                
                v_import[ind][s] = np.hstack([v_import[ind][s], np.zeros((1,1))])
                               
            # concatenate 0 to the other value added
            v_added[ind][s] = np.hstack([v_added[ind][s],[[0],[0]]])
            hh_lab[ind][s] = np.hstack([hh_lab[ind][s],[[0]]])
            
            v_L_hh[ind][s] = v_import[ind][s] + i_gva*v_added[ind][s] - hh_lab[ind][s]                        # need to include import now 
            
            # update value added and import coefficients
            v_coeff[ind][s] = v_added[ind][s] * np.linalg.inv(np.diagflat(x_ar[ind][s])   )         # recall value added and household labour does not have value for hh consumption. so coefficients will be 0/x
            import_coeff[ind][s] = v_import[ind][s] *np.linalg.inv(np.diagflat(x_ar[ind][s]) )      
            
            # add hh labour and consumption into transaction matrix

            z_scrubbed[ind][s] = np.hstack([z_scrubbed[ind][s],hh_scrubbed[ind][s]])
            z_scrubbed[ind][s] = np.vstack([z_scrubbed[ind][s],hh_lab[ind][s]])            
            # calculate new inputs and outputs
            
            x_i[ind][s] = z_scrubbed[ind][s] * i_hh + f_scrubbed[ind][s]
            x_j[ind][s] = i_hh.T*z_scrubbed[ind][s] + v_L_hh[ind][s]
#            test2[ind][s] = x_i[ind][s] - x_j[ind][s].T                         # must equal 0
#            test2[ind][s] = x_i[ind][s] - x_ar[ind][s]                          # must equal
         
            A_ixi_scrub[ind][s] = z_scrubbed[ind][s]*np.linalg.inv(np.diagflat(x_i[ind][s]))
            L[ind][s] =np.linalg.inv(np.identity(len(industry_hr)) - (A_ixi_scrub)[ind][s])
#            test2[ind][s] = L[ind][s] * f_scrubbed[ind][s] - x_i[ind][s]        # must equal 0
            
############# applying AR changes to final demand at model ############
            alpha_f =  {'Residential building construction':scenario[s][2],
                               'Non-residential building construction':scenario[s][2]}

            # temporarily make f_scrubbed a dataframe        
            f_scrubbed[ind][s] = pd.DataFrame(data=  f_scrubbed[ind][s], index = industry_hr)
            f_scrubbed[ind][s].loc[ind,:] *=(1+beta*alpha_f[ind])
            # return to array
            f_scrubbed[ind][s] = f_scrubbed[ind][s].values            
            f_change[ind][s] = i_hh.T*(f_scrubbed[ind][s] - f_scrubbed['Existing']) # total change in final demand
############################################################################
#            test2[ind][s] = f_scrubbed[ind][s] - f_scrubbed['Existing']    
            
            # calculate new x_ar
            x_ar[ind][s] = L[ind][s]*f_scrubbed[ind][s]
#            test2[ind][s] = x_ar['Existing'] - x_ar[ind][s]
 
############ calculate new transactions, imports and value added    
            
            v_added[ind][s] =  v_coeff[ind][s] * np.diagflat(x_ar[ind][s])    # this is value added not leontief valeu added ( Leontief value added includes imports)
            v_import[ind][s] = import_coeff[ind][s]*np.diagflat(x_ar[ind][s])
            z_scrubbed[ind][s]= A_ixi_scrub[ind][s]*np.diag(x_ar[ind][s].flat) #z_scrubbed is actually the new model transaction matrix
            v_L_hh[ind][s] = x_ar[ind][s].T - i_hh.T*z_scrubbed[ind][s]
            test2[ind][s] = v_added[ind][s][1,:] + v_import[ind][s] - v_L_hh[ind][s]
############ check if gdp change = import change + final demand change + household change
            hhc_change[ind][s] = z_scrubbed[ind][s][:-1,-1] - hh_scrubbed[ind][s]
            hhc_change[ind][s] = i.T*hhc_change[ind][s]
            gdp[ind][s] = i_gva*v_added[ind][s]*i_hh 
            
            # NOTE DO NOT INCLUDE THE HH COLUMN WHEN CALCULATING V_IMPORT CHANGES 
            tot_import_change[ind][s] = np.matrix(v_import[ind][s][:,:-1] - v_import['Existing'][:,:-1])*i
            test2[ind][s] = gdp[ind][s] - (gdp['Existing'] - tot_import_change[ind][s] + f_change[ind][s]+hhc_change[ind][s]) # should equal 0 
#            test2[ind][s] = gdp[ind][s] -gdp['Existing']
           
            #""" split final demand for RoW model """
            # are these even used?
            h_con_ex[ind][s] = np.diag((1+r[ind][s]).flat)*\
                        np.matrix(f_con_ex)*\
                        np.matrix(np.ones((len(f_con_ex.columns),1)))
            h_con_ex[ind][s] = D*h_con_ex[ind][s]
            h_cap_form[ind][s] = np.diag((1+r[ind][s]).flat)*\
                        np.matrix(f_cap_form)*\
                        np.matrix(np.ones((len(f_cap_form.columns),1)))
            h_cap_form[ind][s] = D*h_cap_form[ind][s]
            h_export[ind][s] = np.diag((1+r[ind][s]).flat)*\
                        np.matrix(f_export)*\
                        np.matrix(np.ones((len(f_export.columns),1)))
            h_export[ind][s] = D*h_export[ind][s]
            #h_total = h_con_ex + h_cap_form  + h_export
            #test = f_scrubbed-h_total                                                             # should equal zero

###### update v_prop
            v_prop_hh[ind][s] = np.hstack([v_prop_hh[ind][s],[[0],[1]]])                  # v_prop dropped households in Ontario code

""" ENERGY INTENSITY """
## censored industry assumed to have 0 TJ /$ output
#energy_int = pd.read_csv('Energy_intensities_ordered.csv').set_index('SUIC') 

# estimated by proportionality
energy_int = pd.read_csv('Energy_intensities_ordered - estimated censored industries.csv').set_index('SUIC') 

