# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:06:29 20184
@author: jacky
"""
import pandas as pd

""" importing data and calculating proportions"""
""" 2019-05-17 include household labour 'employment'. employment is assumed to equal population """

""" INPUTS """ 
employ = pd.read_csv('Employment_2014_hh.csv').set_index('NAICS')                  # employment x 1000
agg = pd.read_csv('Aggregation_detailed_hh.csv')
agg_emp = pd.read_csv('Aggregation_detailed_hh.csv')
emp_porp = pd.DataFrame(index = employ.index, columns= list(employ))
pop_2014 = pd.read_csv('cansim-0510056.csv', index_col= 0)                      # read csv and set first column as index
industry=pd.Series(pd.read_csv('Input_SUIC_D_hh.csv').iloc[:,0])
# drop columns for Two Region model
col_drop = list(employ)
col_drop.remove('RoW')
col_drop.remove('ON')
employ.drop(col_drop,axis=1,inplace=True)
emp_porp.drop(col_drop,axis=1,inplace=True)
pop_2014.drop(col_drop,axis=0,inplace=True)

""" Employment proportions """
#filling hidden NAICS
for j in list(employ):
    emp_porp[j] = employ[j]/employ['ON'] 
    employ[j]=employ[j].replace(0,employ.loc[employ[j]==0]['ON']*
                                              emp_porp[employ[j]>0][j].mean())
    emp_porp[j] = employ[j]/employ['ON']
    
employ['Rest of ON'] = 2*employ['ON'] - employ.sum(axis=1)                      # calculating rest of ON, accounts for double counting 'ON'
emp_porp['Rest of ON'] = employ['Rest of ON']/employ['ON']

pop_2014.loc['Rest of ON','Population'] = 2*pop_2014.loc['ON','Population'] - \
                                              sum(pop_2014.loc[:,'Population'])
pop_2014.loc['Rest of ON','Proportion'] = pop_2014.loc['Rest of ON','Population']\
                                          /pop_2014.loc['ON','Population'] 
#%%                                          
# aggregating data based on NAICS industry        
for j in list(employ):
    agg[j] = agg['NAICS'].map(emp_porp[j])
    agg_emp[j] = agg['NAICS'].map(employ[j])
agg_emp = agg_emp.set_index(keys = 'SUIC', drop = True)                         # move SUIC column to index for the nested forloops 

""" calculating CILQ, following similar format as 2018-04-30 ROW model.py"""
""" agg_CILQ_rest was calculated because older code used 'two-region logic
    with more than two regions"""

agg_CILQ = {}
agg_CILQ_rest = {}
for j in list(employ):
    agg_CILQ[j] = pd.DataFrame(index = industry, columns = industry)
    agg_CILQ_rest[j] = pd.DataFrame(index = industry, columns = industry)
    for row in industry:
        for col in industry:
            if j == 'ON':                                                       # passing ON to avoid dvision by zero
                next
            elif j != 'ON':    
                if row != col:   
                    agg_CILQ[j].loc[row,col]=(
                            agg_emp[j][row]/agg_emp['ON'][row]
                                             )/\
                            (agg_emp[j][col]/agg_emp['ON'][col])
                    agg_CILQ_rest[j].loc[row,col]=\
                    ((agg_emp['ON'][row]-agg_emp.loc[row,j])/agg_emp['ON'][row])/ \
                    ((agg_emp['ON'][col]-agg_emp.loc[col,j])/agg_emp['ON'][col])
                       
                elif row == col:
                    agg_CILQ[j].loc[row,col] = (
                            agg_emp[j][row]/agg_emp[j].sum()
                                               )/(
                            agg_emp['ON'][row]/agg_emp['ON'].sum()
                                                 )
                    
                    agg_CILQ_rest[j].loc[row,col]=(
                                        (agg_emp['ON'][row]-agg_emp[j][row])\
                                        /(agg_emp['ON'].sum()-agg_emp[j].sum())
                                                  )\
                                        /(agg_emp['ON'][row]/agg_emp['ON'].sum())
    agg_CILQ[j][agg_CILQ[j]>=1] =1                                              # filtering values greater than 1 
    agg_CILQ[j] = agg_CILQ[j].fillna(0).values
    agg_CILQ_rest[j][agg_CILQ_rest[j]>=1] =1                                    # filtering values greater than 1 
    agg_CILQ_rest[j] = agg_CILQ_rest[j].fillna(0).values