import sys
import os
import argparse
import datetime
from time import time
from string import Template
import numpy as np
import pandas as pd

def get_data():
    
    """Function to load the dataset"""
    
    df_to_treat = pd.read_csv('../data/test.csv')
    
    print(f'Shape Main dataset:{df_to_treat.shape}')
    
    return df_to_treat


def feats_transform_type(df_to_treat):
    
    "Preprocessing function to treat the datatypes of the inference dataset"
    
    """Args: 
            data: dataset pandas
       Return:
            inference dataset pandas with datatypes treated
       Goal:
            Use this function the transform the datatypes of the inference dataset
    """
    
    df_to_treat = df_to_treat.copy()
    
    feats_int = ['Age','Num_of_Loan','Num_of_Delayed_Payment']
    feats_float = ['Annual_Income','Changed_Credit_Limit','Num_Credit_Inquiries','Outstanding_Debt','Amount_invested_monthly','Monthly_Balance']
    
    for feat in feats_int:
        df_to_treat[feat] = df_to_treat[feat].astype(str)  
        df_to_treat[feat] = df_to_treat[feat].str.replace('_', '', regex=False)  
        df_to_treat[feat] = df_to_treat[feat].replace(['', 'nan'], pd.NA)  
        df_to_treat[feat] = df_to_treat[feat].fillna(-1)  
        df_to_treat[feat] = df_to_treat[feat].astype(int)  
    
    for feat in feats_float:
        df_to_treat[feat] = df_to_treat[feat].astype(str)  
        df_to_treat[feat] = df_to_treat[feat].str.replace('_', '', regex=False)  
        df_to_treat[feat] = df_to_treat[feat].replace(['', 'nan'], pd.NA)  
        df_to_treat[feat] = df_to_treat[feat].fillna(-1.0)  
        df_to_treat[feat] = df_to_treat[feat].astype(float)  
    
    return df_to_treat

def preprocessing_data (df_to_treat):
    
    "Preprocessing function to treat the inference dataset"
    
    """Args: 
            data: dataset pandas
       Return:
            inference dataset pandas treated for the model
       Goal:
            Use this function the transform the inference dataset into a proper dataset for the model
    """
    
    df_to_treat = df_to_treat.copy()
    
    # Treating the Credit_History_Year feature
    df_to_treat['Credit_History_Year'] = df_to_treat['Credit_History_Age'].str.extract(r'(\d+) Years')
    df_to_treat['Credit_History_Year'] = df_to_treat['Credit_History_Year'].fillna(-1)
    df_to_treat['Credit_History_Year'] = df_to_treat['Credit_History_Year'].astype(int) 
    del df_to_treat['Credit_History_Age']
    
    # Treating the Age feature
    df_to_treat['Age'] = df_to_treat['Age'].apply(lambda x: -1 if x < 18 or x > 120 else x)
    
    # Treating the feature names
    df_to_treat.columns = [col.lower() for col in df_to_treat.columns]
    
    # Treating the weird feature values
    df_to_treat['occupation'] = df_to_treat['occupation'].replace('_______', np.nan)
    df_to_treat['credit_mix'] = df_to_treat['credit_mix'].replace('_', np.nan)
    df_to_treat['payment_behaviour'] = df_to_treat['payment_behaviour'].replace('!@9#%8', np.nan)
    df_to_treat['delay_from_due_date'] = df_to_treat['delay_from_due_date'].apply(lambda x: np.nan if x <= -2 else x)
    df_to_treat['amount_invested_monthly'] = df_to_treat['amount_invested_monthly'].replace('__10000__', np.nan)
    
    #Treating the type_of_loan feature
    loan_mapping = {
        'Not Specified': 0,
        'Student Loan': 1,
        'Payday Loan': 2,
        'Mortgage Loan': 3,
        'Personal Loan': 4,
        'Credit-Builder Loan': 5,
        'Auto Loan': 6,
        'Debt Consolidation Loan': 7,
        'Home Equity Loan': 8,
        'Personal Loan, and Student Loan': 9
    }

    df_to_treat['type_of_loan'] = df_to_treat['type_of_loan'].replace(loan_mapping)
    df_to_treat['type_of_loan'] = df_to_treat['type_of_loan'].apply(lambda x: x if isinstance(x, int) else 10)
    
    return df_to_treat

def treating_features_to_model(df_to_treat):
    
    #target encoding    
    df_to_treat['occupation'] = df_to_treat['occupation'].fillna('null')
    df_to_treat['type_of_loan'] = df_to_treat['type_of_loan'].fillna('null')
    df_to_treat['credit_mix'] = df_to_treat['credit_mix'].fillna('null')
    df_to_treat['payment_of_min_amount'] = df_to_treat['payment_of_min_amount'].fillna('null')
    df_to_treat['payment_behaviour'] = df_to_treat['payment_behaviour'].fillna('null')
    
    mapping_occupation = {
        'Accountant':   0.626749,
        'Architect':    0.607622,
        'Developer':    0.615482,
        'Doctor':       0.599268,
        'Engineer':     0.621480,
        'Entrepreneur': 0.637573,
        'Journalist':   0.593723,
        'Lawyer':       0.607901,
        'Manager':      0.603464,
        'Mechanic':     0.653813,
        'Media_Manager':0.589697,
        'Musician':     0.596059,
        'Scientist':    0.632461,
        'Teacher':      0.621343,
        'Writer':       0.677188,
        'null':-1
    }
    
    df_to_treat['occupation'] = df_to_treat['occupation'].replace(mapping_occupation)
    
    mapping_loan = {
        0:     0.341253,
        1:     0.304878,
        2:     0.306024,
        3:     0.228792,
        4:     0.261307,
        5:     0.244949,
        6:     0.328841,
        7:     0.260504,
        8:     0.236842,
        9:     0.474454,
        10:    0.654609
    }
    
    df_to_treat['type_of_loan'] = df_to_treat['type_of_loan'].replace(mapping_loan)
    
    mapping_mix = {
        'Bad':   0.974594,
        'Good':  0.244559,
        'Standard': 0.791324,
        'null':-1
    }
    
    df_to_treat['credit_mix'] = df_to_treat['credit_mix'].replace(mapping_mix)
    
    mapping_payment_min = {
        'NM':   0.610508,
        'Yes':  0.914622,
        'No': 0.258301
    }
    
    df_to_treat['payment_of_min_amount'] = df_to_treat['payment_of_min_amount'].replace(mapping_payment_min)
    
    mapping_payment = {
        'High_spent_Large_value_payments':  0.481272,
        'High_spent_Medium_value_payments': 0.559731,
        'High_spent_Small_value_payments':  0.594451,
        'Low_spent_Large_value_payments':   0.611529,
        'Low_spent_Medium_value_payments':  0.629492,
        'Low_spent_Small_value_payments':   0.733886,
        'null':                             -1
    }
    
    df_to_treat['payment_behaviour'] = df_to_treat['payment_behaviour'].replace(mapping_payment)
    
    return df_to_treat

print('Running preprocessing script')

print('Loading the inference dataset')
start_time = time()
df_to_treat = get_data()
total_time = time() - start_time
print(f'Total time to load the dataset: {total_time: .1f} seconds \n')

print('Preprocessing the data')
start_time = time()
df_to_treat = feats_transform_type(df_to_treat)
df_to_treat = preprocessing_data(df_to_treat)
df_to_treat = treating_features_to_model(df_to_treat)
total_time = time() - start_time
print(f'Total time to preprocess the dataset: {total_time: .1f} seconds \n')

print('Saving the dataset...')
df_to_treat.to_csv('../data/inference_data.csv')
print('End of the preprocessing script')
