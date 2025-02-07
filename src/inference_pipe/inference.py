import sys
import os
import argparse
import datetime
from time import time
import numpy as np
import pandas as pd
import joblib
import pickle 

def model_fn():
    
    """Function for model serving to unpack the model"""
    
    with open("../model_artifact/exp6_lgbm_classifier_iv_vars.pkl", "rb") as arquivo:
        model_dir = pickle.load(arquivo)
    
    return model_dir

def load_data():
    
    """Function to load the dataset"""
    
    df = pd.read_csv('../data/inference_data.csv')
    
    print(f'Shape Main dataset:{df.shape}')
    
    return df

def predict_fn(data, model):
    
    """
        Function to make predictions
        
        Args:
            data (pandas.DataFrame): returned dataframe from input_fn 
            model (sklearn model): returned model loaded from model_fn
            
        Output:
            data (pandas.DataFrame): return the scored dataframe with predictions
        
    """
    
    model_features = [
        'outstanding_debt',
        'interest_rate',
        'delay_from_due_date',
        'credit_mix',
        'num_credit_inquiries',
        'credit_history_year',
        'num_bank_accounts',
        'num_of_loan',
        'num_credit_card'
    ]
    
    bins_opt = [0,17.99478817, 148.1335907 , 367.32455444, 525.04098511, 653.3704834 , 731.36291504, 808.33175659, 880.6439209, 1000]

    labels_opt = [0,1,2,3,4,5,6,7,8]
    
    df_scored = data.copy() 
    
    df_scored['score_modelv6'] = 1000*model.predict_proba(df_scored[model_features])[:,1]
    df_scored['score_modelv6'] = 1000-df_scored['score_modelv6']
    df_scored['gh'] = pd.cut(df_scored['score_modelv6'], bins = bins_opt, labels = labels_opt)
    
    return df_scored

print('Running inference script')

print('Loading the inference dataset and the model artifact')
start_time = time()
df_to_inf = load_data()
model = model_fn()
total_time = time() - start_time
print(f'Total time to load the inference dataset and the model artifact: {total_time: .1f} seconds \n')

print('Predicting the score')
start_time = time()
df_scored = predict_fn(df_to_inf,model)
total_time = time() - start_time
print(f'Total time to preprocess the dataset: {total_time: .1f} seconds \n')

print('Saving the scored dataset...')
df_scored.to_csv('../data/inference_data_scored.csv')
print('End of the inference script')
