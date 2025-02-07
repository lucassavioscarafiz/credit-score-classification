import pandas as pd
from sklearn.model_selection import train_test_split 
import os

# Load Dataset
def load_df():
    
    """Function to load the main and the oot dataset"""
    
    df = pd.read_csv('../data/train.csv')
    df_oot = pd.read_csv('../data/test.csv')
    
    print(f'Shape Main dataset:{df.shape}')
    print(f'Shape OOT dataset:{df_oot.shape}')
    
    return df, df_oot

#Split data
def split_data(df, oos_size):
    
    SEED = 42
    
    df_train, df_oos = train_test_split(df, test_size = oos_size, 
                                        random_state = SEED)
    
    print(f'Shape Dataset Train:{df_train.shape[0]}')
    print(f'Shape Dataset OOS:{df_oos.shape[0]}')
    
    return df_train, df_oos

