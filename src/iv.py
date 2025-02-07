import pandas as pd 
from pandas.api.types import is_numeric_dtype
import copy 
import numpy as np
from scipy.stats import ks_2samp, stats 
import matplotlib.pyplot as plt
import seaborn as sns 

pd.set_option("mode.chained_assignment", None)

class CategoricalFeature:
    def __init__(self, df, feature, TARGET):
        self.df = df
        self.feature = feature
        self.target = TARGET 
        
    @property 
    def df_lite(self):
        df_lite = self.df
        df_lite['bin'] = df_lite[self.feature].fillna("MISSING")
        
        return df_lite[['bin', self.target]]
    
class ContinuousFeature:
    def __init__(self, df, feature, TARGET):
        self.df = df
        self.feature = feature
        self.bin_min_size = int(len(self.df) * 0.05)
        self.target = TARGET 
        
    def generate_bins(self, bins_num):
        df = self.df[[self.feature, self.target]]
        df['bin'] = (
            pd.qcut(df[self.feature], bins_num, duplicate = 'drop').apply(lambda x: x.left).astype(float)  
        )
        
        return df
    
def __correct_bins(self, bins_max = 5):
    for bins_num in range(bins_max, 1, -1):
        df = self.__generate_bins(bins_num)
        df_grouped = pd.DataFrame(
            df.groupby('bin').agg({self.feature: 'count', self.target: 'sum'})
        ).reset_index()
        
        if (
          df_grouped[self.feature].min() > self.bin_min_size
          and not (df_grouped[self.feature] == df_grouped[self.target]).any()
        ):
            break
    
    return df

    @property
    def df_lite(self):
        

def calculate_iv(data, TARGET):
    
    """ Fuction to calculate the Information Value (IV) for the model features """
    
    data = copy.deepcopy(data)
    
    for col in [c for c in data.columns if c not in [TARGET]]:
        if is_numeric_dtype(data[col]):
            feats_dict[col] = ContinuousFeature(data, col, TARGET)
        else:
            feats_dict[col] = CategoricalFeature(data, col, TARGET)
    
    feats = list(feats_dict.values())
    
    iv = IV(TARGET)
    ar = AttributeRelevance()
    iv_data = ar.bulk_iv(feats, iv).sort_values(by = ['iv'], ascending = False)
    
    ar.draw_iv(feats, iv)
    
    ar.draw_woe_extremes(feats, iv)
    
    iv_data['iv'] = iv_data['iv'].astype('string')
    iv_data['iv'] = iv_data['iv'].str.replace('.',',',regex = False)
    
    iv_data.to_csv('tabela_iv_csv')
    
    return iv_data