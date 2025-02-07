import pandas as pd

def load_df():
    
    """Function to load the main and the oot dataset"""
    
    df = pd.read_csv('../data/train.csv')
    df_oot = pd.read_csv('../data/test.csv')
    
    print(f'Shape Main dataset:{df.shape}')
    print(f'Shape OOT dataset:{df_oot.shape}')
    
    return df, df_oot

#df, df_oot = load_df()

#feats_int = ['Age','Num_of_Loan','Num_of_Delayed_Payment']
#feats_float = ['Annual_Income','Changed_Credit_Limit','Num_Credit_Inquiries','Outstanding_Debt','Amount_invested_monthly','Monthly_Balance']

class FeatureTypes:
    
    """Class to handle with the feature dtypes. Turning the object features into int32 or float64"""
    
    def __init__(self, df, feats_int, feats_float):
        self.df = df
        self.feats_int = feats_int
        self.feats_float = feats_float

    """Functions to treat the feature types"""
    def feats_transform(self):
        df_to_treat = self.df.copy()
        
        for feat in self.feats_int:
            df_to_treat[feat] = df_to_treat[feat].astype(str)  
            df_to_treat[feat] = df_to_treat[feat].str.replace('_', '', regex=False)  
            df_to_treat[feat] = df_to_treat[feat].replace(['', 'nan'], pd.NA)  
            df_to_treat[feat] = df_to_treat[feat].fillna(-1)  
            df_to_treat[feat] = df_to_treat[feat].astype(int)  
        
        for feat in self.feats_float:
            df_to_treat[feat] = df_to_treat[feat].astype(str)  
            df_to_treat[feat] = df_to_treat[feat].str.replace('_', '', regex=False)  
            df_to_treat[feat] = df_to_treat[feat].replace(['', 'nan'], pd.NA)  
            df_to_treat[feat] = df_to_treat[feat].fillna(-1.0)  
            df_to_treat[feat] = df_to_treat[feat].astype(float)  
        
        return df_to_treat    
    
#Chamando a classe e armazenando em um objeto
#feature_handler = FeatureTypes(df, feats_int, feats_float)

# Use os métodos
#df_int_transformed = feature_handler.feats_int_transform()
    
def credit_hist_transform(df):
    
    """Function to split the Credit_History_Year column in Years
    
    Args:
        df (pd.DataFrame): DataFrame that will be transform.

    Returns:
        pd.DataFrame: DataFrame transformed.
    
    """
    
    df_to_treat = df.copy()
    
    df_to_treat['Credit_History_Year'] = df_to_treat['Credit_History_Age'].str.extract(r'(\d+) Years')
    df_to_treat['Credit_History_Year'] = df_to_treat['Credit_History_Year'].fillna(-1)
    df_to_treat['Credit_History_Year'] = df_to_treat['Credit_History_Year'].astype(int) 
    
    del df_to_treat['Credit_History_Age']
    
    return df_to_treat

def age_transform(df):
    
    """Function to fill <18 and >120 values to -1
    
    Args:
        df (pd.DataFrame): DataFrame that will be transform.

    Returns:
        pd.DataFrame: DataFrame transformed.
    
    """
    
    df_to_treat = df.copy()
    
    df_to_treat['Age'] = df_to_treat['Age'].apply(lambda x: -1 if x < 18 or x > 120 else x)
    
    return df_to_treat

def target_filter(df):
    
    """
    Function to filter only the Good and Poor customers
    
    Args:
        df (pd.DataFrame): DataFrame that will be transform.

    Returns:
        pd.DataFrame: DataFrame transformed.
    
    """
    
    df_to_treat = df.copy()
    
    df_to_treat = df_to_treat[df_to_treat['Credit_Score'].isin(['Good', 'Poor'])]
    df_to_treat['Credit_Score'] = df_to_treat['Credit_Score'].replace({'Good': 0, 'Poor': 1})
    df_to_treat = df_to_treat.rename(columns={'Credit_Score': 'target'})
    
    
    return df_to_treat

def feature_names(df):
    
    """Function to change the feature names in lower cases
    
    Args:
        df (pd.DataFrame): DataFrame that will be transform.

    Returns:
        pd.DataFrame: DataFrame transformed.
    """
    
    df_to_treat = df.copy()
    
    df_to_treat.columns = [col.lower() for col in df_to_treat.columns]
    
    return df_to_treat
    
    




