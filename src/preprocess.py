import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def pipeline_preprocess(data: pd.DataFrame, estimator = None):
    
    "Preprocessing pipeline to fill nummerical and categorical features null values"
    
    """Args: 
            data: dataset pandas
            estimator: the machine learning algorithm
       Return:
            Preprocess pipeline
       Goal:
            Use this pipeline to fill categorical and nummerical features null values using Simple Imputer and One Hot Encoder 
    """
    
    #selection of numeric features 
    numeric_features = data.select_dtypes(exclude = ['object','category']).columns
    
    #fill num features with -1
    numeric_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = -1))
    ]) 
    
    #selection of categorical features
    categorical_features = data.select_dtypes(include = ['object','category']).columns
    
    #fill cat features with 'null'
    categorical_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'null')),
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ])
    
    #Create the preprocessor pipeline
    preprocessor = ColumnTransformer (
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

def get_pipeline(data: pd.DataFrame, estimator):
    
    #run the pipeline_preprocess 
    preprocessor_pipeline = Pipeline([
        ('preprocessor', pipeline_preprocess(data, estimator)),
        ('estimator', estimator)
    ])
    
    return preprocessor_pipeline