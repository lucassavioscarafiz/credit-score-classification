# Credit Risk Scoring Classification

## 1.0. Description 💻

▶ The main goal of this project is to create an credit score classification machine learning algorithm, which is know for bank as an Application Score Model.  

▶ Application Models serves to help the decision to approve, or not, new clients for a bank.

▶ The Application Score model is classification algorithm, so the target is basically True (Approved) or False (Reproved). But in terms of credit we will use score from 0 to 1000. Highest the score, better the client, and lowest the score, worst the client will be. The client will be allocated in different bins, and the credit analysts can use those bins to decide which clients will be approved based on the bad rate.

### ⚙ Project Structure ⚙

/credit-scoring-inference/
│── data/                      # Datasets
│   ├── test.csv
│   ├── oos_scored.csv
│   ├── df_transformed.csv
│   ├── df_transformed_new.csv
│   ├── train.csv
│   ├── train_scored.csv
│   ├── inference_data.csv
│   ├── inference_data_scored.csv
│── model_artifact/            # Best Model folder
│   ├── exp6_lgbm_classifier_iv_vars.pkl
│── notebooks/                 # Notebooks 
|   ├── 1_data_cleaning.ipynb
|   ├── 2_EDA.ipynb
|   ├── 3_modelling.ipynb
│   ├── 4_inference_pipe.ipynb  # Inference pipe notebook
│── src/                      #Archives .py
|   ├── __init__.py
|   ├── data_load.py
|   ├── df_cleaning.py
|   ├── eval.py
|   ├── iv.py
|   ├── pre_process.py
|   │── inference_pipe/
│           ├── preprocessing.py
│           ├── inference.py
│── requirements.txt
│── .github/
│   ├── workflows/
│       ├── run_pipeline.yml  # Arquivo do GitHub Actions
│── README.md

## 2.0. Model Building 🔧

▶ Divided the train.csv dataset into train and oos (out of sample) and the test.csv are the oot (out-of-time) dataset to make the inference.

▶ The Application model was created using the `LGBMClassifier()` algorithm.

▶ The hyperparameters were found by using `BayesianSearch()` technique.

params = {
    'lgbm': {
        'learning_rate': 0.11699106844426276,
        'max_depth': 5,
        'num_leaves': 220,
        'n_estimators': 350,
        'min_child_samples': 250,
        'reg_lambda': 0.6,
        'reg_alpha': 0.1,
        'subsample': 0.7799999999999999
    }
}

▶ The features choosed for the model are the top features with highest Information Values (IV) values (statistical technique used on credit risk models) and with the best SHAP shape

## 3.0. Results 📈

▶ To evaluate the model the follow techniques are used: 
1) General and KS (Kolmogorov-Smirnov) test per month
2) AUC & Calibration plot curve

Train KS  | OOS KS | AUC Curve | 
--------- | ------ | --------- |  
0.831     | 0.793  | 0.975     | 

Sample type  | Month | Bad Rate (%) | KS    |  
------------ | ----- | ------------ | ----- |  
Train        | Jan   | 64.1%        | 0.90  |
Train        | Feb   | 63.6%        | 0.90  |
Train        | Mar   | 64.1%        | 0.91  |
Train        | Apr   | 60.8%        | 0.81  |
Train        | May   | 62.1%        | 0.81  |
Train        | Jun   | 61.2%        | 0.80  |
Train        | Jul   | 60.0%        | 0.77  |
Train        | Aug   | 59.4%        | 0.77  |

![image](https://github.com/user-attachments/assets/46ea3bb4-19cc-494e-8c18-ba5ac5cc6261)

Sample type  | Month | Bad Rate (%) | KS    |  
------------ | ----- | ------------ | ----- |  
Test         | Jan   | 65.7%        | 0.89  |
Test         | Feb   | 64.0%        | 0.89  |
Test         | Mar   | 64.3%        | 0.89  |
Test         | Apr   | 61.4%        | 0.77  |
Test         | May   | 59.8%        | 0.77  |
Test         | Jun   | 61.1%        | 0.78  |
Test         | Jul   | 59.7%        | 0.70  |
Test         | Aug   | 61.8%        | 0.72  |

![image](https://github.com/user-attachments/assets/319f47a5-fa0e-4ff4-ac11-43dc4e710498)

3) Stability of bins in Train and OOS datasets

Bins  | Vol    | Bad Rate (%) |  
----- | ------ | ------------ |   
0     | 13,014 | 99.96%       |  
1     |  5,165 | 98.18%       | 
2     |  1,874 | 81.37%       |  
3     |  2,028 | 69.03%       |  
4     |  2,431 | 43.89%       |  
5     |  1,898 | 28.08%       |  
6     |  2,243 | 15.56%       | 
7     |  3,014 |  5.74%       | 
8     |  5,793 |  0.83%       |  

![image](https://github.com/user-attachments/assets/f8cd5f77-b924-4eef-a792-6f1af3cfd001)

## 4.0. Inference (MLOps) 📊

▶ The `inference_pipe` notebook serves to make the inference (predict score) for new datasets.

▶ The `test.csv` dataset is used as an example for the inference pipe.

▶ The `inference_pipe` notebook uses the `preprocessing.py` and `inference.py` files to treat the dataset and predict the scores and bins

▶ The code uses MLOps and Software Engineering techniches 


<h3 align="left">Contact me:</h3>
<p align="left">
<a href="https://linkedin.com/in/lucassavioscarafiz" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="lucassavioscarafiz" height="30" width="40" /></a>
</p>

📫 **lucassavioscarafiz@gmail.com**

<h3 align="left">Linguagens e Ferramentas:</h3>
<p align="left"> <a href="https://cloud.google.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" alt="gcp" width="40" height="40"/> </a> <a href="https://www.mysql.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> </p>








