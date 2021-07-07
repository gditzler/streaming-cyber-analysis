#!/usr/bin/env python 

# Copyright 2021 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

import pandas as pd 
import numpy as np 
from azureml.core import Workspace, Dataset


def load_dataset(subscription_id, resource_group, workspace_name, dataset_name:str='unswnb'): 
    """load the unswnb15 dataset saved in the azure blob
    """
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    
    if dataset_name == 'unswnb': 
        dataset_train = Dataset.get_by_name(workspace, name='UNSWNB-15_training')
        dataset_test = Dataset.get_by_name(workspace, name='UNSWNB-15_testing')
        df_tr, df_te = proc_unswnb(dataset_train.to_pandas_dataframe(), dataset_test.to_pandas_dataframe())
    elif dataset_name == 'nslkdd': 
        dataset_train = Dataset.get_by_name(workspace, name='NSLKDD_training')
        dataset_test = Dataset.get_by_name(workspace, name='NSLKDD_testing')
        df_tr, df_te = proc_nslkdd(dataset_train.to_pandas_dataframe(), dataset_test.to_pandas_dataframe())
    elif dataset_name == 'awid': 
        dataset_train = Dataset.get_by_name(workspace, name='AWID_training')
        dataset_test = Dataset.get_by_name(workspace, name='AWID_testing')
        df_tr, df_te = proc_awid(dataset_train.to_pandas_dataframe(), dataset_test.to_pandas_dataframe())
    else: 
        raise(ValueError('Unknown dataset name: %s' % dataset_name))

    return df_tr, df_te

def proc_awid(df_tr:pd.DataFrame, df_te:pd.DataFrame): 
    """
    """
    df_tr, df_te = df_tr.sample(frac=1.).reset_index(drop=True), df_te.sample(frac=1.).reset_index(drop=True)
    df_tr, df_te = standardize_df_off_tr(df_tr, df_te)
    return df_tr, df_te

def nslkddProtocolType(df_set:pd.DataFrame):
    df_set['protocol_type'][df_set['protocol_type'] == 'tcp'] = 0
    df_set['protocol_type'][df_set['protocol_type'] == 'udp'] = 1
    df_set['protocol_type'][df_set['protocol_type'] == 'icmp'] = 2
    return df_set


def nslkddService(df_set:pd.DataFrame):
    servicetypes = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf',
                    'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i',
                    'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data',
                    'gopher', 'harvest', 'hostnames', 'http', 'http_2784',
                    'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap',
                    'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name',
                    'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat',
                    'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2',
                    'pop_3', 'printer', 'private', 'red_i', 'remote_job',
                    'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc',
                    'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time',
                    'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois',
                    'X11', 'Z39_50'
                    ]
    for i, servicename in enumerate(servicetypes):
        df_set['service'][df_set['service'] == servicename] = i
    return df_set


def nslkddFlag(df_set:pd.DataFrame):
    flagtypes = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2',
                 'S3', 'SF', 'SH'
                 ]
    for i, flagname in enumerate(flagtypes):
        df_set['flag'][df_set['flag'] == flagname] = i
    return df_set



def proc_nslkdd(df_tr, df_te):
    """Load the NSL-KDD dataset from the data/ folder. Note you need to download the data 
    and add it to the folder. 
    :return Four Numpy arrays with X_tr, y_tr, X_te and y_te
    """
    df_tr = df_tr.sample(frac=1).reset_index(drop=True).rename(columns={"class": "target"})
    df_te = df_te.sample(frac=1).reset_index(drop=True).rename(columns={"class": "target"})

    # change the name of the label column. this needs be be done if we are going to feed it into the 
    # data frame standardizer 
    df_tr['target'][df_tr['target']=='normal'] = 0
    df_tr['target'][df_tr['target']=='anomaly'] = 1

    df_te['target'][df_te['target']=='normal'] = 0
    df_te['target'][df_te['target']=='anomaly'] = 1

    df_tr, df_te = nslkddProtocolType(df_tr), nslkddProtocolType(df_te)
    df_tr, df_te = nslkddService(df_tr), nslkddService(df_te)
    df_tr, df_te = nslkddFlag(df_tr), nslkddFlag(df_te)
    
    df_tr, df_te = standardize_df_off_tr(df_tr, df_te)
    df_tr = df_tr.drop(df_tr.keys()[19], axis = 1)
    df_te = df_te.drop(df_te.keys()[19], axis = 1)

    df_tr = df_tr.drop(['protocol_type', 'service', 'flag'], axis = 1)
    df_te = df_te.drop(['protocol_type', 'service', 'flag'], axis = 1)
    df_te = df_te.astype(float)
    df_tr = df_tr.astype(float)
    df_tr['target'], df_te['target'] = df_tr['target'].astype(int), df_te['target'].astype(int)
    return df_tr, df_te



def proc_unswnb(df_tr, df_te): 
    """
    """
    drop_cols = ['id', 'proto', 'service', 'state', 'attack_cat', 'is_sm_ips_ports']
    df_tr = df_tr.sample(frac=1).reset_index(drop=True).rename(columns={"label": "target"}).drop(drop_cols, axis = 1)
    df_te = df_te.sample(frac=1).reset_index(drop=True).rename(columns={"label": "target"}).drop(drop_cols, axis = 1)
    df_tr, df_te = standardize_df_off_tr(df_tr, df_te)
    return df_tr, df_te

def get_attack_counts(df:pd.DataFrame):
      """Count the number of attacks per class.
      """
      attacks = np.unique(df['attack_cat'])
      n_attacks = np.array([(df['attack_cat']==attack).values.sum() for attack in attacks])
      sorted_idx = np.argsort(n_attacks)[::-1]
      return sorted_idx, attacks, n_attacks
    
def standardize_df(df:pd.DataFrame): 
    """Standardize a dataframe
    """
    for key in df.keys(): 
        if key != 'target': 
            df[key] = (df[key].values - df[key].values.mean())/df[key].values.std()
    return df

def standardize_df_off_tr(df_tr:pd.DataFrame, 
                          df_te:pd.DataFrame): 
    """Standardize dataframes from a training and testing frame, where the means
    and standard deviations that are calculated from the training dataset. 
    """
    for key in df_tr.keys(): 
        if key != 'target': 
            # scale the testing data w/ the training means/stds
            ssd = df_tr[key].values.std()
            if np.abs(ssd) < .0001: 
                ssd = .001
            df_te[key] = (df_te[key].values - df_tr[key].values.mean())/ssd
            # scale the training data 
            df_tr[key] = (df_tr[key].values - df_tr[key].values.mean())/ssd
    return df_tr, df_te

def calc_metrics(y:np.ndarray, yhat:np.ndarray):
    """Calculate the accuracy, f1-score and kappa statistics 
    """
    tp = 1.*len(np.where((y == 0) & (yhat == 0))[0])
    tn = 1.*len(np.where((y == 1) & (yhat == 1))[0])
    fp = 1.*len(np.where((y == 1) & (yhat == 0))[0])
    fn = 1.*len(np.where((y == 0) & (yhat == 1))[0])
    n = tp+tn+fp+fn
    acc = (tp+tn)/(tp+tn+fp+fn)
    f1 = 2*tp/(2*tp+fp+fn)
  
    po = 1.*acc
    py, pn = ((tp+fp)/n)*((tp+fn)/n), ((fn+tn)/n)*((fp+tn)/n)
    pe = py+pn
    kappa = (po-pe)/(1-pe)
    return acc, f1, kappa 

def jaccard(a, b): 
    """Compute the jaccard index between two feature sets 
    """
    return 1.*len(set(a).intersection(set(b)))/len(set(a).union(set(b)))

def kuncheva(a:np.ndarray, b:np.ndarray, K:int): 
    """Compute the kuncheva index between two sets 
    """
    k = len(a)
    a, b = set(a[:k]), set(b[:k])
    r = 1.*len(a.intersection(b))
    return (r*K-k**2)/(k*(K-k))
