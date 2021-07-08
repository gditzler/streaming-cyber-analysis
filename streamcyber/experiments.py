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

import os 
import numpy as np 
import pandas as pd 
import pickle as pkl 
from scipy.stats import poisson 

from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import OzaBaggingClassifier 
from skmultiflow.meta import OzaBaggingADWINClassifier 
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.evaluation import EvaluatePrequential, EvaluateHoldout

from .utils import load_dataset, standardize_df_off_tr
from .utils import jaccard, kuncheva, calc_metrics

import matplotlib.pylab as plt 
from skfeature.function.information_theoretical_based import JMI, LCSI


def exp_make_poisson_plots(output_path:str='outputs/'):
    """make the poisson plots for the paper 

    Parameters 
    ----------------- 
    output_path : str
       path to the output directory [save a pdf figure]
    """ 
    p1, p2, p3 = poisson(1.), poisson(5.), poisson(10.)
    x = [i for i in range(20)]

    plt.figure()
    plt.plot(x, p1.pmf(x), marker='s', label='Lambda=1')
    plt.plot(x, p2.pmf(x), marker='s', label='Lambda=5')
    plt.plot(x, p3.pmf(x), marker='s', label='Lambda=10')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('Pr(X=k)')
    plt.savefig(output_path + 'poisson.pdf')


def exp_make_jmi_plots(subscription_id, 
                       resource_group, 
                       workspace_name, 
                       dataset_name:str='unswnb', 
                       output_path:str='outputs/'): 
    """plot the mutual information between variables and labels

    Parameters 
    ----------------- 
    subscription_id : str 
        Mircroft Azure ML subscription ID 
    resource_group : str 
        Micrsoft Azure Resource Group 
    workspace_name : str 
        Micrsoft Azure Workspace Name 
    dataset_name : str
        dataset to process (saved as an Azure blob). [unswnb, nslkdd] 
    output_path : str
        path to the output directory 
    """
    df_tr, df_te = load_dataset(subscription_id, resource_group, workspace_name, dataset_name)

    Xtr, Xte, ytr, yte = df_tr.values[:,:-1], df_te.values[:,:-1], df_tr.values[:,-1], df_te.values[:,-1]

    mitr, mite = mutual_info_classif(Xtr, ytr), mutual_info_classif(Xte, yte)
    sort_mitr, sort_mite = np.argsort(mitr)[::-1], np.argsort(mite)[::-1]

    k_max = 25
    K = Xtr.shape[1]

    jacs, kunc, x = [], [], []
    for k in range(3, k_max+1):
        x.append(k)
        jacs.append(jaccard(sort_mitr[:k], sort_mite[:k]))
        kunc.append(kuncheva(sort_mitr[:k], sort_mite[:k], K))

    plt.figure()
    plt.plot(x, jacs, marker='o', label='Jaccard')
    plt.plot(x, kunc, marker='s', label='Kuncheva')
    plt.xlabel('# Features Selected')
    plt.ylabel('Index')
    plt.legend()
    plt.savefig(''.join([output_path, dataset_name, '_feature_stabilities.pdf']))

    plt.figure()
    plt.plot(mitr[sort_mitr], marker='o', label='Train')
    plt.plot(mite[sort_mitr], marker='s', label='Test')
    plt.xlabel('Feature Index')
    plt.ylabel('I(X;Y)')
    plt.legend()
    plt.savefig(''.join([output_path, dataset_name, '_feature_ranks.pdf']))

    # print out the top 10 features in a latex list
    # print('\begin{enumerate}')
    # for i in sort_mitr[:10]:
    #     print('  \item ' + feature_names[i] + ': ' + feature_descr[i]) 
    # print('\end{enumerate}')

def exp_make_jmi_2D(subscription_id, 
                    resource_group, 
                    workspace_name, 
                    dataset_name:str='unswnb', 
                    output_path:str='outputs/'): 
    """generate a correlation-like plot of the joint mutual information

    Parameters 
    ----------------- 
    subscription_id : str 
        Mircroft Azure ML subscription ID 
    resource_group : str 
        Micrsoft Azure Resource Group 
    workspace_name : str 
        Micrsoft Azure Workspace Name 
    dataset_name : str
        dataset to process (saved as an Azure blob). [unswnb, nslkdd] 
    output_path : str
        path to the output directory 
    """
    # 1) randomly shuffle rows; 2) rename label column; 3) drop not useful columns 
    df_tr, _ = load_dataset(subscription_id, resource_group, workspace_name, dataset_name)
    df_tr_drop = df_tr

    # separate the data from the labels for the training and testing data 
    Xtr = df_tr_drop.values[:,:-1]
    ytr = df_tr_drop['target'].values

    # standardize the data based on the transform found from the training data 
    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr) 
    n, K = Xtr.shape
    feat_names = df_tr_drop.drop('target', axis = 1).keys()

    J = np.zeros((K, K))

    # fill in the off diagonals
    print('Running JMI')
    for i in range(K): 
        for j in range(K): 
            if j > i: 
                _, jcmi, mify = LCSI.lcsi(Xtr[:,[i,j]], ytr, function_name='JMI', n_selected_features=2)
                J[i,j], J[j,i] = mify[0] - jcmi[1]/2, mify[0] - jcmi[1]/2


    fig, ax = plt.subplots(figsize=(16, 12), dpi=80)
    im = ax.imshow(J)


    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('JMI Score', rotation=-90, va="bottom")
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(feat_names)))
    ax.set_yticks(np.arange(len(feat_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(feat_names, rotation=90)
    ax.set_yticklabels(feat_names)

    plt.savefig(''.join([output_path, dataset_name, '_jmis.pdf']))


def evaluate_binary_prequential(subscription_id, 
                                resource_group, 
                                workspace_name, 
                                poisson_parameter:float=4.,
                                dataset_name:str='unswnb', 
                                output_path:str='outputs/'): 
    """run the prequntial experiments 

    Parameters 
    ----------------- 
    subscription_id : str 
        Mircroft Azure ML subscription ID 
    resource_group : str 
        Micrsoft Azure Resource Group 
    workspace_name : str 
        Micrsoft Azure Workspace Name 
    poisson_paremeter : float 
        Poisson distribution parameter 
    dataset_name : str
        dataset to process (saved as an Azure blob). [unswnb, nslkdd] 
    output_path : str
        path to the output directory 
    """
    # number of data samples used to pre-train the model 
    pretrain_size = 2000
    # prequential batch sizes 
    batch_size = 1000
    # classification metrics to compute in prequential online learning 
    metrics = ['accuracy', 'f1', 'kappa', 'running_time', 'model_size']
    # maximum number of samples to process in the stream
    max_samples = 1000000
    output_files = [
                        'output_ht_m',
                        'output_bag_m',
                        'output_bagwin_m',
                        'output_leverage_m'
                    ]


    df_tr, df_te = load_dataset(subscription_id, resource_group, workspace_name, dataset_name) 

    N1 = len(df_tr)

    # instantiate a based classifier from each of the classifiers in multiflow that 
    # we are going to benchmark against. 
    clfrs = [
                HoeffdingTreeClassifier(), 
                OzaBaggingClassifier(base_estimator=HoeffdingTreeClassifier()),
                OzaBaggingADWINClassifier(base_estimator=HoeffdingTreeClassifier()),
                LeveragingBaggingClassifier(base_estimator=HoeffdingTreeClassifier(), w=poisson_parameter)
            ]
    N = len(clfrs)

    # concat the training and testing data into a stream 
    df = pd.concat([df_tr, df_te])
    stream = DataStream(df, allow_nan=True)

    mdls = []
    evals = []

    for n in range(N): 
        mdl = clfrs[n]   # get the classifier 
        # configure the prequential datastream evaluator. results are saved to the 
        # google drive. 
        eval = EvaluatePrequential(show_plot=False, 
                                   pretrain_size=pretrain_size, 
                                   batch_size=batch_size, 
                                   metrics=metrics, 
                                   max_samples=max_samples,
                                   output_file=''.join([output_path, output_files[n], '.csv']))
        # process the datastream then save off the models and evaluation 
        mdl = eval.evaluate(stream=stream, model=mdl)
        mdls.append(mdl[0])
        evals.append(eval)

    # write the output to a pickle file 
    data = {'mdls': mdls, 'evals': evals}
    pkl.dump(data, open(''.join([output_path, dataset_name, '_prequential_models_evaluators.pkl']), 'wb'))


    # the results from the prequential experiements need to be loaded from the drive 
    # since they are not in the environment
    df_ht = pd.read_csv(''.join([output_path, 'output_ht_m.csv']), comment='#')
    df_bag = pd.read_csv(''.join([output_path, 'output_bag_m.csv']), comment='#')
    df_bagwin = pd.read_csv(''.join([output_path, 'output_bagwin_m.csv']), comment='#')
    df_leverage = pd.read_csv(''.join([output_path, 'output_leverage_m.csv']), comment='#')

    def make_stat_plot(df_ht, df_bag, df_bagwin, df_leverage, N1, ylb:str, opath:str, idx:str, vals:list):

        # plot the accuracies 
        plt.figure()
        plt.plot(df_ht['id'], df_ht[idx], label='HT')
        plt.plot(df_bag['id'], df_bag[idx], label='Bag')
        plt.plot(df_bagwin['id'], df_bagwin[idx], label='BagADWIN')
        plt.plot(df_leverage['id'], df_leverage[idx], label='Leverage')
        plt.plot([N1,N1], [np.min(df_ht[idx]), vals[0]], color='k', linestyle='--')
        plt.text(N1+5000, vals[1], 'Change', fontsize=12)
        plt.legend()
        plt.xlabel('Sample Number')
        plt.ylabel(ylb)
        plt.savefig(opath)


    def make_time_plot(df_ht, df_bag, df_bagwin, df_leverage, opath:str, idx:str, ylb:str):
        plt.figure()
        plt.plot(df_ht['id'], df_ht[idx], label='HT')
        plt.plot(df_bag['id'], df_bag[idx], label='Bag')
        plt.plot(df_bagwin['id'], df_bagwin[idx], label='BagADWIN')
        plt.plot(df_leverage['id'], df_leverage[idx], label='Leverage')
        plt.legend()
        plt.xlabel('Sample Number')
        plt.ylabel(ylb)
        plt.savefig(opath)


    
    make_stat_plot(df_ht, df_bag, df_bagwin, df_leverage, N1, 'Accuracy', 
                   ''.join([output_path, dataset_name, '_online_accuracy.pdf']), 'mean_acc_[M0]', vals=[.96, .88])
    make_stat_plot(df_ht, df_bag, df_bagwin, df_leverage, N1, 'Kappa', 
                   ''.join([output_path, dataset_name, '_online_kappa_m.pdf']), 'mean_kappa_[M0]', vals=[.9, .72])
    make_stat_plot(df_ht, df_bag, df_bagwin, df_leverage, N1, 'F1-Score', 
                   ''.join([output_path, dataset_name, '_online_f1_m.pdf']), 'mean_f1_[M0]', vals=[.97, .92])
    make_time_plot(df_ht, df_bag, df_bagwin, df_leverage, 
                   ''.join([output_path, dataset_name, '_online_time_m.pdf']), 'total_running_time_[M0]', 'Time (s)')
    make_time_plot(df_ht, df_bag, df_bagwin, df_leverage, 
                   ''.join([output_path, dataset_name, '_online_test_time_m.pdf']), 'testing_time_[M0]', 'Time (s)')
    make_time_plot(df_ht, df_bag, df_bagwin, df_leverage, 
                   ''.join([output_path, dataset_name, '_online_size_m.pdf']), 'model_size_[M0]', 'Size (kB)')

    file = open(''.join([output_path, dataset_name, '_prequential_latex.txt']), 'w')
    # print out the latex tables 
    file.write('\\bf Metric & \\bf HT & \\bf Bagging & \\bf BagADWIN & \\bf Leverage \\\\')
    file.write('\\hline\\hline')
    file.write(''.join(['Accuracy & ', str(100*df_ht['mean_acc_[M0]'].values[-1]), 
                        ' & ', str(100*df_bag['mean_acc_[M0]'].values[-1]), 
                        ' & ', str(100*df_bagwin['mean_acc_[M0]'].values[-1]), 
                        ' & ', str(100*df_leverage['mean_acc_[M0]'].values[-1]), 
                        ' \\\\ '])
    )
    file.write(''.join(['$\kappa$ & ', str(100*df_ht['mean_kappa_[M0]'].values[-1]), 
               ' & ', str(100*df_bag['mean_kappa_[M0]'].values[-1]),  
               ' & ', str(100*df_bagwin['mean_kappa_[M0]'].values[-1]), 
               ' & ', str(100*df_leverage['mean_kappa_[M0]'].values[-1]) + ' \\\\ '])
    )
    file.write(''.join(['F1-Score & ', str(100*df_ht['mean_f1_[M0]'].values[-1]),
               ' & ', str(100*df_bag['mean_f1_[M0]'].values[-1]), 
               ' & ', str(100*df_bagwin['mean_f1_[M0]'].values[-1]), 
               ' & ', str(100*df_leverage['mean_f1_[M0]'].values[-1]) + ' \\\\ '])
    )
    file.close()


def evaluate_binary_holdout(subscription_id, 
                            resource_group, 
                            workspace_name, 
                            poisson_parameter:float=4.,
                            dataset_name:str='unswnb', 
                            output_path:str='outputs/'):
    """run the holdout experiments 

    Parameters 
    ----------------- 
    subscription_id : str 
        Mircroft Azure ML subscription ID 
    resource_group : str 
        Micrsoft Azure Resource Group 
    workspace_name : str 
        Micrsoft Azure Workspace Name 
    poisson_parameter : float 
        Poisson distribution parameter 
    dataset_name : str
        dataset to process (saved as an Azure blob). [unswnb, nslkdd] 
    output_path : str
        path to the output directory 
    """
    # number of data samples used to pre-train the model 
    pretrain_size = 2000
    # prequential batch sizes 
    batch_size = 1000
    # classification metrics to compute in prequential online learning 
    metrics = ['accuracy', 'f1', 'kappa', 'running_time', 'model_size']
    # maximum number of samples to process in the stream
    max_samples = 1000000
    df_tr, df_te = load_dataset(subscription_id, resource_group, workspace_name, dataset_name) 
    
    # set the data stream for learning. we are only going to use the training stream 
    # to build the model; however, unlike the previous experiement we do not care 
    # about the output result stream. rather we want the hold out performance 
    stream = DataStream(df_tr)

    X_train = df_tr.values[:,:-1]
    y_train = df_tr['target'].values
    X_hold = df_te.values[:,:-1]
    y_hold = df_te['target'].values

    # instantiate a based classifier from each of the classifiers in multiflow that 
    # we are going to benchmark against.
    clfrs = [
                HoeffdingTreeClassifier(), 
                OzaBaggingClassifier(base_estimator=HoeffdingTreeClassifier()),
                OzaBaggingADWINClassifier(base_estimator=HoeffdingTreeClassifier()),
                LeveragingBaggingClassifier(base_estimator=HoeffdingTreeClassifier(), w=poisson_parameter)
            ]
    N = len(clfrs)
    clfrs_names = ['HT', 'Bag', 'BagADWIN', 'Leverage']

    N = len(clfrs)
    accs, f1s, kappas = [], [], []

    for n in range(N): 
        # learn the model on the training data stream that is defined by the training 
        # data. this is specified by the original authors 
        mdl = clfrs[n]
        eval = EvaluatePrequential(show_plot=False, 
                             pretrain_size=pretrain_size, 
                             batch_size=batch_size, 
                             metrics=metrics, 
                             max_samples=max_samples,
                             output_file=''.join([output_path, 'tmp.csv']))
        mdl = eval.evaluate(stream=stream, model=mdl)

        # get the predictions on the hold out dataset then calculate the accuracy, 
        # f1-score and kappa statistics. 
        y_hat = mdl[0].predict(X_hold)
        acc, f1, kappa = calc_metrics(y_hold, y_hat)
        accs.append(acc)
        f1s.append(f1)
        kappas.append(kappa)

    # remove the temporary data file that was used to save the prequential results 
    # of the online models. all the algorithms write the same tmp.csv file. 
    os.remove(output_path + 'tmp.csv')

    # write the classification statistics to a python file  
    data = {'accs': accs, 'f1s': f1s, 'kappas': kappas, 'clfr_names': clfrs_names}
    pkl.dump(data, open(''.join([output_path, 'holdout_classifcation_statistics.pkl']), 'wb'))

    # evaluate the static classifiers on the train/test hold out experiment
    static_clfr_name = [
                    'CART (static)',
                    'Bagging (static)'
                    ]
    static_clfr = [DecisionTreeClassifier(), BaggingClassifier()]
    accs_static, f1s_static, kappas_static = [], [], []

    for n in range(len(static_clfr)):
        y_hat = static_clfr[n].fit(X_train, y_train).predict(X_hold)
        acc, f1, kappa = calc_metrics(y_hold, y_hat)
        accs_static.append(acc)
        f1s_static.append(f1)
        kappas_static.append(kappa)

    # print out the latex tables 
    file = open(''.join([output_path, dataset_name, '_holdout_latex.txt']), 'w')  
    ' & '.join(clfrs_names) + '\\\\'
    for n in range(N): 
        file.write(''.join([clfrs_names[n], ' & ' + str(100*accs[n]), ' & ', str(100*f1s[n]), ' & ',  str(100*kappas[n]), '\\\\']))

    for n in range(len(static_clfr)): 
        file.write(''.join([static_clfr_name[n], ' & ', str(100*accs_static[n]), ' & ', str(100*f1s_static[n]), ' & ', str(100*kappas_static[n]), '\\\\']))
    file.close()

 
def evaluate_lambda(subscription_id, 
                    resource_group, 
                    workspace_name, 
                    dataset_name:str='unswnb', 
                    output_path:str='outputs/'): 
    """run the ablation study 

    Parameters 
    ----------------- 
    subscription_id : str 
        Mircroft Azure ML subscription ID 
    resource_group : str 
        Micrsoft Azure Resource Group 
    workspace_name : str 
        Micrsoft Azure Workspace Name 
    dataset_name : str
        dataset to process (saved as an Azure blob). [unswnb, nslkdd] 
    output_path : str
        path to the output directory 
    """
    # number of data samples used to pre-train the model 
    pretrain_size = 2000
    # prequential batch sizes 
    batch_size = 1000
    # classification metrics to compute in prequential online learning 
    metrics = ['accuracy', 'f1', 'kappa', 'running_time', 'model_size']
    # maximum number of samples to process in the stream
    max_samples = 1000000
    LAMBDA = [1., 2., 5., 7., 10.]
  
    # read in the csv files saved in the google drive 
    df_tr, df_te = load_dataset(subscription_id, resource_group, workspace_name, dataset_name) 

    # concat the training and testing data into a stream 
    df = pd.concat([df_tr, df_te])
    stream = DataStream(df)

    mdls = []
    evals = []

    for n in range(len(LAMBDA)): 
        mdl = LeveragingBaggingClassifier(base_estimator=HoeffdingTreeClassifier(), w=LAMBDA[n])
        # configure the prequential datastream evaluator. results are saved to the 
        # google drive. 
        eval = EvaluatePrequential(show_plot=False, 
                             pretrain_size=pretrain_size, 
                             batch_size=batch_size, 
                             metrics=metrics, 
                             max_samples=max_samples,
                             output_file=''.join([output_path, dataset_name, '_lambda_', str(int(LAMBDA[n])), '.csv']))
        # process the datastream then save off the models and evaluation 
        mdl = eval.evaluate(stream=stream, model=mdl)
        mdls.append(mdl[0])
        evals.append(eval)
        
    # write the output to a pickle file 
    data = {'mdls': mdls, 'evals': evals}
    pkl.dump(data, open(''.join([output_path, 'prequential_models_evaluators_lambda.pkl']), 'wb'))

    # the results from the prequential experiements need to be loaded from the drive 
    # since they are not in the environment
    df_l1 = pd.read_csv(''.join([output_path, dataset_name, '_lambda_1.csv']), comment='#')
    df_l2 = pd.read_csv(''.join([output_path, dataset_name, '_lambda_2.csv']), comment='#')
    df_l5 = pd.read_csv(''.join([output_path, dataset_name, '_lambda_5.csv']), comment='#')
    df_l7 = pd.read_csv(''.join([output_path, dataset_name, '_lambda_7.csv']), comment='#')
    df_l10 = pd.read_csv(''.join([output_path, dataset_name, '_lambda_10.csv']), comment='#')


    def make_plot(y_label:str, index:str, label:str):
        plt.figure()
        plt.plot(df_l1['id'], df_l1[index], label='Lambda=1')
        plt.plot(df_l2['id'], df_l2[index], label='Lambda=2')
        plt.plot(df_l5['id'], df_l5[index], label='Lambda=5')
        plt.plot(df_l7['id'], df_l7[index], label='Lambda=7')
        plt.plot(df_l10['id'], df_l10[index], label='Lambda=10')
        plt.legend()
        plt.xlabel('Sample Number')
        plt.ylabel(y_label)
        plt.savefig(''.join([output_path, dataset_name, label])) 
        return None 

    make_plot('Accuracy', 'mean_acc_[M0]', '_lambda_accuracy_m.pdf')
    make_plot('Kappa', 'mean_kappa_[M0]', '_lambda_kappa_m.pdf')
    make_plot('F1-Score', 'mean_f1_[M0]', '_lambda_f1_m.pdf')
    make_plot('Time (s)', 'total_running_time_[M0]', '_lambda_time_m.pdf')
    make_plot('Time (s)', 'testing_time_[M0]', '_lambda_test_time_m.pdf')
    make_plot('Size (kB)', 'model_size_[M0]', '_lambda_size_m.pdf')
    