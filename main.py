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

from streamcyber import exp_make_poisson_plots, exp_make_jmi_plots
from streamcyber import exp_make_jmi_2D, evaluate_binary_prequential
from streamcyber import evaluate_lambda, evaluate_binary_holdout
from streamcyber import read_azure

DATASET = ['awid', 'nslkdd', 'unswnb']

if __name__ == '__main__': 

    subscription_id, resource_group, workspace_name = read_azure()
    
    # generate the plots for the poisson parameters 
    exp_make_poisson_plots(output_path='outputs/')
    
    for dataset_name in DATASET: 
        # make the JMI plots for the 
        exp_make_jmi_plots(subscription_id=subscription_id, 
                           resource_group=resource_group, 
                           workspace_name=workspace_name, 
                           dataset_name=dataset_name, 
                           output_path='outputs/')
    
        # make the 2D JMI plot 
        exp_make_jmi_2D(subscription_id=subscription_id, 
                        resource_group=resource_group, 
                        workspace_name=workspace_name, 
                        dataset_name=dataset_name, 
                        output_path='outputs/')

        # perform prequential exp
        evaluate_binary_prequential(subscription_id=subscription_id, 
                                    resource_group=resource_group, 
                                    workspace_name=workspace_name, 
                                    poisson_parameter=4.,
                                    dataset_name=dataset_name, 
                                    output_path='outputs/')
        
        # perform hold out exp 
        evaluate_binary_holdout(subscription_id=subscription_id, 
                                resource_group=resource_group, 
                                workspace_name=workspace_name, 
                                poisson_parameter=4.,
                                dataset_name=dataset_name, 
                                output_path='outputs/')

        # run the study of sampling 
        evaluate_lambda(subscription_id=subscription_id, 
                        resource_group=resource_group, 
                        workspace_name=workspace_name,
                        dataset_name=dataset_name, 
                        output_path='outputs/')
