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

from .experiments import evaluate_binary_prequential
from .experiments import evaluate_binary_holdout
from .experiments import evaluate_lambda
from .experiments import exp_make_poisson_plots
from .experiments import exp_make_jmi_plots
from .experiments import exp_make_jmi_2D
from .read_azure import read_azure

__all__ = [
        'evaluate_binary_prequential',
        'evaluate_binary_holdout',
        'evaluate_lambda', 
        'exp_make_poisson_plots', 
        'exp_make_jmi_plots', 
        'exp_make_jmi_2D', 
        'read_azure'
    ]

__version__ = '0.1.0'