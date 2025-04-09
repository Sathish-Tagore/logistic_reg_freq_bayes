# logistic_reg_freq_bayes

## Order Structure: 

MODELS~1/ - This folder contains the model implementations of Frequentist methods, bayesian methods and stancodes

datasets/ - This folder contains the preprocessed data. Training, validation and test sets. Also the different sample sizes with seperate training and validation sets as csv values.

plots/- This folder contains the pdf versions of all the plots that has been used for the paper.

utils/- This folder contains two utility functions for creating the plots for uncertainty, misclassification and to calculate confidence intervals and classification performance metrics.

results/- This folder contains all the results saved as csv files for all the models compared in this paper. Due to the size of the folder, it is available here: https://zenodo.org/records/15181882


## Description of code files:

dataset.ipynb - This notebook is to prepare the dataset and preprocess. Also to subsample to obtain smaller datasets.

Bayesian_Analysis.ipynb - Code to perform the step by step analysis of the Bayesian results. Change only the "Loading the prior and posterior results" section to do the same for bayesian logistic regression with different priors. Also you can change the training, validation set to perform the sensitivity analysis on smaller samples.

Frequentist_Analysis.ipynb - Code to perform the step by step analysis of the Frequentist results. Change only the "Loading the frequentist bootstrapped results" section to do the same for Frequentist logistic regression with different regularisations.

Paperwork.ipynb - This code contains all steps followed to obtain specific figures for the paperwork. 

Tables.txt - This text file contains the classification performance of different methods as tables.

requirements.txt - This text file contains the python packages needed to perform this analysis and to run this models.

run_python_script.sh - This shell file contains to code to run the models in the HPC environment of the Boehringer Ingelheim.



## All models ran in a HPC environment. 

Hardware info:  A single node equipped with 48 CPU cores, 48GB RAM, and an Intel ® Xeon ® Platinum 8360Y CPU @ 2.40GHz.

MODELS~1/stancodes.py - This file contains the Stan codes for different priors and its respective posteriors. 

MODELS~1/run_bayesian_models.py - This file runs the Stan code, reading the input and MCMC sampling using the PyStan interface. Takes longer time. 

MODELS~1/Freq_logistic_reg.py - This file runs the frequentist logistic regression with no penalty by boostrapping and using random search cross validation.

MODELS~1/Freq_lasso_reg.py - This file runs the frequentist logistic regression with lasso penalty by boostrapping and using random search cross validation.

MODELS~1/Freq_elasticnet_reg.py - This file runs the frequentist logistic regression with elastic net penalty by boostrapping and using random search cross validation.








