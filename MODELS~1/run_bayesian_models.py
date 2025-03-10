import pandas as pd
import numpy as np
import stan
from sklearn.model_selection import train_test_split
from Models import stancodes
import logging
import datetime


'''
To run this file, for a bayesian approach both with prior and posterior approach.
--------------------------------------------------------------------------------------------------------------
Weakly informative prior 
Stan code: winp_prior
prior_data = {"N": X_train.shape[0], 
              "d": X_train.shape[1], 
              "X" : X_train} 
Stan code: winp_posterior
posterior_data = {"N": X_train.shape[0], 
                  "d": X_train.shape[1], 
                  "X" : X_train, 
                  "y": y_train, 
                  "N_val": X_val.shape[0], 
                  "X_val": X_val}
--------------------------------------------------------------------------------------------------------------
Lasso prior
---------------
Stan code: lasso_prior
prior_data = {"N": X_train.shape[0], 
              "d": X_train.shape[1], 
              "X" : X_train} 
Stan code: lasso_posterior
posterior_data = {"N": X_train.shape[0], 
                  "d": X_train.shape[1], 
                  "X" : X_train, 
                  "y": y_train, 
                  "N_val": X_val.shape[0], 
                  "X_val": X_val}
--------------------------------------------------------------------------------------------------------------
For Elastic net prior
---------------------
Stan code: esnet_prior
prior_data = {"N": X_train.shape[0], 
              "d": X_train.shape[1], 
              "X" : X_train} 
Stan code: esnet_posterior
posterior_data = {"N": X_train.shape[0], 
                  "d": X_train.shape[1], 
                  "X" : X_train, 
                  "y": y_train, 
                  "N_val": X_val.shape[0], 
                  "X_val": X_val}
--------------------------------------------------------------------------------------------------------------
For regularised horseshoe prior
--------------------------------
Stan code: rhs_prior
prior_data = { "d": X_train.shape[1],
               "scale_global": (10 / (X_train.shape[1] - 10)) / np.sqrt(X_train.shape[0]),
               "N": X_train.shape[0],
               "X" : X_train} 
Stan code: rhs_posterior
posterior_data = {"N": X_train.shape[0], 
                  "d": X_train.shape[1],
                  "X" : X_train,
                  "y": y_train,
                  "N_val": X_val.shape[0],
                  "X_val": X_val,
                  "scale_global": (10 / (X_train.shape[1] - 10)) / np.sqrt(X_train.shape[0]),
                  "nu_global":1,
                  "nu_local":1,
                  "slab_scale":2,
                  "slab_df":4}
--------------------------------------------------------------------------------------------------------------

'''

# Create or append to a logger
logging.basicConfig(filename='runtimes7.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='a')

# Enter the model name to track time
#logging.info('Model name: Logistic prior and posterior sampling')
#logging.info('Model name: Lasso prior and posterior sampling')
#logging.info('Model name: Elasticnet prior and posterior sampling')
logging.info('Model name: Regularised horseshoe prior and posterior sampling (300)')

# Load the training and validation sets
training_set = pd.read_csv("datasets/training_set_300.csv", index_col=0)
validation_set = pd.read_csv("datasets/validation_set_300.csv", index_col=0)

X_train = training_set.drop('label', axis=1)
y_train = training_set['label']
X_val = validation_set.drop('label', axis=1)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)

#Change the prior data here
prior_data = { "d": X_train.shape[1],
               "scale_global": (10 / (X_train.shape[1] - 10)) / np.sqrt(X_train.shape[0]),
               "N": X_train.shape[0],
               "X" : X_train} 

#Log the start time before building and sampling
start_time = datetime.datetime.now()
logging.info(f'Start time prior: {start_time}')

# logging.getLogger().setLevel(logging.CRITICAL)
#Change the prior name
prior = stan.build(stancodes.rhs_prior, data=prior_data, random_seed=10)  
prior_samples = prior.sample(num_chains=4, num_samples=5000)

logging.getLogger().setLevel(logging.INFO)

prior_df = prior_samples.to_frame()
#prior_df.to_csv("results/bayesian_results/logistic_prior_samples.csv")
#prior_df.to_csv("results/bayesian_results/lasso_prior_samples.csv")
#prior_df.to_csv("results/bayesian_results/esnet_prior_samples.csv")
#prior_df.to_csv("results/bayesian_results/rhs_prior_samples.csv")
#prior_df.to_csv("results/bayesian_results/rhs_prior_samples_1000.csv")
#prior_df.to_csv("results/bayesian_results/rhs_prior_samples_500.csv")
prior_df.to_csv("results/bayesian_results/rhs_prior_samples_300.csv")

#Log the end time after sampling
end_time = datetime.datetime.now()
logging.info(f'End time prior: {end_time}')

# Calculate time difference in minutes
time_diff = end_time - start_time
time_diff_minutes = time_diff.total_seconds() / 60
logging.info(f'Prior run Time: {time_diff_minutes} minutes')


#Change the posterior data here
posterior_data = {"N": X_train.shape[0], 
                  "d": X_train.shape[1],
                  "X" : X_train,
                  "y": y_train,
                  "N_val": X_val.shape[0],
                  "X_val": X_val,
                  "scale_global": (10 / (X_train.shape[1] - 10)) / np.sqrt(X_train.shape[0]),
                  "nu_global":1,
                  "nu_local":1,
                  "slab_scale":2,
                  "slab_df":4}

start_time_post = datetime.datetime.now()
logging.info(f'Start time posterior: {start_time_post}')

logging.getLogger().setLevel(logging.CRITICAL)
#Change the prior name
posterior = stan.build(stancodes.rhs_posterior, data=posterior_data, random_seed=10)  
posterior_samples = posterior.sample(num_chains=4, num_samples=5000, delta = 0.99, save_warmup = False)
logging.getLogger().setLevel(logging.INFO)

posterior_df = posterior_samples.to_frame()
#posterior_df.to_csv("results/bayesian_results/logistic_posterior_samples.csv")
#posterior_df.to_csv("results/bayesian_results/lasso_posterior_samples.csv")
#posterior_df.to_csv("results/bayesian_results/esnet_posterior_samples.csv")
#posterior_df.to_csv("results/bayesian_results/rhs_posterior_samples.csv")
#posterior_df.to_csv("results/bayesian_results/rhs_posterior_samples_1000.csv")
#posterior_df.to_csv("results/bayesian_results/rhs_posterior_samples_500.csv")
posterior_df.to_csv("results/bayesian_results/rhs_posterior_samples_300.csv")

end_time_post = datetime.datetime.now()
logging.info(f'End time posterior: {end_time_post}')

# Calculate time difference in minutes
time_diff_post = end_time_post - start_time_post
time_diff_minutes_post = time_diff_post.total_seconds() / 60
logging.info(f'Posterior run time: {time_diff_minutes_post} minutes')
