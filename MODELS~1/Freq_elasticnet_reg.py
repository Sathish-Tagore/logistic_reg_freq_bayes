# Import the necessary libraries needed
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning
import logging
import datetime

logging.basicConfig(filename='runtimeee8.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='a')
logging.info('Model name: Frequentist elasticnet logistic regression')
#Log the start time before building and sampling
start_time = datetime.datetime.now()
logging.info(f'Start time: {start_time}')

# Load the data
training_set = pd.read_csv("datasets/training_set.csv", index_col=0)

X_train = training_set.drop('label', axis=1)
y_train = training_set['label']

param_grid = {"C": loguniform(1e-4, 10), "l1_ratio": uniform(0, 1) }

clf = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, verbose=0)

# Function to train and evaluate the model
def train_and_evaluate(state):
    # Resample
    print(f"Currently loop {state} in progress")
    X_resample, y_resample = resample(X_train, y_train, random_state=state)

    # Perform grid search
    gsc = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=10, n_jobs=-1, n_iter=50, random_state = state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gsc.fit(X_resample, y_resample)

    # Get the best estimator
    best_clf = gsc.best_estimator_

    # Calculate metrics
    y_pred_train = best_clf.predict(X_resample)
    result = {
        'random_state': state,
        'lambda': best_clf.C,
        'l1_ratio':best_clf.l1_ratio,
        'intercept': best_clf.intercept_[0],
        **{f'beta.{i+1}': coef for i, coef in enumerate(best_clf.coef_[0])}
    }
    print(f"Loop {state} completed")
    return result
# Run the loop in parallel
results = Parallel(n_jobs=-1, backend='loky')(delayed(train_and_evaluate)(state) for state in range(500))

# Convert the results to a DataFrame and save to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("results/frequentist_results/Frequentist_elasticnet.csv", index=False)

end_time = datetime.datetime.now()
logging.info(f'End time: {end_time}')

# Calculate time difference in minutes
time_diff = end_time - start_time
time_diff_minutes = time_diff.total_seconds() / 60
logging.info(f'Run time: {time_diff_minutes} minutes')