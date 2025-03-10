'''
These codes has been adapted from Piironen & Vehtari (2017), Sara van Erp & Daniel L. Oberski & Joris Mulder (2019)

These code contains the implementation of the logistic regression with the following priors with prior & posterior predictive checks
Weakly informative normal priors 
Lasso prior
Elastic net prior
Regularised horseshoe

### The following codes are for the setting the priors  
### Note: These priors are ran with the HMC NUTS sampler as there were complex hierarchical priors
'''

# Weakly informative normal prior 
winp_prior = """
data {
  int<lower=0> d;
  int<lower=0> N;
  matrix[N, d] X;
}

parameters {
  real alpha;
  vector[d] beta;
}

model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
}
generated quantities {  
  vector[N] y_pred;
  for (n in 1:N) {
    y_pred[n] = inv_logit(alpha + dot_product(X[n], beta));
  }
}
"""

# Bayesian lasso prior
lasso_prior = """
data {
  int<lower=0> d;
  int<lower=0> N;
  matrix[N, d] X;
}

parameters {
  real alpha;
  real<lower=0> lambda;
  vector[d] beta;
}

model {
  alpha ~ normal(0,5);
  lambda ~ cauchy(0,1);
  beta ~ double_exponential (0, 1 / lambda);
}

generated quantities {  
  vector[N] y_pred;
  for (n in 1:N) {
    y_pred[n] = inv_logit(alpha + dot_product(X[n], beta));
  }
}
"""

# Bayesian Elastic net prior
esnet_prior = """
data {
  int<lower=0> N;
  int<lower=0> d;
  matrix[N, d] X;
}

parameters {
  real alpha;
  real<lower=0> lambda1;
  real<lower=0> lambda2;
  vector<lower=1>[d] tau; 
  vector[d] beta_raw;
}

transformed parameters{
  vector[d] beta;
  for (k in 1:d) {
    beta[k] = sqrt(((tau[k]-1)/(lambda2*tau[k]))) * beta_raw[k];
  }
}

model {
  alpha ~ normal(0, 5);
  beta_raw ~ normal(0, 1);
  lambda1 ~ cauchy(0,1);
  lambda2 ~ cauchy(0,1);
  tau ~ gamma(0.5, (8*lambda2)/(lambda1^2));
}

generated quantities {  
  vector[N] y_pred;
  for (n in 1:N) {
    y_pred[n] = inv_logit(alpha + dot_product(X[n], beta));
  }
}
"""

# Bayesian regularised horseshoe prior
rhs_prior = """
data {
    int<lower=0> d; 
    real<lower=0> scale_global; 
    int<lower=0> N;
    matrix[N, d] X;
}
parameters {
    real alpha;
    vector[d] z;
    real<lower=0> aux1_global;
    real<lower=0> aux2_global;
    vector<lower=0>[d] aux1_local;
    vector<lower=0>[d] aux2_local;
    real<lower=0> caux ;
}
transformed parameters {
    real<lower=0> tau ;
    vector<lower=0>[d] lambda; 
    vector< lower=0>[d] lambda_tilde; 
    real<lower=0> c;
    vector[d] beta;
    lambda = aux1_local .* sqrt(aux2_local);
    tau = aux1_global * sqrt(aux2_global) * scale_global;
    c = 2 * sqrt (caux);
    lambda_tilde = sqrt( c^2 * square(lambda) ./ (c^2 + tau^2 * square(lambda)) );
    beta = z .* lambda_tilde * tau ;
}
model {
    z ~ normal(0,1);
    aux1_local ~ normal(0,1);
    aux2_local ~ inv_gamma(0.5*1, 0.5*1);
    aux1_global ~ normal(0,1);
    aux2_global ~ inv_gamma(0.5*1 , 0.5*1 );
    caux ~ inv_gamma(0.5*4 , 0.5*4 );
    alpha ~ normal(0,5);
}
generated quantities {  
  vector[N] y_pred;
  for (n in 1:N) {
    y_pred[n] = inv_logit(alpha + dot_product(X[n], beta));
  }
}
"""

'''
These code contains the implementation of the logistic regression with the following priors for posterior approximation
Weakly informative normal priors 
Lasso
Elastic net
Regularised horseshoe


### The following codes are for the posterior approximation
### Note: These priors are run with the HMC NUTS sampler as there were complex hierarchical priors

In the generated quantities we calculate for both the training and validation sets for easier computation of posterior distribution 
for both the training and validation sets
'''

# Weakly informative normal prior -> posterior 
winp_posterior = """
data {
  int<lower=0> N;
  int<lower=0> d;
  matrix[N, d] X;
  int<lower=0, upper=1> y[N];
  int<lower=0> N_val;
  matrix[N_val, d] X_val;
}

parameters {
  real alpha;
  vector[d] beta;
}

model {
  alpha ~ normal(0,1);
  beta ~ normal(0,1);
  
  y ~ bernoulli_logit(alpha + X * beta);
}

generated quantities {  
  vector[N] y_train_pred;
  vector[N_val] y_val_pred;

  for (n in 1:N) {
    y_train_pred[n] = inv_logit(alpha + dot_product(X[n], beta));
  }
  
  for (n in 1:N_val) {
    y_val_pred[n] = inv_logit(alpha + dot_product(X_val[n], beta));
  }
}
"""

# Bayesian lasso prior -> posterior
lasso_posterior = """
data {
  int<lower=0> N;
  int<lower=0> d;
  matrix[N, d] X;
  int<lower=0, upper=1> y[N];
  int<lower=0> N_val;
  matrix[N_val, d] X_val;
}

parameters {
  real alpha;
  real<lower=0> lambda;
  vector[d] beta;
}

model {
  alpha ~ normal(0, 5);
  lambda ~ cauchy(0,1);
  beta ~ double_exponential (0, 1 / lambda);
  
  y ~ bernoulli_logit(alpha + X * beta);
}

generated quantities {  
  vector[N] y_train_pred;
  vector[N_val] y_val_pred;

  for (n in 1:N) {
    y_train_pred[n] = inv_logit(alpha + dot_product(X[n], beta));
  }
  
  for (n in 1:N_val) {
    y_val_pred[n] = inv_logit(alpha + dot_product(X_val[n], beta));
  }
}
"""

# Bayesian Elastic net prior -> posterior
esnet_posterior = """
data {
  int<lower=0> N;
  int<lower=0> d;
  matrix[N, d] X;
  int<lower=0, upper=1> y[N];
  int<lower=0> N_val;
  matrix[N_val, d] X_val;
}

parameters {
  real alpha;
  real<lower=0> lambda1;
  real<lower=0> lambda2;
  vector<lower=1>[d] tau; 
  vector[d] beta_raw;
}

transformed parameters{
  vector[d] beta;
  for (k in 1:d) {
    beta[k] = sqrt(((tau[k]-1)/(lambda2*tau[k]))) * beta_raw[k];
  }
}

model {
  alpha ~ normal(0, 5);
  beta_raw ~ normal(0, 1);
  lambda1 ~ cauchy(0,1);
  lambda2 ~ cauchy(0,1);
  tau ~ gamma(0.5, (8*lambda2)/(lambda1^2));
  
  y ~ bernoulli_logit(alpha + X * beta);
}

generated quantities {  
  vector[N] y_train_pred;
  vector[N_val] y_val_pred;

  for (n in 1:N) {
    y_train_pred[n] = inv_logit(alpha + dot_product(X[n], beta));
  }
  
  for (n in 1:N_val) {
    y_val_pred[n] = inv_logit(alpha + dot_product(X_val[n], beta));
  }
}
"""

# Bayesian regularised horseshoe prior -> posterior
rhs_posterior = """
data {
    int<lower=0> N; 
    int<lower=0> d; 
    int<lower=0,upper =1> y[N]; 
    matrix[N ,d] X;
    int<lower=0> N_val;
    matrix[N_val,d] X_val;
    real<lower=0> scale_global; 
    real<lower=1> nu_global; 
    real<lower=1> nu_local; 
    real<lower=0> slab_scale; 
    real<lower=0> slab_df; 
}
parameters {
    real alpha;
    vector[d] z;
    real<lower=0> aux1_global;
    real<lower=0> aux2_global;
    vector<lower=0>[d] aux1_local;
    vector<lower=0>[d] aux2_local;
    real<lower=0> caux ;
}
transformed parameters {
    real<lower=0> tau ;
    vector<lower=0>[d] lambda; 
    vector< lower=0>[d] lambda_tilde; 
    real<lower=0> c;
    vector[d] beta;
    lambda = aux1_local .* sqrt(aux2_local);
    tau = aux1_global * sqrt(aux2_global) * scale_global;
    c = slab_scale * sqrt (caux);
    lambda_tilde = sqrt( c^2 * square(lambda) ./ (c^2 + tau^2 * square(lambda)) );
    beta = z .* lambda_tilde * tau ;
}
model {
    z ~ normal(0,1);
    aux1_local ~ normal(0,1);
    aux2_local ~ inv_gamma(0.5*nu_local, 0.5*nu_local);
    aux1_global ~ normal(0,1);
    aux2_global ~ inv_gamma(0.5*nu_global , 0.5*nu_global );
    caux ~ inv_gamma(0.5*slab_df , 0.5*slab_df );
    alpha ~ normal(0,5);

    y ~ bernoulli(inv_logit(X * beta + alpha));
}

generated quantities {  
  vector[N] y_train_pred;
  vector[N_val] y_val_pred;

  for (i in 1:N) {
    y_train_pred[i] = inv_logit(alpha + dot_product(X[i], beta));
  }
  
  for (i in 1:N_val) {
    y_val_pred[i] = inv_logit(alpha + dot_product(X_val[i], beta));
  }
}
"""
