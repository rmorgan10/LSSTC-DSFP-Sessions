data {
  int nobs;
  vector[nobs] xobs;
  vector[nobs] yobs;
  vector[nobs] sigma_y;
  vector[nobs] sigma_x;
}

parameters {
  real m;
  real b;
  real<lower=0> sigma_int;
  vector[nobs] y_true;
  vector[nobs] x_true;
}

transformed parameters {
   vector[nobs] mu_y_true = m*xobs + b;
}

model {
  /* Flat in m and b. */

  y_true ~ normal(mu_y_true, sigma_int);
  yobs ~ normal(y_true, sigma_y);
  xobs ~ normal(x_true, sigma_x);
}
