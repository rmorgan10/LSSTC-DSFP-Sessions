data {
  int nobs;
  vector[nobs] zobs;
  vector[nobs] zobs_squared;
  vector[nobs] mbobs;
  vector[nobs] dmbobs;
}

parameters {
  real<lower=0,upper=1> Om;
  real<lower=0> H0;
  real w;
}

transformed parameters {
  vector[nobs] mb_true; 
  
  for (n in 1:nobs){
  real value = 5 * log_10( (299792 / H0) * (zobs[n] + zobs_squared[n] * ((3 - 10 * w + 3 * w^2 + 10 * w * Om + 6 * w^2 * Om - 9 * w^2 * Om^2) / (4*(1-3 * w + 3 * w * Om))) / (1 + zobs[n] * (1 - 2*w - 3 * w ^ 2 + 2 * w * Om + 12 * w^2 * Om -9 * w^2 * Om^2) / (2 * (1 - 3 * w + 3 * w * Om)));
  
  mb_true[n] = value;
  }
  
}

model {
  /* Flat in Om and H0 and w. */

  mbobs ~ normal(mb_true, dmbobs);
}
