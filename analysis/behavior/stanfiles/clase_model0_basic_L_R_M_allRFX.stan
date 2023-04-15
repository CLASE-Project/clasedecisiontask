data {
  int N; // number of trials (across participants)
  int nsubj; // number of participants
  int choices[N]; // choice vector
  real gain[N]; // risky gain vector
  real loss[N]; // risky loss vector
  real safe[N]; // safe vector
  int ind[N]; // subject index
}

parameters {
  real meanRho;
  real<lower=0> sdRho;
  real meanMu;
  real<lower=0> sdMu;
  real meanLambda;
  real<lower=0> sdLambda;

  real r[nsubj]; // random effects for rho
  real m[nsubj]; // random effects for mu
  real l[nsubj]; // random effects for lambda
}

transformed parameters {
  real rtmp[nsubj];
  real ltmp[nsubj];
  real mtmp[nsubj];
  
  rtmp = exp(r); // makes "rtmp" values strictly positive
  ltmp = exp(l);
  mtmp = exp(m);
}

model {
  real div;
  real p[N];
  real total_sum[N];
  real gam1u;
  real gam2u;
  real certu;
  
  //Priors
  meanRho ~ normal(0,30);
  sdRho ~ cauchy(0,2.5);
  meanMu ~ normal(0,30);
  sdMu ~ cauchy(0,2.5);
  meanLambda ~ normal(0,30);
  sdLambda ~ cauchy(0,2.5);

  // //Hierarchy
  r ~ normal(meanRho, sdRho);
  m ~ normal(meanMu, sdMu);
  l ~ normal(meanLambda, sdLambda);

  for (t in 1:N) {
    div = 32^rtmp[ind[t]];
    // Model with M, L, R, DB
    
    if (gain[t] < 0)
      gam1u = -0.5 * ltmp[ind[t]] * pow(-gain[t],rtmp[ind[t]]);
    else
      gam1u = 0.5 * pow(gain[t],rtmp[ind[t]]);
    
    if (loss[t] < 0)
      gam2u = -0.5 * ltmp[ind[t]] * pow(-loss[t],rtmp[ind[t]]);
    else
      gam2u = 0.5 * pow(loss[t],rtmp[ind[t]]);
    
    if (safe[t] < 0)
      certu = -ltmp[ind[t]] * pow(-safe[t],rtmp[ind[t]]);
    else
      certu = pow(safe[t],rtmp[ind[t]]);
    
    total_sum[t] = mtmp[ind[t]] / div * (gam1u + gam2u - certu);
  }
  
  p = inv_logit(total_sum);
  
  choices ~ bernoulli(p);
}
