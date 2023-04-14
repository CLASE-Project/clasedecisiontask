choice_probability <- function(choiceset, parameters) {
# A function to calculate the probability of taking a risky option
# using a prospect theory model.
# Assumes parameters are [rho, lambda, mu] as used in S-H 2009, 2013, 2015, etc.
# Assumes choiceset has columns riskygain, riskyloss, and certainalternative.
# Creates binary choices, with True/1 = risky, False/0 = safe.
#
# Peter Sokol-Hessner
# July 2021

  # extract  parameters
  rho = as.double(parameters[1]); # risk attitudes
  lambda = as.double(parameters[2]); # loss aversion
  mu = as.double(parameters[3]); # choice consistency
  
  # calculate utility of the two options, making as few assumptions as possible about the +/- sign of the values
  utility_risky_1 = (choiceset$riskygain >= 0) * abs(choiceset$riskygain)^rho + 
    (choiceset$riskygain < 0) * -lambda * abs(choiceset$riskygain)^rho;
  
  utility_risky_2 = (choiceset$riskyloss >= 0) * abs(choiceset$riskyloss)^rho + 
    (choiceset$riskyloss < 0) * -lambda * abs(choiceset$riskyloss)^rho;
  
  utility_risky_option = 0.5 * utility_risky_1 + 0.5 * utility_risky_2
  
  utility_safe_option = (choiceset$certainalternative >= 0) * abs(choiceset$certainalternative)^rho + 
    (choiceset$certainalternative < 0) * -lambda * abs(choiceset$certainalternative)^rho;
  
  # normalize values using this term
  div <- max(choiceset)^rho; # decorrelates rho & mu
  
  # calculate the probability of selecting the risky option
  p = 1/(1+exp(-mu/div*(utility_risky_option - utility_safe_option)));
  
  return(p)
  }
