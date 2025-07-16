negLLprospect_lambda1 <- function(parameters,choiceset,choices) {
  # A negative log likelihood function for a prospect-theory estimation.
  # Assumes parameters are [rho, mu] as used in S-H 2009, 2013, 2015, etc.
  # Assumes choiceset has columns riskygain, riskyloss, and certainalternative.
  # Assumes choices are binary/logical, with 1 = risky, 0 = safe.
  #
  # Peter Sokol-Hessner
  # July 2025
  
  source('negLLprospect.R')
  
  full_parameters = c(parameters[1], 1, parameters[2])
  
  nll <- negLLprospect(full_parameters, choiceset, choices)
  return(nll)
}