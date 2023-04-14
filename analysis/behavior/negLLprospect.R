negLLprospect <- function(parameters,choiceset,choices) {
# A negative log likelihood function for a prospect-theory estimation.
# Assumes parameters are [rho, lambda, mu] as used in S-H 2009, 2013, 2015, etc.
# Assumes choiceset has columns riskygain, riskyloss, and certainalternative.
# Assumes choices are binary/logical, with 1 = risky, 0 = safe.
#
# Peter Sokol-Hessner
# July 2021

eps = .Machine$double.eps;
source('choice_probability.R')

choiceP = choice_probability(choiceset, parameters);

choiceP[choiceP==1] = 1-eps; # because the log of 0 = -infinity & that breaks things
choiceP[choiceP==0] = eps;

nll <- -sum(choices * log(choiceP) + (1 - choices) * log(1-choiceP));
return(nll)
}