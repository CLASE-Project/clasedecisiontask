#### Environment setup #### 

# Working directory needs to be set to `parameter_recovery` directory of the repository.

source('./choice_probability.R');
source('./binary_choice_from_probability.R')
source('./negLLprospect.R');
source('./check_trial_analysis.R');
library('ggplot2')
library('doParallel')
library('foreach')
library('numDeriv')
eps = .Machine$double.eps;

iterations_per_estimation = 100; # how many times to perform the maximum likelihood estimation procedure on a given choiceset, for robustness.

#### Path to the Data ####

# Configure this for the system it's being used on
datapath = '~/Documents/Dropbox/Academics/Research/CLASE Project 2021/data/Behavioral data files/'; # DATA PATH FOR PSH
fn = dir(datapath,pattern = glob2rx('clase*csv'),full.names = T);
number_of_subjects = length(fn)

#### Get the data ####
data = as.data.frame(matrix(data = NA, nrow = 0, ncol = 14))

for(i in 1:number_of_subjects){
  tmpf = read.csv(fn[i]);
  data = rbind(data,tmpf)
}

subjIDs = unique(data$subjID);

likelihood_correct_check_trial = array(dim = c(number_of_subjects,1));

#### Initialize estimation procedure ####
set.seed(Sys.time()); # Estimation procedure is sensitive to starting values

number_of_parameters = 3;

initial_values_lowerbound = c(0.6, 0.2, 25); # for rho, lambda, and mu
initial_values_upperbound = c(1.4, 5, 100) - initial_values_lowerbound; # for rho, lambda, and mu

estimation_lowerbound = c(eps,eps,eps); # lower bound of parameter values is machine precision above zero
estimation_upperbound = c(2, 8, 300); # sensible/probable upper bounds on parameter values

# Create placeholders for the final estimates of the parameters, errors, and NLLs
estimated_parameters = array(dim = c(number_of_subjects, number_of_parameters),
                             dimnames = list(c(), c('rho','lambda','mu')));
estimated_parameter_errors = array(dim = c(number_of_subjects, number_of_parameters),
                                   dimnames = list(c(), c('rho','lambda','mu')));
estimated_nlls = array(dim = c(number_of_subjects,1));
mean_choice_likelihood = array(dim = c(number_of_subjects,1));
  
# # Initialize the progress bar
# progress_bar = txtProgressBar(min = 0, max = number_of_subjects, style = 3)

# Set up the parallelization
n.cores <- parallel::detectCores() - 1; # Use 1 less than the full number of cores.
my.cluster <- parallel::makeCluster(
  n.cores,
  type = "FORK"
)
doParallel::registerDoParallel(cl = my.cluster)

estimation_start_time = proc.time()[[3]]; # Start the clock

for (subject in 1:number_of_subjects){
  
  subject_data = data[data$subjID == subjIDs[subject],];
  
  finite_ind = is.finite(subject_data$choice);
  tmpdata <- subject_data[finite_ind,]; # remove rows with NAs (missed choices) from estimation
  choiceset = tmpdata[, 1:3];
  choices = tmpdata$choice;
  
  likelihood_correct_check_trial[subject] = check_trial_analysis(tmpdata);
  
  print(sprintf('Subject %03i missed %i trials; had a %.2f likelihood of correctly answering check trials.',subjIDs[subject],0+sum((data$subjID == subjIDs[subject]) & is.na(data$choice)),likelihood_correct_check_trial[subject]),quote = F)

  # Placeholders for all the iterations of estimation we're doing
  all_estimates = matrix(nrow = iterations_per_estimation, ncol = number_of_parameters);
  all_nlls = matrix(nrow = iterations_per_estimation, ncol = 1);
  all_hessians = array(dim = c(iterations_per_estimation, number_of_parameters, number_of_parameters))
  
  # The parallelized loop
  alloutput <- foreach(iteration=1:iterations_per_estimation, .combine=rbind) %dopar% {
    initial_values = runif(3)*initial_values_upperbound + initial_values_lowerbound; # create random initial values

    # The estimation itself
    output <- optim(initial_values, negLLprospect, choiceset = choiceset, choices = choices,
                    method = "L-BFGS-B", lower = estimation_lowerbound, upper = estimation_upperbound, hessian = TRUE);
    
    c(output$par,output$value); # the things (parameter values & NLL) to save/combine across parallel estimations
  }
  
  all_estimates = alloutput[,1:3];
  all_nlls = alloutput[,4];
  
  best_nll_index = which.min(all_nlls); # identify the single best estimation
  
  # Save out the parameters & NLLs from the single best estimation
  estimated_parameters[subject,] = all_estimates[best_nll_index,];
  estimated_nlls[subject] = all_nlls[best_nll_index];
  
  # Calculate & store the mean choice likelihood given our best estimates
  choiceP = choice_probability(choiceset,estimated_parameters[subject,]);
  mean_choice_likelihood[subject] = mean(choices * choiceP + (1 - choices) * (1-choiceP));
  
  # Calculate the hessian at those parameter values & save out
  best_hessian = hessian(func=negLLprospect, x = all_estimates[best_nll_index,], choiceset = choiceset, choices = choices)
  estimated_parameter_errors[subject,] = sqrt(diag(solve(best_hessian)));
  
  binary_gainloss_plot = ggplot(data = tmpdata[tmpdata$riskyloss < 0,], aes(x = riskygain, y = riskyloss)) + 
    geom_point(aes(color = as.logical(tmpdata$choice[tmpdata$riskyloss < 0]), alpha = 0.7, size = 3)) + 
    scale_color_manual(values = c('#ff0000','#00ff44'), guide=FALSE) + 
    theme_linedraw() + theme(legend.position = "none", aspect.ratio=1) + 
    ggtitle(sprintf('Gain-Loss Decisions: CLASE%03g',subjIDs[subject]));
  print(binary_gainloss_plot);
  ggsave(sprintf('gainloss_CLASE%03g.png',subjIDs[subject]),height=4.2,width=4.6,dpi=300);
  
  binary_gainonly_plot = ggplot(data = tmpdata[tmpdata$riskyloss >= 0,], aes(x = riskygain, y = certainalternative)) + 
    geom_point(aes(color = as.logical(tmpdata$choice[tmpdata$riskyloss >= 0]), alpha = 0.7, size = 3)) + 
    scale_color_manual(values = c('#ff0000','#00ff44'),guide=FALSE) + 
    theme_linedraw() + theme(legend.position = "none", aspect.ratio=1) + 
    ggtitle(sprintf('Gain-Only Decisions: CLASE%03g',subjIDs[subject]));
  print(binary_gainonly_plot);
  ggsave(sprintf('gainonly_CLASE%03g.png',subjIDs[subject]),height=4.2,width=4.6,dpi=300);
}

parallel::stopCluster(cl = my.cluster)
estimation_time_elapsed = (proc.time()[[3]] - estimation_start_time)/60/60; # time elapsed in HOURS

# save(list = c('recovered_parameters','recovered_nlls','recovered_parameter_errors',
#               'simulated_choice_data','truevals_rho','truevals_lambda','truevals_mu',
#               'truevals','number_of_subjects','simulations_per_subject','iterations_per_estimation',
#               'choiceset'), file = 'parameter_recovery_output.RData')

