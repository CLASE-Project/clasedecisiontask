#### Environment setup #### 

# Working directory needs to be set to `parameter_recovery` directory of the repository.
setwd('/Users/sokolhessner/Documents/gitrepos/clasedecisiontask/parameter_recovery/')

source('./choice_probability.R');
source('./binary_choice_from_probability.R')
source('./negLLprospect.R');
source('./check_trial_analysis.R');
library('ggplot2')
library('doParallel')
library('foreach')
library('numDeriv')
eps = .Machine$double.eps;

iterations_per_estimation = 200; # how many times to perform the maximum likelihood estimation procedure on a given choiceset, for robustness.

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
  
  number_check_trials = sum(tmpdata$ischecktrial);
  
  likelihood_correct_check_trial[subject] = check_trial_analysis(tmpdata);
  
  cat(sprintf('Subject %03i missed %i trials; had a %.2f likelihood of correctly answering %g check trials.\n',subjIDs[subject],0+sum((data$subjID == subjIDs[subject]) & is.na(data$choice)), likelihood_correct_check_trial[subject], number_check_trials))

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
  fig_name = sprintf('gainloss_CLASE%03g.png',subjIDs[subject]);
  if (!file.exists(fig_name)){
    ggsave(fig_name,height=4.2,width=4.6,dpi=300);
  }
  
  binary_gainonly_plot = ggplot(data = tmpdata[tmpdata$riskyloss >= 0,], aes(x = riskygain, y = certainalternative)) + 
    geom_point(aes(color = as.logical(tmpdata$choice[tmpdata$riskyloss >= 0]), alpha = 0.7, size = 3)) + 
    scale_color_manual(values = c('#ff0000','#00ff44'),guide=FALSE) + 
    theme_linedraw() + theme(legend.position = "none", aspect.ratio=1) + 
    ggtitle(sprintf('Gain-Only Decisions: CLASE%03g',subjIDs[subject]));
  print(binary_gainonly_plot);
  fig_name = sprintf('gainonly_CLASE%03g.png',subjIDs[subject]);
  if (!file.exists(fig_name)){
    ggsave(fig_name,height=4.2,width=4.6,dpi=300);
  }
}

parallel::stopCluster(cl = my.cluster)
estimation_time_elapsed = (proc.time()[[3]] - estimation_start_time)/60; # time elapsed in MINUTES

cat(sprintf('Estimation finished. Took %.1f minutes.\n', estimation_time_elapsed));

to_exclude = c(6, 20);
# CLASE 006 dropped (boundary estimates; poor performance on check trials)
# CLASE 020 dropped (boundary estimate for L; so-so performance on check trials)
keepsubj = !(subjIDs %in% to_exclude)

cat(sprintf('Total of %g participants kept (out of %g collected).\n', sum(keepsubj), length(keepsubj)))

print(estimated_parameters[keepsubj,])

cat(sprintf('Mean choice likelihood = %.2f', mean(mean_choice_likelihood[keepsubj])))

df_foroutput = cbind(subjIDs[keepsubj],estimated_parameters[keepsubj,],estimated_parameter_errors[keepsubj,])
colnames(df_foroutput) <- c('subjectIDs','rho','lambda','mu','rhoSE','lambdaSE','muSE')
write.csv(df_foroutput,file = sprintf('estimation_results_%s.csv',format(Sys.Date(), format="%Y%m%d")), row.names = F)

number_of_subjects_kept = sum(keepsubj);

par(mfrow = c(1,3))
plot(array(data = 1, dim = c(number_of_subjects_kept,1)),estimated_parameters[keepsubj,'lambda'],
     ylab = 'Loss aversion coefficient (lambda)', xlab = '', xaxt = 'n', ylim = c(0,6), col = 'red', cex = 3)
lines(c(0,2), c(1,1), lty = 'dashed')
plot(array(data = 1, dim = c(number_of_subjects_kept,1)),estimated_parameters[keepsubj,'rho'],
     ylab = 'Risk attitudes (rho)', xlab = '', xaxt = 'n', ylim = c(0,2), col = 'green', cex = 3)
lines(c(0,2), c(1,1), lty = 'dashed')
plot(array(data = 1, dim = c(number_of_subjects_kept,1)),estimated_parameters[keepsubj,'mu'],
     ylab = 'Choice consistency (mu)', xlab = '', xaxt = 'n', ylim = c(0,60), col = 'blue', cex = 3)
mtext(paste0('Estimates for ', number_of_subjects_kept, ' subjects'),side = 3,line = - 2,outer = TRUE)
par(mfrow = c(1,1))

# 
# original_estimated_parameters = estimated_parameters; 
# original_estimated_parameter_errors = estimated_parameter_errors;
# 
# ind = rank(estimated_parameters[,'lambda'])
# 
# estimated_parameters = estimated_parameters[ind,];
# estimated_parameter_errors = estimated_parameter_errors[ind,];

pdf(file="behavioral_estimates.pdf", width = 5, height = 2.8)

layout(matrix(c(1,1,2,3,4,4,5,6),2,4,byrow = T), heights = c(.5,2))
# layout.show(6)

# Plot densities


# Plot estimates
barplot_lambda <- barplot(horiz = T, estimated_parameters[keepsubj,'lambda'], 
              col = rgb(1,.45,.2), xlim = c(0,5), xlab = 'Loss aversion (lambda)')
axis(side = 2, at = c(-1,10))
arrows(y0 = barplot_lambda,
       x0 = estimated_parameters[keepsubj,'lambda'] - estimated_parameter_errors[keepsubj,'lambda'],
       x1 = estimated_parameters[keepsubj,'lambda'] + estimated_parameter_errors[keepsubj,'lambda'],
       length = 0)
axis(side = 2, at = c(-1,10))
lines(x = c(1,1), y = c(0,10), lty = 'dashed')
points(x = mean(estimated_parameters[keepsubj,'lambda']), y = 0, pch = 24, cex = 2, bg = rgb(1,.45,.2))
points(x = 2.22, y = 0, pch = 24, cex = 1, bg = 'black')
points(x = 1.62, y = 0, pch = 24, cex = 1, bg = 'white')

barplot_rho <- barplot(horiz = T, estimated_parameters[keepsubj,'rho'], 
                          col = rgb(1,1,0), xlim = c(0,2), xlab = 'Risk attitudes (rho)')
arrows(y0 = barplot_lambda,
       x0 = estimated_parameters[keepsubj,'rho'] - estimated_parameter_errors[keepsubj,'rho'],
       x1 = estimated_parameters[keepsubj,'rho'] + estimated_parameter_errors[keepsubj,'rho'],
       length = 0)
axis(side = 2, at = c(-1,10))
lines(x = c(1,1), y = c(0,10), lty = 'dashed')
points(x = mean(estimated_parameters[keepsubj,'rho']), y = 0, pch = 24, cex = 2, bg = rgb(1,1,0))
points(x = 0.92, y = 0, pch = 24, cex = 1, bg = 'black')
points(x = 0.88, y = 0, pch = 24, cex = 1, bg = 'white')

barplot_mu <- barplot(horiz = T, estimated_parameters[keepsubj,'mu'], 
                       col = rgb(0,1,1), xlim = c(0,100), xlab = 'Consistency (mu)')
arrows(y0 = barplot_lambda,
       x0 = estimated_parameters[keepsubj,'mu'] - estimated_parameter_errors[keepsubj,'mu'],
       x1 = estimated_parameters[keepsubj,'mu'] + estimated_parameter_errors[keepsubj,'mu'],
       length = 0)
axis(side = 2, at = c(-1,10))
points(x = mean(estimated_parameters[keepsubj,'mu']), y = 0, pch = 24, cex = 2, bg = rgb(0,1,1))
points(x = 25.9, y = 0, pch = 24, cex = 1, bg = 'black')
points(x = 65.0, y = 0, pch = 24, cex = 1, bg = 'white')
# save(sprintf('gainloss_CLASE%03g_forgrant.eps',subjIDs[subject]),height=4.2,width=4.6,dpi=1200);

# mtext(paste0('Estimates for ', number_of_subjects_kept, ' subjects'),side = 3,line = - 2,outer = TRUE)
# par(mfrow = c(1,1))
dev.off()

# save(list = c('recovered_parameters','recovered_nlls','recovered_parameter_errors',
#               'simulated_choice_data','truevals_rho','truevals_lambda','truevals_mu',
#               'truevals','number_of_subjects','simulations_per_subject','iterations_per_estimation',
#               'choiceset'), file = 'parameter_recovery_output.RData')

gain_val = 10;
loss_vals = seq(from = 0, to = -19, by = -.2)

gainloss_vals_diff = gain_val + loss_vals;

p_risky = array(dim = c(number_of_subjects_kept, length(loss_vals)));

keeponly_estimated_parameters = estimated_parameters[keepsubj,];

for (s in 1:number_of_subjects_kept){
  tmprho = keeponly_estimated_parameters[s,'rho'];
  tmplambda = keeponly_estimated_parameters[s,'lambda'];
  tmpmu = keeponly_estimated_parameters[s,'mu'];
  p_risky[s,] = 1/(1 + exp(-tmpmu / (32^tmprho) * (gain_val^tmprho + -tmplambda * abs(loss_vals)^tmprho)));
}

pdf(file="softmaxes.pdf", width = 3, height = 3.5)

plot(gainloss_vals_diff, p_risky[1,], type = 'l', col = rgb(0, 0, 0, .5), lwd = 5,
     yaxt = "n", xaxt = "n")
axis(2, at = c(0, 0.5, 1))
axis(1, at = c(-8, 0, 8), labels = c("-$8", "$0", "$8"))

for (s in 2:number_of_subjects_kept){
  if (s == 4){
    lines(x = gainloss_vals_diff, y = p_risky[s,], col = rgb(0, 0, 0, .9), lwd = 5)
  } else {
    lines(x = gainloss_vals_diff, y = p_risky[s,], col = rgb(0, 0, 0, .5), lwd = 5)
  }
}

dev.off()

