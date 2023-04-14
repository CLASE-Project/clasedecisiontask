check_trial_analysis <- function(data) {
  # A function to analyze check trials to determine how many participants missed.
  #
  # Assumes data has columns riskygain, riskyloss, certainalternative, choice, and ischecktrial
  #
  # Peter Sokol-Hessner
  # December 2021
  
  # Isolate check trials
  check_trial_data = data[data$ischecktrial==1,];
  
  # Calculate the correct answer (using EV formulation, though this doesn't matter as long as $ are valued positively)
  correct_answer = (check_trial_data$riskygain*.5 + check_trial_data$riskyloss*.5) > check_trial_data$certainalternative;
  
  # Calculate this participant's likelihood of getting a check trial correct
  likelihood_correct_check_trial = mean(correct_answer==check_trial_data$choice)
  
  return(likelihood_correct_check_trial)
}
