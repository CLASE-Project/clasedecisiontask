subject_to_be_graphed = 9;

subject_data = data[data$subjID == subject_to_be_graphed,];
ind = which(subjIDs == subject_to_be_graphed);
subject = ind;


graphing_data = subject_data[subject_data$ischecktrial==0,];
graphing_data$choice[is.na(graphing_data$choice)] = 2;
# graphing_data = tmpdata[tmpdata$ischecktrial==0,];
graphing_data = graphing_data[order(graphing_data$riskyloss),];

missedtrials = subject_data[!is.finite(subject_data$choice),]
missedtrials = missedtrials[missedtrials$ischecktrial == 0,]

gainonly_xvals = c(0,40);
gainonly_lines = data.frame('xvals' = gainonly_xvals);
gainonly_lines$yvals_fit = (.5 * gainonly_xvals^estimated_parameters[ind,'rho'])^(1/estimated_parameters[ind,'rho'])
gainonly_lines$yvals_neutral = c(0,20);
# gainonly_lines$yvals_riskaverse09 = (.5 * gainonly_xvals^.9)^(1/.9);
# gainonly_lines$yvals_riskaverse075 = (.5 * gainonly_xvals^.75)^(1/.75);
# gainonly_lines$yvals_riskseeking11 = (.5 * gainonly_xvals^1.1)^(1/1.1);

gainloss_xvals = c(0,13);
gainloss_lines = data.frame('xvals' = gainloss_xvals);
gainloss_lines$yvals_fit = -((.5 * gainloss_xvals^estimated_parameters[ind,'rho'])/(0.5 * estimated_parameters[ind,'lambda']))^(1/estimated_parameters[ind,'rho'])
gainloss_lines$yvals_neutral = -gainloss_xvals;
# gainloss_lines$yvals_gainseeking = -((.5 * gainloss_xvals^1)/(0.5 * .8)^(1/1));
# gainloss_lines$yvals_lossaverse2 = -((.5 * gainloss_xvals^1)/(0.5 * 2)^(1/1));
# gainloss_lines$yvals_lossaverse5 = -((.5 * gainloss_xvals^1)/(0.5 * 5)^(1/1));

graphing_data = graphing_data[order(graphing_data$riskyloss),];

binary_gainonly_plot = ggplot(data = graphing_data[graphing_data$riskyloss >= 0,], aes(x = riskygain, y = certainalternative)) + 
  # geom_point(shape = 21, color = 'black', aes(fill = as.logical(graphing_data$choice[graphing_data$riskyloss >= 0]), size = 2, stroke = 0.5)) + 
  geom_point(shape = 16, size = 3, aes(color = as.factor(graphing_data$choice[graphing_data$riskyloss >= 0]))) + 
  scale_color_manual(values = alpha(c('#ff0000','#00ff44','#BEBEBE'), 1), guide=FALSE) + 
  theme_linedraw() + theme(legend.position = "none", aspect.ratio=1) +
  ggtitle(sprintf('Gain-Only Decisions: CLASE%03g',subjIDs[subject])) + 
  # geom_point(data = missedtrials[missedtrials$riskyloss >= 0,], 
  #            shape = 21, color = 'black', fill = alpha('gray',0.5), stroke = 0.5, aes(size = 2)) +
  geom_line(data = gainonly_lines, aes(x = xvals, y = yvals_fit), color = 'black', linetype = 'dashed', size = 1) + 
  geom_line(data = gainonly_lines, aes(x = xvals, y = yvals_neutral), color = 'black', size = 1) + 
  # geom_line(data = gainonly_lines, aes(x = xvals, y = yvals_riskaverse09), color = 'red4', size = 1) +
  # geom_line(data = gainonly_lines, aes(x = xvals, y = yvals_riskaverse075), color = 'red', size = 1) +
  # geom_line(data = gainonly_lines, aes(x = xvals, y = yvals_riskseeking11), color = 'steelblue1', size = 1) + 
  coord_cartesian(xlim = c(0, 33), ylim = c(0, 14.4), expand = F);
print(binary_gainonly_plot);
ggsave(sprintf('gainonly_CLASE%03g_forgrant.eps',subjIDs[subject]),height=4.2,width=4.6,dpi=1200);

binary_gainloss_plot = ggplot(data = graphing_data[graphing_data$riskyloss < 0,], aes(x = riskygain, y = riskyloss)) + 
  # geom_point(shape = 21, color = 'black', size = 3, aes(fill = as.factor(graphing_data$choice[graphing_data$riskyloss < 0]), stroke = 0.4)) + 
  geom_point(shape = 16, size = 3, aes(color = as.factor(graphing_data$choice[graphing_data$riskyloss < 0]))) + 
  scale_color_manual(values = alpha(c('#ff0000','#00ff44','#BEBEBE'), 1), guide=FALSE) + 
  theme_linedraw() + theme(legend.position = "none", aspect.ratio=1) + 
  ggtitle(sprintf('Gain-Loss Decisions: CLASE%03g',subjIDs[subject])) + 
  # geom_point(data = missedtrials[missedtrials$riskyloss < 0,], 
  #            shape = 21, color = 'black', fill = alpha('gray',1), stroke = 0.4, size = 3) + 
  geom_line(data = gainloss_lines, aes(x = xvals, y = yvals_fit), color = 'black', linetype = 'dashed', size = 1) + 
  geom_line(data = gainloss_lines, aes(x = xvals, y = yvals_neutral), color = 'black', size = 1) + 
  # geom_line(data = gainloss_lines, aes(x = xvals, y = yvals_gainseeking), color = 'steelblue1', size = 1) + 
  # geom_line(data = gainloss_lines, aes(x = xvals, y = yvals_lossaverse2), color = 'red', size = 1) + 
  # geom_line(data = gainloss_lines, aes(x = xvals, y = yvals_lossaverse5), color = 'red4', size = 1) + 
  coord_cartesian(xlim = c(0, 12.5), ylim = c(-21, 0), expand = F);
print(binary_gainloss_plot);
ggsave(sprintf('gainloss_CLASE%03g_forgrant.eps',subjIDs[subject]),height=4.2,width=4.6,dpi=1200);


