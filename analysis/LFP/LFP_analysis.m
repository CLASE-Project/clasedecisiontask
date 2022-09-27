% CLASE Analysis: LFP data
%

clear all;

path_to_data = '/Users/sokolhessner/Documents/Dropbox/Academics/Research/CLASE Project 2021/data/LFP_data';

cd(path_to_data)

fn = dir('clase*_LFPprocessByTrial.mat');

number_of_subjects = length(fn);

for s = 1:number_of_subjects
    load(fn(s).name)
end

% ave_uPv = average power in microvolts [for a specific frequency band]:
%   data matrix 135 x 7 x 22 = trial x band x channel
%       135 trials
%       7 bands (delta, theta, alpha, low beta, high beta, low gamma, high gamma)
%       22 channels (all contacts in amygdala for 006)

% peak_uPv = peak power in microvolt for specific frequency band.

frequency_bands = {'delta', 'theta', 'alpha', 'low beta', 'high beta', 'low gamma', 'high gamma'};

av_amyg = mean(outDATA.ave_uPv, 3); % averages across amyg. contacts & trials

av_normalized_within_band = av_amyg;

for b = 1:size(outDATA.ave_uPv,2)
    av_normalized_within_band(:,b) = av_normalized_within_band(:,b)/mean(av_normalized_within_band(:,b));
end

fN = 0;

fN = fN + 1; figure(fN);
subplot(1,2,1)
imagesc(av_amyg);
xticklabels(frequency_bands)
title('Average Amygdala power, averaged across channels')

subplot(1,2,2)
imagesc(av_normalized_within_band)
xticklabels(frequency_bands)
title('Average Amygdala power, averaged across channels, normalized within frequency bands')
