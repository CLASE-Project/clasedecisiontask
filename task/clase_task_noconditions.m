function [subjdata] = clase_task_noconditions(subjID, do_full, do_TTL, do_prac)

% Last update to this f(x) on this machine: 07.19.22 -MLD
%
%
% To run task: clase_task_noconditions('997', 1, 1, 1)
%
% CHANGED pwd on line 47
%
% [subjdata] = clase_task_noconditions(subjID, do_full, do_TTL, do_prac)
% 
% subjID: '###', i.e. '001', '012', etc.
%       '000' is understood to be the test subject number
% do_full:
%       0 = do a shortened version (runs 12 trials in 3 blocks)
%       1 = do the full version (run 125 trials in 5 blocks)
% do_TTL:
%       0 = do not send TTL pulses (for testing on systems w/o TTL
%       functionality)
%       1 = send TTLs
% do_prac:
%       0 = do not do practice trials (for testing)
%       1 = do practice trials
% 
%
% Script for running the choice set once, with no separate conditions
% for the CLASE study. Should be run from within the GitHub repository, so 
% that the choice set can appropriately load.
% 
% Based on the VNI neuroimaging script updated on 6/8/20, by Hayley 
% Brooks, University of Denver
%
% Peter Sokol-Hessner
% University of Denver
% Peter.Sokol-Hessner@du.edu
% GitHub: github.com/psokolhessner

% % For use with actual path on testing laptop
% try
%     homepath = 'C:\Users\inc\Documents\Sokol-Hessner Lab\';
%     cd(homepath);
% catch
    homepath = pwd;
    choiceset_fullpath = 'C:\Users\jatne\OneDrive\Documents\GitHub\clasedecisiontask-main\choice_set\novel_choiceset_creation\clasechoiceset_N135_gainloss_gainonly.csv';
    %choiceset_fullpath = '../choice_set/novel_choiceset_creation/clasechoiceset_N135_gainloss_gainonly.csv';
    cd(homepath);
% end

clc % clear command window

if nargin < 4
    do_prac = 1; % default: do the practice
    if nargin < 3
        do_TTL = 1; % default: do TTL pulses
        if nargin < 2
            do_full = 1; % default: do full-length version
            if nargin < 1 % if number of input arguments for this function is less than one,
                warning('Do not collect data without a subject ID number!')
                warning('If you are collecting data, YOU MUST SPECIFY A SUBJECT ID NUMBER in this format: ''###''');
                subjID = '000'; % set the default subject ID in case we don't include one.
            end
        end
    end
end

fprintf('do_prac set to %i\n',do_prac);
fprintf('do_TTL set to %i\n',do_TTL);
fprintf('do_full set to %i\n',do_full);

if isa(subjID,'double')
    subjID = sprintf('%03d',subjID);
    warning('Specify subject ID numbers with the ''###'' format, e.g. ''001''!')
end

if length(subjID) > 3
    error('Subject ID too long')
elseif length(subjID) < 3
    error('Subject ID too short')
end

fprintf('Proceeding with subject ID of %s\n',subjID);

%------- General Setup -------%

% Initialize rand to a different state each time:
if verLessThan('matlab','7.12')
    rand('twister',sum(100*clock))
    fprintf('Set rng using oldest method\n');
elseif verLessThan('matlab','7.14')
    RandStream.setDefaultStream ...
        (RandStream('mt19937ar','seed',sum(100*clock))); % 'shuffle' makes the seed based on the current time
    fprintf('Set rng using intermediate method\n')
else
    rng('shuffle'); % 'shuffle' makes the seed based on the current time
    fprintf('Set rng using newest method\n')
end

% Initialize TTL setup 
if do_TTL == 1
    [~, device] = Cedrus_TTL();
end  

if do_full
    HideCursor; % remove the cursor so it doesn't just show up on our screen
end

Screen('Preference', 'SkipSyncTests', 1);
% problems with display refresh rate synchronization tests, so set to 1 to
% skip tests and avoid errors, these tests are more important for tasks
% where timing needs to be very precise

[wind,rect] = Screen('OpenWindow', max(Screen('Screens'))); % open window on the max number of screens available (if two screens,max = 1)
Priority(MaxPriority(wind)); % make sure the computer will default to updating the window you care about
h = rect(4); w = rect(3); % define screens height and width

% setting up some stimulus stuff for placing shapes and text
xCenter = w/2; % center of x axis
yCenter = h/2; % center of y axis
radius = round(w*.15); % define radius

lCent = w/4; % center of the left side
rCent = 3*w/4; % center of the right side

% dimensions for left rectangle and right rectangle
lRect = [lCent-radius h/2-radius lCent+radius h/2+radius];
rRect = [rCent-radius h/2-radius rCent+radius h/2+radius];

Screen('TextFont',wind,'Courier'); % set the font to something non-offensive & portable
Screen('TextSize',wind,50); % set the font size to something reasonable

blk = BlackIndex(wind); % automatically pull out the black, white, and gray CLUTs
wht = WhiteIndex(wind);
gry = GrayIndex(wind);

Screen('FillRect', wind, blk); % set background
Screen('Flip',wind); %THIS YOU DON'T HAVE TO RE-SET EACH FLIP

DrawFormattedText(wind,'Setting up...','center','center',wht,40); % put up something on the screen while we finish prep
Screen('Flip',wind);
WaitSecs(2); % display content for two seconds

% Keyboard setup
KbName('UnifyKeyNames'); % help PTB deal with different keyboards.
esc_key_code = KbName('`~'); % this is the key we'll use to exit the experiment if we want to
continue_key_code = KbName('space'); % this is key to continue
% scanner_trigger = KbName('5%');
resp_keys = {'z','m'}; % FIRST entry for selecting left, SECOND entry for selecting right
    % These keys must be listed in order i.e. {left,right}
resp_key_codes = KbName(resp_keys); % get the key codes for the response keys

%------- Specific Study Setup -------%

fname = sprintf('clase_behavior_CLASE%s_%.4f.mat',subjID,now);
% this will create a unique data file name so that data will not be
% overwrote each iteration

cs = csvread(choiceset_fullpath,1); %read in the Choice Set

if do_full
    nT = size(cs,1); % number of trials for real task (must be divisible by 5!)
    nB = 5; % Number of blocks those trials are divided into (needs to be a factor of nT)
    triBlockStart = 1:(nT/nB):nT; % the first trial in each block
    subjdata.cs.triBlock = repmat(1:(nT/nB),1,nB); % trial numbers within-block
else
    % Set up numbers for a shortened 'test' version
    nT = 12;
    nB = 3;
    triBlockStart = 1:(nT/nB):nT;
    testTriBlock = repmat(1:(nT/nB),1,nB);
end

% set the timing for each trial event
prestime = 2; % number of seconds the image is presented
choicetime = 2; % number of seconds participants have to respond
isitime = 1; % time between choice & outcome
feedbacktime = 1; % number of seconds for display of feedback
ititime = repmat([1:3]',[nT,1]); % creating ITIs of 1, 2, or 3s
ititime = ititime(randperm(nT)); % select the first nT ITIs, and randomly sort them

total_trial_time = prestime + choicetime + isitime + feedbacktime + mean(ititime);
total_experiment_time = total_trial_time * nT / 60; % in MINUTES

cs = cs(randperm(size(cs,1),nT),:); % randomly sort the right # of trials
% Column 1: risky gain
% Column 2: risky loss
% Column 3: certain alternative
% Column 4: is check trial (1 = is check trial, 0 = not)

% Prep the struct which we'll eventually save out with all the study data
subjdata.subjID = subjID; % subject ID
subjdata.nT = nT; % number of trials
subjdata.cs.riskyGain = cs(:,1); % store the choiceset: risky gain
subjdata.cs.riskyLoss = cs(:,2); % store the choiceset: risky loss
subjdata.cs.alternative = cs(:,3); % store the choiceset: guaranteed alternative
subjdata.cs.ischecktrial = cs(:,4); % store the identifier for whether it was a check trial or not
subjdata.cs.choice = nan(size(cs,1),1); % used size of cs instead of nT for testing purposes when nT = 6
subjdata.cs.outcome = nan(size(cs,1),1);
subjdata.cs.loc = nan(size(cs,1),1);% what side of the screen were gamble and alt presented;loc = 1: gamble on left, alternative or right, loc = 2 alt on left, gamble on the right
subjdata.cs.response = cell(size(cs,1),1);
subjdata.cs.subjectIndex = repmat(subjID,size(cs,1),1); % put subject index in every row - will want this for analysis
subjdata.cs.RTs = nan(size(cs,1),1);
subjdata.cs.trial = (1:size(cs,1))'; % add trial number

subjdata.ts = struct(); %create a struct to store variable timing events
subjdata.ts.blockStart = nan(nB,1);

subjdata.params = struct(); % create a struct to store the preset timing events
subjdata.params.prestime = prestime; % forced viewing time
subjdata.params.choicetime = choicetime; % response window
subjdata.params.feedbackdelay = isitime; % ISI btwn choice & outcome
subjdata.params.feedbacktime = feedbacktime; % outcome viewing
subjdata.params.ititime = ititime; % ITI between outcome & subsequent choice

nTp = 10; % 5 quick practice trials
subjdata.practice = struct();
subjdata.practice.riskyGain =   [10, 33.15,    4,    2, 8,  5,  8, 22, 18,   10];
subjdata.practice.riskyLoss =   [-8,     0, -6.5, -3.5, 0, -4, -8,  0,  0, -7.5];
subjdata.practice.alternative = [ 0, 18.35,    0,    0, 2,  0,  0, 19,  7,    0];
subjdata.practice.choice = nan(nTp,1); %used size of cs instead of nT for testing purposes when nT = 6
subjdata.practice.outcome = nan(nTp,1);
subjdata.practice.loc = nan(nTp,1);% what side of the screen were gamble and alt presented;loc = 1: gamble on left, alternative or right, loc = 2 alt on left, gamble on the right
subjdata.practice.response = cell(nTp,1);
subjdata.practice.RTs = nan(nTp,1);

fid = fopen(sprintf('clase_behavior_CLASE%s_%.4f.csv',subjID,now),'w'); % open a new text file, set it to "write" status
fprintf(fid,'riskygain, riskyloss, certainalternative, ischecktrial, loc, choice, outcome, RT, ISI, ITI, trial, block, triBlock, subjID\n'); % Make the header row


% The start of the script will be different depending on reStart input

try
    %-------------Brief INTRO-----------%
    DrawFormattedText(wind,sprintf('As discussed in the instructions, this task consists of %g trials (anticipated to take roughly %g minutes). You will choose between a gamble and a guaranteed alternative.\n\n\n Press the left button (%s) to choose the option on the left and the right button (%s) to choose the option on the right.', nT, round(total_experiment_time), resp_keys{1}, resp_keys{2}),'center',.1*h,wht,40);
    Screen('Flip',wind,[],1);
    WaitSecs(1.5);
    DrawFormattedText(wind,'Press either response key to continue.','center',.9*h,wht,40);
    Screen('Flip',wind);
    while 1
        [keyIsDown,~,keyCode] = KbCheck(-1);
        if (keyIsDown && size(find(keyCode),2)==1)
            if keyCode(esc_key_code)
                error('Experiment aborted by user'); % allow aborting the study here
            elseif any(keyCode(resp_key_codes)) % participant can start study when ready
                break % change screen as soon as they respond
            end
        end
    end
    
    DrawFormattedText(wind,'Remember that you can only enter your choice by pressing a response key AFTER the ''Z - left    M - right'' prompt comes up on the bottom of the screen.\n\nTo ensure that you can respond in time, please keep one finger on the left key and one on the right key at all times during the task.','center',.1*h,wht,40);
    Screen('Flip',wind,[],1);
    WaitSecs(1.5);
    DrawFormattedText(wind,'Press either response key to continue.','center',.9*h,wht,40);
    Screen('Flip',wind);
    while 1
        [keyIsDown,~,keyCode] = KbCheck(-1);
        if (keyIsDown && size(find(keyCode),2)==1)
            if keyCode(esc_key_code)
                error('Experiment aborted by user'); % allow aborting the study here
            elseif any(keyCode(resp_key_codes)) % participant can start study when ready
                break % change screen as soon as they respond
            end
        end
    end

    DrawFormattedText(wind,'Please remember to evaluate each choice on its own merits, and to take your choices seriously.','center',.1*h,wht,40);
    Screen('Flip',wind,[],1);
    WaitSecs(1.5);
    DrawFormattedText(wind,'Press either response key to continue.','center',.9*h,wht,40);
    Screen('Flip',wind);
    while 1
        [keyIsDown,~,keyCode] = KbCheck(-1);
        if (keyIsDown && size(find(keyCode),2)==1)
            if keyCode(esc_key_code)
                error('Experiment aborted by user'); % allow aborting the study here
            elseif any(keyCode(resp_key_codes)) % participant can start study when ready
                break % change screen as soon as they respond
            end
        end
    end
    
    DrawFormattedText(wind,'If you have ANY questions, please ask your experimenter now!','center',.1*h,wht,40);
    Screen('Flip',wind,[],1);
    WaitSecs(1.5);
    DrawFormattedText(wind,'Press either response key to continue.','center',.9*h,wht,40);
    Screen('Flip',wind);
    while 1
        [keyIsDown,~,keyCode] = KbCheck(-1);
        if (keyIsDown && size(find(keyCode),2)==1)
            if keyCode(esc_key_code)
                error('Experiment aborted by user'); % allow aborting the study here
            elseif any(keyCode(resp_key_codes)) % participant can start study when ready
                break % change screen as soon as they respond
            end
        end
    end

    %---------------------Practice-----------------------%
    
    if do_prac     
        DrawFormattedText(wind,'You will now do ten (10) practice trials. These look and work identically to real trials (i.e. timing, dollar values), but do not count!','center',.1*h,wht,40);
        Screen('Flip',wind,[],1);
        WaitSecs(1.5);
        DrawFormattedText(wind,'Press BOTH response keys at the SAME time to continue to the practice trials! When you do, there will be a 5 second warning, and then the trials will begin.','center',.75*h,wht,40);
        Screen('Flip',wind);
        while 1
            [keyIsDown,~,keyCode] = KbCheck(-1);
            if (keyIsDown && size(find(keyCode),2)==2)
                if keyCode(esc_key_code)
                    error('Experiment aborted by user'); % allow aborting the study here
                elseif sum(keyCode(resp_key_codes))==2 % participant can start study when ready
                    break % change screen as soon as they respond
                end
            end
        end
        
        DrawFormattedText(wind, 'Get ready! Beginning in 5 seconds...', 'center', 'center', wht,40);
        Screen('Flip', wind); % update display
        WaitSecs(4);
        
        DrawFormattedText(wind, '+', 'center', 'center', wht,40);
        Screen('Flip', wind); % update display
        WaitSecs(1);
        
        %--Start practice--%
        subjdata.practice.studystart = GetSecs; % log the study start time if it's the first trial
        if do_TTL
            write(device,sprintf("mh%c%c", 20, 0), "char"); % TTL pulse marking start of the practice block
        end
        
        for t = 1:nTp
            subjdata.practice.trialStart(t) = GetSecs; %when does the trial start? there will be a slight delay between this and stimStart(t)
            loc = randperm(2,1); %randomly select location of options on the screen. loc = 1: gamble on left, alternative or right, loc = 2 alt on left, gamble on the right
            subjdata.practice.loc(t) = loc; %store location
            %------stimulus presentation---------%
            %prepare stimulus%
            Screen('FillOval',wind,wht,[lRect' rRect']); %draw two ovals on the left and right side of the screen
            if loc == 1
                Screen('DrawLine', wind, blk, lRect(1),yCenter,lRect(3), yCenter, 5); %draw line in the middle of gamble on left side of screen
                %place text
                DrawFormattedText(wind,nicemoneytext(subjdata.practice.riskyGain(t)),'center','center',blk,[],[],[],[],[],[lRect(1) lRect(2) lRect(3) lRect(4)- radius]); %gain text
                DrawFormattedText(wind,nicemoneytext(subjdata.practice.riskyLoss(t)),'center','center',blk,[],[],[],[],[],[lRect(1) lRect(2)+radius lRect(3) lRect(4)]); %loss text
                DrawFormattedText(wind,nicemoneytext(subjdata.practice.alternative(t)),'center','center',blk,[],[],[],[],[],rRect); % alt text
            elseif loc == 2
                Screen('DrawLine', wind, blk, rRect(1),yCenter, rRect(3), yCenter, 5); % draw line in middle of gamble on right side of screen
                %place text
                DrawFormattedText(wind,nicemoneytext(subjdata.practice.alternative(t)),'center','center',blk,[],[],[],[],[],lRect); %alt text
                DrawFormattedText(wind,nicemoneytext(subjdata.practice.riskyGain(t)),'center','center',blk,[],[],[],[],[],[rRect(1) rRect(2) rRect(3) rRect(4)-radius]); %gain text
                DrawFormattedText(wind,nicemoneytext(subjdata.practice.riskyLoss(t)),'center','center',blk,[],[],[],[],[],[rRect(1) rRect(2)+radius rRect(3) rRect(4)]); % loss text
            end
            DrawFormattedText(wind,'OR','center','center',wht); % place OR between both ovals
            
            
            subjdata.practice.stimStart(t) = Screen('Flip',wind,[],1); %show the stimuli
            if do_TTL
                write(device,sprintf("mh%c%c",30, 0), "char"); % TTL pulse marking start of option presentation
            end
            
            % ready the response period stuff that will show up in the response
            % collection period
            DrawFormattedText2('<size=60><b>Z - left','win', wind, 'sx', 'center', 'sy', 'center', 'xalign', 'center', 'yalign', 'center', 'winRect', [lRect(1),lRect(4) lRect(3) lRect(4)+lRect(2)]);
            DrawFormattedText2('<size=60><b>M - right','win', wind, 'sx', 'center', 'sy', 'center', 'xalign', 'center', 'yalign', 'center', 'winRect', [rRect(1),rRect(4) rRect(3) rRect(4)+rRect(2)]);
            Screen('FrameRect',wind,[],[],10);
            
            while GetSecs - subjdata.practice.studystart < t*prestime + (t-1)*(choicetime + isitime + feedbacktime) + sum(ititime(1:(t-1)))
                [keyIsDown,~,keyCode] = KbCheck(-1);
                %if keyIsDown
                if (keyIsDown && size(find(keyCode),2) ==1)
                    if keyCode(esc_key_code)
                        error('Experiment aborted by user'); % allow aborting during stimulus presentation
                    end
                end
            end
            
            %----------response collection-----------%
            %display keys for choices on screen
            subjdata.practice.choiceStart(t) = Screen('Flip',wind); % show the v and n on the screen, nows ps can respond
            if do_TTL
                write(device,sprintf("mh%c%c",31, 0), "char"); % TTL pulse marking start of response window
            end
            
            while GetSecs - subjdata.practice.studystart < t*(prestime + choicetime) + (t-1)*(isitime + feedbacktime) + sum(ititime(1:(t-1)))
                [keyIsDown,~,keyCode] = KbCheck(-1);
                %outp(adOut,1); %send marking pulse
                %if keyIsDown
                if (keyIsDown && size(find(keyCode),2) ==1)
                    if keyCode(esc_key_code)
                        error('Experiment aborted by user'); % allow aborting the study here
                    elseif any(keyCode(resp_key_codes)) % IF the pressed key matches a response key...
                        subjdata.practice.RTs(t) = GetSecs - subjdata.practice.choiceStart(t); % record RT
                        subjdata.practice.extraTime(t) = choicetime - subjdata.practice.RTs(t); % store the extra time left over once p makes a choice
                        %subjdata.ts.feedbackdelayStart(t) = GetSecs; %now the isi starts
                        %outp(adOut,0); % turn off marker - tell biopac that event is over
                        break % change screen as soon as they respond
                    end
                end
            end
            
            
            %------ISI------%
            Screen('gluDisk',wind,wht,xCenter,yCenter,3); % make a white dot that will be displayed during ISI
            subjdata.practice.isiStart(t) = Screen('Flip',wind); %now isi starts
            if do_TTL
                write(device,sprintf("mh%c%c",32, 0), "char"); % TTL pulse marking start of ISI
            end
            
            if any(keyCode(resp_key_codes))
                subjdata.practice.response{t} = KbName(keyCode); % record response
                if subjdata.practice.loc(t) == 1 && (find(keyCode(resp_key_codes)) == 1) %if gamble is on the left and p selected v
                    subjdata.practice.choice(t) = 1; % then they gambled
                elseif subjdata.practice.loc(t) == 2 && (find(keyCode(resp_key_codes)) == 2) % if gamble is on the right and p selected n
                    subjdata.practice.choice(t) = 1; %then they gambled
                else
                    subjdata.practice.choice(t) = 0; % otherwise they did not gamble
                end
            else
                %subjdata.cs.response{t} = NaN; %did not respond in time
                subjdata.practice.choice(t) = NaN; %did not respond in time
            end
            
            while GetSecs - subjdata.practice.studystart < t*(prestime+isitime) + min([subjdata.practice.RTs(t) choicetime]) + (t-1)*(choicetime+feedbacktime) + sum(ititime(1:(t-1)));
                [keyIsDown,~,keyCode] = KbCheck(-1);
                if (keyIsDown && size(find(keyCode),2) ==1)
                    %if keyIsDown
                    if keyCode(esc_key_code)
                        error('Experiment aborted by user'); % allow aborting during stimulus presentation
                    end
                end
            end
            
            %-----FEEDBACK-----%
            
            % prepare feedback stimulus during ISI
            if ~isnan(subjdata.practice.choice(t))
                if subjdata.practice.choice(t) == 1 %if gamble taken
                    OC = randi(2,1); % generate a random number, 1 or 2; if 1, outcome is gain. if 2, outcome is loss
                    if OC == 1 %if OC is 1, gain is outcome
                        subjdata.practice.outcome(t) = subjdata.practice.riskyGain(t); %store gain amount
                        if loc == 1 %if gamble is on left
                            Screen('FillArc',wind,wht,lRect,270,180); %display gain amount on left
                            DrawFormattedText(wind,nicemoneytext(subjdata.practice.riskyGain(t)),'center','center',blk,[],[],[],[],[],[lRect(1) lRect(2) lRect(3) lRect(4)-radius]);
                        elseif loc ==2 % if gamble is on the right
                            Screen('FillArc',wind,wht,rRect,270,180); %display gain amount on right
                            DrawFormattedText(wind,nicemoneytext(subjdata.practice.riskyGain(t)),'center','center',blk,[],[],[],[],[],[rRect(1) rRect(2) rRect(3) rRect(4)-radius]);
                        end
                    elseif OC ==2  %if OC is 2, loss is the outcome
                        subjdata.practice.outcome(t) = subjdata.practice.riskyLoss(t); % store loss amount
                        if loc ==1 %if gamble is on the right
                            subjdata.practice.outcome(t) = subjdata.practice.riskyLoss(t); % store loss amount
                            Screen('FillArc',wind,wht,lRect,90,180); %display outcome
                            DrawFormattedText(wind,nicemoneytext(subjdata.practice.riskyLoss(t)),'center','center',blk,[],[],[],[],[],[lRect(1) lRect(2)+radius lRect(3) lRect(4)]);
                        elseif loc == 2 % if
                            Screen('FillArc',wind,wht,rRect,90,180);
                            DrawFormattedText(wind,nicemoneytext(subjdata.practice.riskyLoss(t)),'center','center',blk,[],[],[],[],[],[rRect(1) rRect(2)+radius rRect(3) rRect(4)]);
                        end
                    end
                else
                    subjdata.practice.outcome(t) = subjdata.practice.alternative(t); %store alt amount
                    if loc == 1 %if alt is on the right side of screen
                        Screen('FillOval', wind, wht,rRect);
                        DrawFormattedText(wind,nicemoneytext(subjdata.practice.alternative(t)),'center','center',blk,[],[],[],[],[],rRect);
                    else
                        
                        Screen('FillOval', wind, wht,lRect);
                        DrawFormattedText(wind,nicemoneytext(subjdata.practice.alternative(t)),'center','center',blk,[],[],[],[],[],lRect);
                    end
                end
            else
                DrawFormattedText(wind,'You did not respond in time','center','center', wht, 40);
                subjdata.practice.outcome(t) = NaN;
                
            end % end this feedback for loop; now feedback has been prepped and is ready to be displayed when isi is over
            
            
            subjdata.practice.otcStart(t) =  Screen('Flip',wind); % show feedback
            if do_TTL
                write(device,sprintf("mh%c%c",40, 0), "char"); % TTL pulse marking outcome presentation
            end
            
            Screen('gluDisk',wind,wht,xCenter,yCenter,3); % make a white dot that will be displayed during iti
            
            
            while GetSecs - subjdata.practice.studystart < t*(prestime + isitime + feedbacktime) + (t-1)*(choicetime) + sum(ititime(1:(t-1))) + min([subjdata.practice.RTs(t) choicetime])
                [keyIsDown,~,keyCode] = KbCheck(-1);
                if (keyIsDown && size(find(keyCode),2) ==1)
                    %if keyIsDown
                    if keyCode(esc_key_code)
                        error('Experiment aborted by user'); % allow experiment aborting here
                    end
                end
            end
            
            
            %subjdata.ts.otcEnd(t) = GetSecs; % end of outcome display
            
            % ------ ITI ------ %
            
            subjdata.practice.itiStart(t) = Screen('Flip',wind); %show white dot on the screen
            if do_TTL
                write(device,sprintf("mh%c%c",41, 0), "char"); % TTL pulse marking start of ITI
            end
            
            %fprintf(fid,'riskyGain %0.2f, safe, %0.2f, loc, %g, choice, %g, outcome, %0.2f, RT, %0.2f, ISI, %g, ITI, %g\n', subjdata.practice.riskyGain(t),subjdata.practice.alternative(t),subjdata.practice.loc(t), subjdata.practice.choice(t), subjdata.practice.outcome(t), subjdata.practice.RTs(t)); %save txt file of what we have
            save(fname,'subjdata'); % save what data we have
            
            %wait a certain amount of time until iti ends
            while GetSecs - subjdata.practice.studystart < t*(prestime + isitime + choicetime + feedbacktime) + sum(ititime(1:t))  % control the entire experiment length on a per-trial basis
                [keyIsDown,~,keyCode] = KbCheck(-1);
                if (keyIsDown && size(find(keyCode),2) ==1)
                    %if keyIsDown
                    if keyCode(esc_key_code)
                        error('Experiment aborted by user'); % allow experiment aborting here
                    end
                end
            end
            
        end %ends the loop for the practice trials.
        
        subjdata.practice.studystop = GetSecs; % mark the end-time
        if do_TTL
            write(device,sprintf("mh%c%c",21, 0), "char"); % TTL pulse marking end of practice block
        end
        
        DrawFormattedText(wind,'You have completed the practice!\n\n\n If you have any questions, please ask the experimenter now!','center','center', wht, 40);
        DrawFormattedText(wind,'Waiting for the experimenter.','center',.9*h, wht, 40);
        Screen('Flip',wind);
        WaitSecs(2);
        while 1
            [keyIsDown,~,keyCode] = KbCheck(-1);
            if (keyIsDown && size(find(keyCode),2) ==1)
                if keyCode(esc_key_code)
                    error('Experiment aborted by user'); % allow aborting the study here
                elseif any(keyCode(continue_key_code)) % participant can start study when ready
                    break % change screen as soon as they respond
                end
            end
        end
        Screen('Flip',wind);
        WaitSecs(1);
        
    end

    % Once we are ready to start a block, show waiting for experimenter screen:
    DrawFormattedText(wind, 'Computer ready.', 'center', 'center', wht,40);
    DrawFormattedText(wind,'Waiting for the experimenter.','center',.9*h, wht, 40);
    Screen('Flip', wind); % update display to show break_message
    WaitSecs(1);
    while 1
        [keyIsDown,~,keyCode] = KbCheck(-1);
        if (keyIsDown && size(find(keyCode),2) ==1)
            if keyCode(esc_key_code)
                error('Experiment aborted by user'); % allow aborting the study here
            elseif keyCode(continue_key_code) %
                break % change screen as soon as they respond
            end
        end
    end
    Screen('Flip',wind);
    WaitSecs(1);
    
    DrawFormattedText(wind,'You''re ready to begin the main task.\n\n\nPress BOTH response keys at the SAME TIME to begin!','center','center',wht,40);
    Screen('Flip',wind);
    while 1
        [keyIsDown,~,keyCode] = KbCheck(-1);
        if (keyIsDown && size(find(keyCode),2)==2)
            if keyCode(esc_key_code)
                error('Experiment aborted by user'); % allow aborting the study here
            elseif sum(keyCode(resp_key_codes))==2 % participant can start study when ready
                break % change screen as soon as they respond
            end
        end
    end

    DrawFormattedText(wind, 'Get ready! Beginning in 5 seconds...', 'center', 'center', wht,40);
    Screen('Flip', wind); % update display
    WaitSecs(4);
    
    DrawFormattedText(wind, '+', 'center', 'center', wht,40);
    Screen('Flip', wind); % update display
    WaitSecs(1);
    
    %Now the real task can begin!
    
    if do_TTL
        write(device,sprintf("mh%c%c",50, 0), "char"); % TTL pulse marking start of the actual study
    end
    subjdata.ts.studystart = GetSecs; % log the study start time
    b = 1;
    triStart = 1;
    
    
    for t = triStart:nT
        
        if do_full
            triBlock = subjdata.cs.triBlock(t);
        else
            triBlock = testTriBlock(t);
        end
        
        %----------interblock break----------%
        if t > ((nT/nB)*b)
            b=b+1;
            DrawFormattedText(wind,sprintf('You have completed block %g of %g.\n\nThere will now be a brief break before continuing the task.\n\nAfter 30 seconds, the experiment will automatically continue.\n\n\nYou may also continue at any time by pressing either the left or right key.',b-1,nB), 'center', 'center', wht,40);
            Screen('Flip', wind); % update display to show break_message
            subjdata.ts.blockBreakStart(b) = GetSecs;
            WaitSecs(1);
            while (GetSecs - subjdata.ts.blockBreakStart(b)) < 29
                [keyIsDown,~,keyCode] = KbCheck(-1);
                if (keyIsDown && size(find(keyCode),2) ==1)
                    if keyCode(esc_key_code)
                        error('Experiment aborted by user'); % allow aborting the study here
                    elseif any(keyCode(resp_key_codes)) %
                        break % change screen as soon as they respond
                    end
                end
                % wait 1 ms before checking the keyboard again to prevent
                % overload of the machine at elevated Priority()
            end
            
            DrawFormattedText(wind, 'Beginning the next block in 5 seconds...', 'center', 'center', wht,40);
            Screen('Flip', wind); % update display
            WaitSecs(4);
            
            DrawFormattedText(wind, '+', 'center', 'center', wht,40);
            Screen('Flip', wind); % update display
            WaitSecs(1);
        end %if t > ((nT/nB)*b)
        
        subjdata.ts.trialStart(t) = GetSecs;
        
        %store the start of each block as when the first trial starts
        for blockNum = 1:nB
            if t == triBlockStart(blockNum)
                subjdata.ts.blockStart(blockNum) = subjdata.ts.trialStart(t);
            end
        end
                
        loc = randperm(2,1); %randomly select location of options on the screen. loc = 1: gamble on left, alternative or right, loc = 2 alt on left, gamble on the right
        subjdata.cs.loc(t) = loc; %store location
        %------stimulus presentation---------%
        %prepare stimulus%
        Screen('FillOval',wind,wht,[lRect' rRect']); %draw two ovals on the left and right side of the screen
        if loc == 1
            Screen('DrawLine', wind, blk, lRect(1),yCenter,lRect(3), yCenter, 5); %draw line in the middle of gamble on left side of screen
            %place text
            DrawFormattedText(wind,nicemoneytext(subjdata.cs.riskyGain(t)),'center','center',blk,[],[],[],[],[],[lRect(1) lRect(2) lRect(3) lRect(4)- radius]); %gain text
            DrawFormattedText(wind,nicemoneytext(subjdata.cs.riskyLoss(t)),'center','center',blk,[],[],[],[],[],[lRect(1) lRect(2)+radius lRect(3) lRect(4)]); %loss text
            DrawFormattedText(wind,nicemoneytext(subjdata.cs.alternative(t)),'center','center',blk,[],[],[],[],[],rRect); % alt text
        elseif loc == 2
            Screen('DrawLine', wind, blk, rRect(1),yCenter, rRect(3), yCenter, 5); % draw line in middle of gamble on right side of screen
            %place text
            DrawFormattedText(wind,nicemoneytext(subjdata.cs.alternative(t)),'center','center',blk,[],[],[],[],[],lRect); %alt text
            DrawFormattedText(wind,nicemoneytext(subjdata.cs.riskyGain(t)),'center','center',blk,[],[],[],[],[],[rRect(1) rRect(2) rRect(3) rRect(4)-radius]); %gain text
            DrawFormattedText(wind,nicemoneytext(subjdata.cs.riskyLoss(t)),'center','center',blk,[],[],[],[],[],[rRect(1) rRect(2)+radius rRect(3) rRect(4)]); % loss text
        end
        DrawFormattedText(wind,'OR','center','center',wht); % place OR between both ovals
        
        subjdata.ts.stimStart(t) = Screen('Flip',wind,[],1); %show the stimuli
        if do_TTL
            write(device,sprintf("mh%c%c",60, 0), "char"); % TTL pulse marking screen flip that shows choice options
        end
        
        % ready the response period stuff that will show up in the response
        % collection period
        DrawFormattedText2('<size=60><b>Z - left','win', wind, 'sx', 'center', 'sy', 'center', 'xalign', 'center', 'yalign', 'center', 'winRect', [lRect(1),lRect(4) lRect(3) lRect(4)+lRect(2)]);
        DrawFormattedText2('<size=60><b>M - right','win', wind, 'sx', 'center', 'sy', 'center', 'xalign', 'center', 'yalign', 'center', 'winRect', [rRect(1),rRect(4) rRect(3) rRect(4)+rRect(2)]);
        Screen('FrameRect',wind,[],[],10);
        
        while GetSecs - subjdata.ts.blockStart(b) < triBlock*(prestime) + (triBlock-1)*(isitime + choicetime + feedbacktime) + sum(ititime(triBlockStart(b):(t-1)))
            [keyIsDown,~,keyCode] = KbCheck(-1);
            if (keyIsDown && size(find(keyCode),2) ==1)
                if keyCode(esc_key_code)
                    error('Experiment aborted by user'); % allow aborting during stimulus presentation
                end
            end
        end
        
        %----------response collection-----------%
        %display keys for choices on screen
        subjdata.ts.choiceStart(t) = Screen('Flip',wind); % show the v and n on the screen, nows ps can respond
        if do_TTL
            write(device,sprintf("mh%c%c", 61, 0), "char"); % TTL pulse marking start of the response window
        end
        
        while GetSecs - subjdata.ts.blockStart(b) < triBlock*(prestime + choicetime) + (triBlock-1)*(isitime + feedbacktime) + sum(ititime(triBlockStart(b):(t-1)))
            [keyIsDown,~,keyCode] = KbCheck(-1);
            %outp(adOut,1); %send marking pulse
            if (keyIsDown && size(find(keyCode),2) ==1)
                if keyCode(esc_key_code)
                    error('Experiment aborted by user'); % allow aborting the study here
                elseif any(keyCode(resp_key_codes)) % IF the pressed key matches a response key...
                    subjdata.cs.RTs(t) = GetSecs - subjdata.ts.choiceStart(t); % record RT
                    subjdata.ts.extraTime(t) = choicetime - subjdata.cs.RTs(t); % store the extra time left over once p makes a choice
                    %outp(adOut,0); % turn off marker - tell biopac that event is over
                    break % change screen as soon as they respond
                end
            end
        end
        
        
        %------ISI------%
        Screen('gluDisk',wind,wht,xCenter,yCenter,3); % make a white dot that will be displayed during ISI
        subjdata.ts.isiStart(t) = Screen('Flip',wind); %now isi starts
        if do_TTL           
            write(device,sprintf("mh%c%c",62, 0), "char"); % TTL pulse marking end of the response window (due to button press OR expiration of time
        end
        
        if any(keyCode(resp_key_codes))
            subjdata.cs.response{t} = KbName(keyCode); % record response
            if subjdata.cs.loc(t) == 1 && (find(keyCode(resp_key_codes)) == 1) %if gamble is on the left and p selected v
                subjdata.cs.choice(t) = 1; % then they gambled
            elseif subjdata.cs.loc(t)==2 && (find(keyCode(resp_key_codes)) == 2) % if gamble is on the right and p selected n
                subjdata.cs.choice(t) = 1; %then they gambled
            else
                subjdata.cs.choice(t) = 0; % otherwise they did not gamble
            end
        else
            %subjdata.cs.response{t} = NaN; %did not respond in time
            subjdata.cs.choice(t) = NaN; %did not respond in time
        end
        
        
        while GetSecs - subjdata.ts.blockStart(b) < triBlock*(prestime + isitime) + min([subjdata.cs.RTs(t) choicetime]) + (triBlock-1)*(choicetime+feedbacktime) + sum(ititime(triBlockStart(b):(t-1)))
            [keyIsDown,~,keyCode] = KbCheck(-1);
            if (keyIsDown && size(find(keyCode),2) ==1)
                %if keyIsDown
                if keyCode(esc_key_code)
                    error('Experiment aborted by user'); % allow aborting during stimulus presentation
                end
            end
        end
        
        
        %-----FEEDBACK-----%
        
        % prepare feedback stimulus during ISI
        if ~isnan(subjdata.cs.choice(t))
            if subjdata.cs.choice(t) == 1 %if gamble taken
                OC = randi(2,1); % generate a random number, 1 or 2; if 1, outcome is gain. if 2, outcome is loss
                if OC == 1 %if OC is 1, gain is outcome
                    subjdata.cs.outcome(t) = subjdata.cs.riskyGain(t); %store gain amount
                    if loc == 1 %if gamble is on left
                        Screen('FillArc',wind,wht,lRect,270,180); %display gain amount on left
                        DrawFormattedText(wind,nicemoneytext(subjdata.cs.riskyGain(t)),'center','center',blk,[],[],[],[],[],[lRect(1) lRect(2) lRect(3) lRect(4)-radius]);
                    elseif loc ==2 % if gamble is on the right
                        Screen('FillArc',wind,wht,rRect,270,180); %display gain amount on right
                        DrawFormattedText(wind,nicemoneytext(subjdata.cs.riskyGain(t)),'center','center',blk,[],[],[],[],[],[rRect(1) rRect(2) rRect(3) rRect(4)-radius]);
                    end
                elseif OC ==2  %if OC is 2, loss is the outcome
                    subjdata.cs.outcome(t) = subjdata.cs.riskyLoss(t); % store loss amount
                    if loc ==1 %if gamble is on the right
                        subjdata.cs.outcome(t) = subjdata.cs.riskyLoss(t); % store loss amount
                        Screen('FillArc',wind,wht,lRect,90,180); %display outcome
                        DrawFormattedText(wind,nicemoneytext(subjdata.cs.riskyLoss(t)),'center','center',blk,[],[],[],[],[],[lRect(1) lRect(2)+radius lRect(3) lRect(4)]);
                    elseif loc == 2 % if
                        Screen('FillArc',wind,wht,rRect,90,180);
                        DrawFormattedText(wind,nicemoneytext(subjdata.cs.riskyLoss(t)),'center','center',blk,[],[],[],[],[],[rRect(1) rRect(2)+radius rRect(3) rRect(4)]);
                    end
                end
            else
                subjdata.cs.outcome(t) = subjdata.cs.alternative(t); %store alt amount
                if loc == 1 %if alt is on the right side of screen
                    Screen('FillOval', wind, wht,rRect);
                    DrawFormattedText(wind,nicemoneytext(subjdata.cs.alternative(t)),'center','center',blk,[],[],[],[],[],rRect);
                else
                    
                    Screen('FillOval', wind, wht,lRect);
                    DrawFormattedText(wind,nicemoneytext(subjdata.cs.alternative(t)),'center','center',blk,[],[],[],[],[],lRect);
                end
            end
        else
            DrawFormattedText(wind,'You did not respond in time','center','center', wht, 40);
            subjdata.cs.outcome(t) = NaN;
            
        end % end this feedback for loop; now feedback has been prepped and is ready to be displayed when isi is over
        
        subjdata.ts.otcStart(t) =  Screen('Flip',wind); % show feedback
        if do_TTL
            write(device,sprintf("mh%c%c", 70, 0), "char"); % TTL pulse marking start of the outcome epoch
        end
        
        Screen('gluDisk',wind,wht,xCenter,yCenter,3); % make a white dot that will be displayed during iti
        
        
        while GetSecs - subjdata.ts.blockStart(b) < triBlock*(prestime + isitime + feedbacktime) + (triBlock-1)*(choicetime) + sum(ititime(triBlockStart(b):(t-1))) + min([subjdata.cs.RTs(t) choicetime])
            [keyIsDown,~,keyCode] = KbCheck(-1);
            if (keyIsDown && size(find(keyCode),2) ==1)
                %if keyIsDown
                if keyCode(esc_key_code)
                    error('Experiment aborted by user'); % allow experiment aborting here
                end
            end
        end
        
        
        %subjdata.ts.otcEnd(t) = GetSecs; % end of outcome display
        
        % ------ ITI ------ %
        
        subjdata.ts.itiStart(t) = Screen('Flip',wind); %show white dot on the screen
        if do_TTL
            write(device,sprintf("mh%c%c", 71, 0), "char"); % TTL pulse marking end of outcome epoch
        end
        
        fprintf(fid,'%0.2f, %0.2f, %0.2f, %g, %g, %g, %0.2f, %0.2f, %g, %g, %g, %g, %g, %s\n', ...
            subjdata.cs.riskyGain(t), subjdata.cs.riskyLoss(t), subjdata.cs.alternative(t), subjdata.cs.ischecktrial(t), ...
            subjdata.cs.loc(t), subjdata.cs.choice(t), subjdata.cs.outcome(t), subjdata.cs.RTs(t), ...
            subjdata.params.feedbackdelay, subjdata.params.ititime(t), ...
            t, b, triBlock, subjID); %save txt file of what we have
        save(fname,'subjdata'); % save what data we have
        
        %wait a certain amount of time until iti ends
        while GetSecs - subjdata.ts.blockStart(b) < triBlock*(prestime + isitime + choicetime + feedbacktime) + sum(ititime(triBlockStart(b):t))
            [keyIsDown,~,keyCode] = KbCheck(-1);
            if (keyIsDown && size(find(keyCode),2) ==1)
                if keyCode(esc_key_code)
                    error('Experiment aborted by user'); % allow experiment aborting here
                end
            end
        end
        
    end %for t = triStart:nT
    
    subjdata.ts.studystop = GetSecs; % mark the end-time
    if do_TTL
        write(device,sprintf("mh%c%c", 51, 0), "char"); % TTL pulse marking end of the study
    end
    
    %---Outcome selection and wrap up---%
    DrawFormattedText(wind,'Task Complete!','center','center',wht,40); % put up a screen placeholder
    Screen('Flip',wind);
    WaitSecs(2);
    
    DrawFormattedText(wind, 'Randomly selecting 1 outcome...', 'center', 'center', wht, 40); %display text while we randomly select choice
    Screen('Flip', wind);
    WaitSecs(2);
    
    
    subjdata.realOutcomesInds = find(isfinite(subjdata.cs.outcome)); %find in outcomes the rows with finite numbers, ignore NaNs
    subjdata.payoutTrial = subjdata.realOutcomesInds(randi(length(subjdata.realOutcomesInds),1)); % randomly select one of the trials to pay out
    subjdata.fullOutcome = subjdata.cs.outcome(subjdata.payoutTrial); % pull out the amount of the outcome
    
    %display results of random generator
    DrawFormattedText(wind,sprintf('Randomly selected trial: #%d.\n\nThe outcome of this trial is $%.2f.\n\n Please let the experimenter know that you are done with this session of the task!', subjdata.payoutTrial, subjdata.fullOutcome), 'center', 'center', wht,40);
    Screen('Flip', wind);
    WaitSecs(1);
    while keyCode(continue_key_code) == 0 % wait for experimenter to press space bar
        [keyIsDown,~,keyCode] = KbCheck(-1);
        if (keyIsDown && size(find(keyCode),2) ==1)
            if keyCode(esc_key_code)
                error('Experiment aborted by user'); % allow aborting the study here
            elseif keyCode(continue_key_code) %
                break % change screen as soon as they respond
            end
        end
        % wait 1 ms before checking the keyboard again to prevent
        % overload of the machine at elevated Priority()
    end
    
    fclose(fid); % close the text file
    save(fname,'subjdata'); % save what we have
    
    
    sca % clear the screen (note: this also clears things like "HideCursor"...)
    Priority(0); % reset screen priority
    
    %fprintf('\nExperiment took %0.5f seconds when it should have taken %0.1f seconds.\n',subjdata.ts.studystop-subjdata.ts.studystart, nT*(prestime+choicetime+feedbacktime) + sum(feedbackdelay) + sum(ititime))
    
catch ME % what to do if something breaks in the "try" statment, ME will show you the exception/error
    
    fclose(fid); % close the text file
    save(fname,'subjdata'); % save what data we have
    sca % clear the screen (note: this also clears things like "HideCursor"...)
    Priority(0); % return screen priority to default
    rethrow(ME); %show the error
end
end % end this function