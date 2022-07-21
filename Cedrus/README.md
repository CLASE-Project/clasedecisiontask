## Using Cedrus
*Last update: July 21 2022*
* ```Cedrus_TTL.m``` is a function that enables hexidecimal TTL to NLX transmission during a behavioral task.
* This function is based off of the script ```CedrusSetup.m```. The script is not needed for any of these functions to work, but provides more detail about the code written in the ```Cedrus_TTL.m``` function. 
* The source of ```CedrusSetup.m```, ```getByte.m```, and ```setPulseDuration.m``` can be found in support documents on the [Cedrus website](https://www.cedrus.com/support/xid/matlab.htm).

### Requirements
* ```getByte.m``` and ```setPulseDuration.m``` must be on the MATLAB path for ```Cedrus_TTL.m``` to work.
* MATLAB must be opened before the Cedrus c-pod is connected to the computer from which the behavioral task runs. If the c-pod is connected before MATLAB is opened, it won't work.
* Must use MATLAB R2020a or later version.

### User guide
#### 1. Assign distinct TTL markers for event types
#### 
* An example of what this would look like can be found in ```hexidecmial TTL key.txt```.
#### 2. Every time you wish to send a TTL, the following code should be written immediately following the line containing the event.
* *Example:*
  ```   
        subjdata.ts.stimStart(t) = Screen('Flip',wind,[],1); %show the stimuli
        if do_TTL
            %DatapixxAOttl(); % TTL pulse marking screen flip that shows choice options
            write(device,sprintf("mh%c%c",70, 0), "char"); % TTL pulse marking screen flip that shows choice options
        end
##### Where ```DatapixxAOttl();```, used to send TTLs with Datapixx and now obsolete, is replaced by ```write(device,sprintf("mh%c%c",70, 0), "char");```
#### 3. Replace the number within the ```sprintf``` call with the event marker of your choosing for every distinct event.
##### 
* In the example above, ```70``` would be used to mark stimuli presentation, and would be changed to a different number for a different event type. 
#### 4. Call ```Cedrus_TTL.m``` with outputs from within the main behavioral task function or script. 
*Note:* This function needs to be called before the first time the ```Screen``` function is called if using Psychtoolbox.
##### 
* *Example:* ```[~, device] = Cedrus_TTL()```
