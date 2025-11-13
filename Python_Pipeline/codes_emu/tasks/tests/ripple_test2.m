%search and add ripple path
disp('Searching for Trellis installation...')

if ~contains(path,'\xippmex;')
    valnames = winqueryreg('name','HKEY_LOCAL_MACHINE','SOFTWARE\Microsoft\Windows\CurrentVersion\Installer\Folders');
    xippmexix = find(cellfun(@(x) ~isempty(regexp(x,'(Ripple\\Trellis\\Tools\\xippmex\\)$','match')),valnames));

    if isempty(xippmexix)
        error('Trellis not found, add path using pathtool.')
    else
        addpath(valnames{xippmexix})
        disp('Found and added to path.')
    end
else
    disp('xippmex already in path.')
end
%%
disp('Connecting via TCP...')
status = xippmex('tcp'); %The processor, Trellis, and Xippmex must be in the same network mode
if status ~= 1
    error('Unable to connect to ripple.')
else
    disp('Connection successful.')
end
%% GETTING INFO ABOUT AVAILABLE CHANNELS
disp('Getting list of micros and analog channels...')
fprintf('Number of micros channels:')
micchans   = xippmex('elec','micro');
disp(length(micchans));
fprintf('Number of  analog channels:')
analchans   = xippmex('elec','analog');
disp(length(analchans))

%% INDEP ENABLING CHANNELS TESTS
% disabling channels
xippmex('signal',analchans,'30ksps',zeros(1,length(analchans)));
xippmex('signal',micchans,'raw',zeros(1,length(micchans)));
% FE TEST
disp('Checking FE change after enabling channel...')
disp('analog...')
status_a11 = xippmex('signal',analchans,'30ksps');
xippmex('signal',analchans(1),'30ksps',1);
status_a12 = xippmex('signal',analchans,'30ksps');
if sum(status_a11 ~= status_a12)==30 %analog in has 30 channels
    disp('enabling one analog chan change state of all.')
else
    disp('analog chans can be enabled independently')
end
disp('micro...')
disp('Checking FE change after enabling channel...')
status_m1 = xippmex('signal',micchans,'raw');
xippmex('signal',micchans(1),'raw',1);
status_m2 = xippmex('signal',micchans,'raw');
if sum(status_m11 ~= status_m12)==32
    disp('enabling one micro chan change state of FE.')
else
    disp('micro chans can be enabled independently')
end

%% TRELLIS CHANGES CHANNELS ANALOG
% disabling channels
status_1ksps_def = xippmex('signal',analchans,'1ksps');

for i =1:100
    pause(1)
    status_1ksps_new = xippmex('signal',analchans,'1ksps');
    if any(status_1ksps_def ~= status_1ksps_new)
       fprintf('analog: status changed after aprox %d seconds.\n',i); 
    end
end

%% TRELLIS CHANGES CHANNELS MICROS

status_lfp_def = xippmex('signal',micchans,'lfp');

for i =1:100
    pause(1)
    status_lfp_new = xippmex('signal',micchans,'lfp');
    if any(status_lfp_def ~= status_lfp_new)
       fprintf('micros: status changed after aprox %d seconds.\n',i); 
    end
end

%% RUN IF SETTING A CHANNEL CHANGE ALL FE
analchans_test = analchans(1);
micchans_test = unique(ceil(micchans/32))*32;

%% RUN IF SETING A CHANNEL CHANGE JUST THAT CHANNEL
analchans_test = analchans;
micchans_test = micchans;

%% TRELLIS ENABLING CHANNELS TEST
% disabling channels
xippmex('signal',analchans_test,'30ksps',zeros(1,length(analchans_test)));
xippmex('signal',micchans_test,'raw',zeros(1,length(micchans_test)));

t1=tic;
xippmex('signal',analchans_test,'30ksps',ones(1,length(analchans_test)));
enabling_analog = toc(t1);

t1=tic;
xippmex('signal',micchans_test,'raw',ones(1,length(micchans_test)));
enabling_micros = toc(t1);


timeZero1 = xippmex('time'); %saving NIP clock
pause(2)
[test1_data_m, test1_time_m]  = xippmex('cont',micchans,6000,'raw',timeZero1);
[test1_data_anal, test1_time_anal]  = xippmex('cont',analchans,6000,'30ksps',timeZero1);

t1=tic;
xippmex('signal',analchans_test,'30ksps',zeros(1,length(analchans_test)));
disabling_analog = toc(t1);

t1=tic;
xippmex('signal',micchans_test,'raw',zeros(1,length(micchans_test)));
disabling_micros = toc(t1);

%% ERROR CONT TEST
timeZero2 = xippmex('time'); %saving NIP clock
pause(2)
[test1_data_m2, test1_time_m2]  = xippmex('cont',micchans,6000,'raw',timeZero2);
test1_data_mic2

%% DIGIN TEST (connect photodiode to analoginput)
disp('DIGIN config and test...')
xippmex('digin', 'bit-change', 1);
for i=1:100
    pause(2)
    [count, timestamps, events] = xippmex('digin');
    count
    if count>0
        break
    end
end

%% TRELLIS CONTROL TEST ('Enable Remote Control' must be selected in Trellis File Save)
ipend = 18; %or maybe 17
operator = xippmex('addoper', ipend);
trial_descriptor0 = xippmex('trial')

trial_descriptor1 = xippmex('trial','recording', operator);
pause(5)
trial_descriptor2 = xippmex('trial','stopped', operator);

%% TRELLIS CONTROL TEST ('Enable Remote Control' must be selected in Trellis File Save)
ip_end = 18; %or maybe 17
operator = xippmex('addoper', ip_end)
trial_descriptor0 = xippmex('trial')

trial_descriptor1 = xippmex('trial','recording', operator);
pause(5)
trial_descriptor2 = xippmex('trial','stopped', operator);

%% SAVING
disp('Saving test_xippmex_results.mat ...')
save('test2_xippmex_results')
disp('DONE')
%% CLOSE
disp('Closing xippmex...')
xippmex('close');
disp('DONE')
