function create_stimulus_struct_AV(which_type)
%creates the stimulus structure for audio or video events, it takes the stimulus
%names form ImageNames.txt and the trial order from finalevents_audio.mat or finalevents_video.mat

% [order]=textread('TrialOrder.txt','%d');
[names] = textread('ImageNames.txt','%s','delimiter','\n');
eval(['load finalevents_' which_type '.mat'])
uni_val=unique(events(:,1));

for i=1:length(uni_val)
    trial_num=find(events(:,1) ==uni_val(i));  
    stimulus(i).ID = i;            
    stimulus(i).trial_list=trial_num;
    stimulus(i).name = names{uni_val(i)};
    stimulus(i).answer = [];
    clear trial_num;
end

eval(['save stimulus_' which_type ' stimulus;'])


