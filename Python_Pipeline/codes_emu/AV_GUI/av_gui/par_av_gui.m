function par=par_av_gui() 
 
 
par.mode = 'a'; % or 'v' 
par.folder_base = []; % [] for pwd 
par.tbeg = 0;
par.rec_length = 'end'; % or number of seconds from tbeg 
par.folder_resus = []; % place to save the final events and figures. [] for pwd 
par.frame_len = 20; % max chunk of data ploted in seconds 
 
par.channels = [2076]; 
par.audio_channel = 2129; %if int nc5 otherwise filename of a .mat to load
% par.audio_channel = 10245; %if int nc5 otherwise filename of a .mat to load

%parameters for mat audio file
par.audio_file = 'test.mat';
par.audio_sr = 30000; %in Hz
par.audio_variable = 'data';


par.show_lfp = false; %only for single channel 
 
%Firing rate parameters 
par.show_fr = true; 
par.sigma_gauss = 49.42; 
par.alpha_gauss = 3.035; %last value of gaussian 0.01 0.025 
% lenght of gaussian window = 2* alpha_gauss * sigma_gauss 

%Moving average parameters 
par.show_mav = true; 
par.window_len_mav = 20; %in seconds

%parameter for video mode, 'v' 
par.video_file = 'HIMYM_S03E05.mp4'; 
 
 
par.show_raster = true; 
% par.show_raster = false; 
par.classes = {[1,2]}; % or 'mu' for multi unit, 'all' for all classes or cell
                    % of the same length as channels, with the classes for
                    % each.
 
par.nrows = 5;  %number of rows in the matrix of frames in video mode 
par.fr_prev = 25; 
end 
 
%Shortcuts 
%========= 
 
% numerical minus('-'): max timescal, all the chunck (zoom out icon) 
% space bar:  pause/resume (||) 
% right_arrow: next chunk (>>) 
% left_arrow: prev chunck (<<) 
% return: Add mark 
% 1-9: choose that event 
 
 
 
