Pp = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(Pp, 'real_response_analysis')); addpath(fullfile(Pp, 'responses'));
addpath(genpath(fullfile(Pp, 'codes_emu'))); addpath(genpath(fullfile(Pp, 'useful_functions'))); addpath(genpath(fullfile(Pp, 'batch_files')));
R = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368';
src = '/mnt/data0/python_p10EMU07CH368/times_2026-10-06_13-46-07/times_mRFUS07 raw_368.mat';
assert(~exist(fullfile(R, 'python_recluster_20261006_sd2'), 'dir'));
arm = build_python_arm_from_times('root', R, 'channel', 368, 'exp_type', 'RSVP_online', ...
    'arm_name', 'python_recluster_20261006_sd2', 'python_times_file', src);
fprintf('CONVERTED -> %s\n', arm.dir);
