P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(P, 'real_response_analysis'));
addpath(fullfile(P, 'responses'));
addpath(genpath(fullfile(P, 'codes_emu')));
addpath(genpath(fullfile(P, 'useful_functions')));
addpath(genpath(fullfile(P, 'batch_files')));
R = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368';
src = {'/mnt/data0/test py pt10emu7/examples/split_no_refract_viol_sd1/times_2026-10-05_15-59-40/times_mRFUS07 raw_368.mat', ...
       '/mnt/data0/test py pt10emu7/examples/split_no_refract_viol_sd2/times_2026-10-05_16-00-32/times_mRFUS07 raw_368.mat'};
names = {'python_rescue_no_refract_sd1', 'python_rescue_no_refract_sd2'};
for k = 1:2
    arm = build_python_arm_from_times('root', R, 'channel', 368, 'exp_type', 'RSVP_online', ...
        'arm_name', names{k}, 'python_times_file', src{k});
    fprintf('CONVERTED %s -> %s\n', names{k}, arm.dir);
end
