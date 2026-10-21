function resp_width334()
% ch334 GT1: python as run vs python width-25 trial vs wave_clus1 vs GT, scored identically
% (characterize_grapes_responses, GT1's stimulus list) in a scratch session root.
P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(P, 'responses'), fullfile(P, 'real_response_analysis'), fullfile(P, 'validation_pipeline'));
addpath(genpath(fullfile(P, 'codes_emu')), genpath(fullfile(P, 'useful_functions')));
src = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 6 emu 7 ch 334';
root = '/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad/gmm_test/resp_width/ch334';
if ~exist(root, 'dir'), mkdir(root); end
for f = {'NSx.mat', 'NSX_TimeStamps.mat', 'stimulus.mat', 'finalevents.mat', 'experiment_properties_online3.mat', 'pre_processing_info.mat'}
    if exist(fullfile(src, f{1}), 'file') && ~exist(fullfile(root, f{1}), 'file'), copyfile(fullfile(src, f{1}), root); end
end
if ~exist(fullfile(root, 'pics_used'), 'dir'), system(sprintf('ln -s "%s" "%s"', fullfile(src, 'pics_used'), fullfile(root, 'pics_used'))); end
c = validation_sessions('pt6emu7_ch334_class1'); stims = c.stim_list;
exp_type = getfield(load(fullfile(src, 'hand selected', 'grapes_blanks.mat'), 'exp_type'), 'exp_type'); %#ok<GFLD>
t = dir('/mnt/data0/recluster_runs/p06_EMU-007_ch334/tuned_width/second_pass_width/*/times_*/times_*.mat');
D = fullfile(root, 'python_width');
if ~exist(fullfile(D, 'grapes_blanks_rescue.mat'), 'file')
    build_python_arm_from_times('root', root, 'channel', 334, 'exp_type', exp_type, 'arm_name', 'python_width', 'python_times_file', fullfile(t(end).folder, t(end).name));
end
links = {'python_as_run', fullfile(src, 'python_recluster_20261013_second_pass', 'grapes_blanks_rescue.mat'), 'grapes_blanks_rescue.mat'; ...
         'wave_clus1', fullfile(src, 'wave_clus1', 'grapes_blanks.mat'), 'grapes_blanks.mat'; ...
         'GT', fullfile(src, 'hand selected', 'grapes_blanks.mat'), 'grapes_blanks.mat'};
for i = 1:size(links, 1)
    L = fullfile(root, links{i, 1}); if ~exist(L, 'dir'), mkdir(L); end
    if ~exist(fullfile(L, links{i, 3}), 'file'), system(sprintf('ln -s "%s" "%s"', links{i, 2}, fullfile(L, links{i, 3}))); end
end
old = pwd;
for s = {'python_width', 'grapes_blanks_rescue.mat'; 'python_as_run', 'grapes_blanks_rescue.mat'; 'wave_clus1', 'grapes_blanks.mat'; 'GT', 'grapes_blanks.mat'}'
    cd(fullfile(root, s{1})); characterize_grapes_responses('channels2plot', 334, 'session_root', root, 'grapes_files', {s{2}}, 'stim_list', stims);
end
cd(old);
end
