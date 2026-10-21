function resp_allfit(folder, ch, run, wc_run)
% Response metrics (all stimuli) for the ks_allfit sort (KS selection on all
% spikes, mask applied at clustering), in the same scratch session root as the
% KS/GMM test; for the 2026-10-13 runs also the as-run sort, wave_clus and GT.
P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(P, 'responses'), fullfile(P, 'real_response_analysis'), fullfile(P, 'validation_pipeline'));
addpath(genpath(fullfile(P, 'codes_emu')), genpath(fullfile(P, 'useful_functions')));
src = fullfile('/mnt/data0/MCWs Sessions/matlab validation/single channel', folder);
root = fullfile('/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad/gmm_test/resp', sprintf('ch%d', ch));
if ~exist(root, 'dir'), mkdir(root); end
for f = {'NSx.mat', 'NSX_TimeStamps.mat', 'stimulus.mat', 'finalevents.mat', 'experiment_properties_online3.mat', 'pre_processing_info.mat'}
    if exist(fullfile(src, f{1}), 'file') && ~exist(fullfile(root, f{1}), 'file'), copyfile(fullfile(src, f{1}), root); end
end
if ~exist(fullfile(root, 'pics_used'), 'dir'), system(sprintf('ln -s "%s" "%s"', fullfile(src, 'pics_used'), fullfile(root, 'pics_used'))); end
exp_type = getfield(load(fullfile(src, 'hand selected', 'grapes_blanks.mat'), 'exp_type'), 'exp_type'); %#ok<GFLD>
t = dir(fullfile('/mnt/data0/recluster_runs', run, 'gmm_feature_test', 'ks_allfit', 'second_pass_*', '*', 'times_*', 'times_*.mat'));
D = fullfile(root, 'python_ks_allfit');
if ~exist(fullfile(D, 'grapes_blanks_rescue.mat'), 'file')
    build_python_arm_from_times('root', root, 'channel', ch, 'exp_type', exp_type, 'arm_name', 'python_ks_allfit', 'python_times_file', fullfile(t(end).folder, t(end).name));
end
jobs = {'python_ks_allfit', 'grapes_blanks_rescue.mat'};
links = {};
if ~exist(fullfile(root, 'python_ks'), 'dir')            % 2026-10-13 runs: as-run sort = the KS control
    links(end+1, :) = {'python_ks', fullfile(src, 'python_recluster_20261013_second_pass', 'grapes_blanks_rescue.mat'), 'grapes_blanks_rescue.mat'};
end
if ~exist(fullfile(root, 'GT'), 'dir'), links(end+1, :) = {'GT', fullfile(src, 'hand selected', 'grapes_blanks.mat'), 'grapes_blanks.mat'}; end
for w = cellstr(wc_run)
    if ~isempty(w{1}) && ~exist(fullfile(root, w{1}), 'dir'), links(end+1, :) = {w{1}, fullfile(src, w{1}, 'grapes_blanks.mat'), 'grapes_blanks.mat'}; end %#ok<AGROW>
end
for i = 1:size(links, 1)
    L = fullfile(root, links{i, 1}); mkdir(L);
    system(sprintf('ln -s "%s" "%s"', links{i, 2}, fullfile(L, links{i, 3})));
    jobs(end+1, :) = {links{i, 1}, links{i, 3}}; %#ok<AGROW>
end
old = pwd;
for j = 1:size(jobs, 1)
    cd(fullfile(root, jobs{j, 1})); characterize_grapes_responses('channels2plot', ch, 'session_root', root, 'grapes_files', jobs(j, 2));
end
cd(old);
fprintf('resp_allfit ch%d done\n', ch);
end
