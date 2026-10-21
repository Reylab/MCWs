function resp_ladder(ch)
% Loosening ladder, responses on each GT unit's configured stimulus list.
% Per channel (scratch session root, copies of the session files):
%   python_width / python_mild / python_within  grapes built from the saved trial sorts
%   python_as_run                                link to the channel's python arm grapes
%   wc_<run>, GT                                 links to wave_clus / hand-selected grapes
% then per unit (channels.json) and sort: characterize_grapes_responses with
% stim_list = validation_sessions(unit.session).stim_list -> <sort>/resp_<unit>.mat
P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(P, 'responses'), fullfile(P, 'real_response_analysis'), fullfile(P, 'validation_pipeline'));
addpath(genpath(fullfile(P, 'codes_emu')), genpath(fullfile(P, 'useful_functions')));
C = jsondecode(fileread(fullfile(P, 'validation_pipeline', 'python_vs_wave_clus', 'channels.json')));
U = C.units([C.units.ch] == ch & arrayfun(@(u) isempty(u.python_class) && isempty(u.wc_class), C.units)');
src = fullfile(C.single_channel, U(1).folder);
root = fullfile('/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad/gmm_test/resp_ladder', sprintf('ch%d', ch));
if ~exist(root, 'dir'), mkdir(root); end
for f = {'NSx.mat', 'NSX_TimeStamps.mat', 'stimulus.mat', 'finalevents.mat', 'experiment_properties_online3.mat', 'pre_processing_info.mat'}
    if exist(fullfile(src, f{1}), 'file') && ~exist(fullfile(root, f{1}), 'file'), copyfile(fullfile(src, f{1}), root); end
end
if ~exist(fullfile(root, 'pics_used'), 'dir'), system(sprintf('ln -s "%s" "%s"', fullfile(src, 'pics_used'), fullfile(root, 'pics_used'))); end
exp_type = getfield(load(fullfile(src, 'hand selected', 'grapes_blanks.mat'), 'exp_type'), 'exp_type'); %#ok<GFLD>
sorts = {};
for tr = {'width', 'mild', 'within'}
    t = dir(fullfile(C.recluster_runs, U(1).run, ['tuned_' tr{1}], ['second_pass_' tr{1}], '*', 'times_*', 'times_*.mat'));
    D = fullfile(root, ['python_' tr{1}]);
    if ~exist(fullfile(D, 'grapes_blanks_rescue.mat'), 'file')
        build_python_arm_from_times('root', root, 'channel', ch, 'exp_type', exp_type, 'arm_name', ['python_' tr{1}], 'python_times_file', fullfile(t(end).folder, t(end).name));
    end
    sorts(end+1, :) = {['python_' tr{1}], 'grapes_blanks_rescue.mat'}; %#ok<AGROW>
end
links = {'python_as_run', fullfile(src, U(1).python_arm, 'grapes_blanks_rescue.mat'), 'grapes_blanks_rescue.mat'; ...
         'GT', fullfile(src, 'hand selected', 'grapes_blanks.mat'), 'grapes_blanks.mat'};
for w = unique(arrayfun(@(u) string(u.wave_clus{1}), U))'
    links(end+1, :) = {['wc_' char(w)], fullfile(src, char(w), 'grapes_blanks.mat'), 'grapes_blanks.mat'}; %#ok<AGROW>
end
for i = 1:size(links, 1)
    L = fullfile(root, links{i, 1}); if ~exist(L, 'dir'), mkdir(L); end
    if ~exist(fullfile(L, links{i, 3}), 'file'), system(sprintf('ln -s "%s" "%s"', links{i, 2}, fullfile(L, links{i, 3}))); end
    sorts(end+1, :) = {links{i, 1}, links{i, 3}}; %#ok<AGROW>
end
old = pwd;
for u = U'
    stims = validation_sessions(u.session).stim_list;
    for s = 1:size(sorts, 1)
        if startsWith(sorts{s, 1}, 'wc_') && ~strcmp(sorts{s, 1}, ['wc_' u.wave_clus{1}]), continue; end
        cd(fullfile(root, sorts{s, 1}));
        characterize_grapes_responses('channels2plot', ch, 'session_root', root, 'grapes_files', sorts(s, 2), ...
            'stim_list', stims, 'outfile', sprintf('resp_%s.mat', u.id));
    end
end
cd(old);
fprintf('resp_ladder ch%d done\n', ch);
end
