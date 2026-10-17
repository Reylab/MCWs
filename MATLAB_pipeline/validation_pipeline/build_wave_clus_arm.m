function arm_dir = build_wave_clus_arm(root, ch, wc_dir, arm_name)
% BUILD_WAVE_CLUS_ARM  Make a validation arm folder from a wave_clus result.
%
%   build_wave_clus_arm(root, ch, wc_dir, arm_name)
%   build_wave_clus_arm(root, ch, wc_dir)            % in place, in wc_dir
%     root     - channel session folder (NSx.mat, finalevents.mat, ...)
%     ch       - channel number
%     wc_dir   - folder holding wave_clus's times_<name>.mat
%     arm_name - arm folder created under root; '' or omitted = in place:
%                the fields below are APPENDED to wc_dir's own times file
%                (cluster_class and everything else in it unchanged) and
%                grapes_blanks.mat is written into wc_dir
%
% The arm's times file is wave_clus's with the fields the validation stages
% expect from the Python layout: index (= spike times, ms) and an all-false
% quarantine (wave_clus quarantines nothing), forced / features when missing.
% grapes_blanks.mat is built from it with do_structure_sorted_BCM_online3
% (use_blanks=1, circshiftblanks=0) in a clean folder and checked against the
% times counts.

here = fileparts(mfilename('fullpath'));
P = fileparts(here);
addpath(genpath(fullfile(P, 'codes_emu'))); addpath(genpath(fullfile(P, 'useful_functions')));
N = load(fullfile(root, 'NSx.mat'), 'NSx');
nm = N.NSx([N.NSx.chan_ID] == ch).output_name;
src = fullfile(wc_dir, ['times_' nm '.mat']);
T = load(src);
n = size(T.cluster_class, 1);
add = struct();
add.index = T.cluster_class(:, 2)';
z = false(1, n);
add.quarantine = struct('all', z, 'refractory_period', z, 'bundle_collision', z, 'within_channel', z, ...
    'rescued', uint8(zeros(n, 1)));
if ~isfield(T, 'forced') || numel(T.forced) ~= n, add.forced = false(n, 1); end
if ~isfield(T, 'features') && isfield(T, 'inspk'), add.features = T.inspk; end

if nargin < 4 || isempty(arm_name)              % in place
    arm_dir = wc_dir;
    tf = src;
    add.source_note = sprintf('wave_clus result; index + all-false quarantine appended for validation %s', ...
        datestr(now));
    save(tf, '-struct', 'add', '-append');
else
    for f = fieldnames(add)', T.(f{1}) = add.(f{1}); end
    T.source_note = sprintf('wave_clus result %s, copied %s; index + all-false quarantine added for validation', ...
        src, datestr(now));
    arm_dir = fullfile(root, arm_name);
    if ~exist(arm_dir, 'dir'), mkdir(arm_dir); end
    tf = fullfile(arm_dir, ['times_' nm '.mat']);
    save(tf, '-struct', 'T', '-v7.3');
end

% grapes_blanks from this times file, built in a clean folder
b = fullfile(arm_dir, 'build_tmp');
if exist(b, 'dir'), rmdir(b, 's'); end
mkdir(b); copyfile(tf, b);
old = pwd; c0 = onCleanup(@() cd(old));
cd(root);
do_structure_sorted_BCM_online3(ch, true, false, false, 'times_dir', b, 'spike_dir', b);
cd(old);
G = load(fullfile(b, 'grapes_blanks.mat'), 'rasters');
lab = T.cluster_class(:, 1);
tc = arrayfun(@(c) nnz(lab == c), 1:max(lab));
d = G.rasters.(sprintf('chan%d', ch)).details;
gc = arrayfun(@(c) d.(sprintf('class%d', c)), 1:numel(tc));
assert(isequal(tc, gc), 'grapes_blanks counts %s ~= times %s', mat2str(gc), mat2str(tc));
% a grapes built from scratch has no exp_type (normally set by the mu step):
% take it from the session's hand-selected grapes
Gf = load(fullfile(b, 'grapes_blanks.mat'));
src = dir(fullfile(root, 'hand selected', '*grapes*.mat'));
for k = 1:numel(src)
    w = whos('-file', fullfile(src(k).folder, src(k).name));
    if any(strcmp({w.name}, 'exp_type'))
        x = load(fullfile(src(k).folder, src(k).name), 'exp_type'); Gf.exp_type = x.exp_type; break
    end
end
assert(isfield(Gf, 'exp_type'), 'no exp_type found in %s/hand selected', root);
save(fullfile(arm_dir, 'grapes_blanks.mat'), '-struct', 'Gf');
rmdir(b, 's');
fprintf('[wave_clus arm] %s: %d spikes, classes %s -> %s\n', nm, n, mat2str(tc), arm_dir);
end
