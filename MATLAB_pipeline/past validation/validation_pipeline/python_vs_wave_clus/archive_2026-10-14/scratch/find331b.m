addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/real_response_analysis');
R = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 9 emu 3 ch 331 check';
H = fullfile(R, 'hand selected');
SE = '/mnt/data0/sEEG_DATA/MCW-FH_009/EMU/EMU-003_subj-MCW-FH_009_task-RSVPDynamicScr_run-01';
OLD = '/mnt/data0/MCWs Sessions/matlab validation/analysis trials old/new trial pt 9 emu 3 ch 331';
% --- grapes-type files: class fields and counts
gf = {fullfile(H,'grapes_blanks.mat'), fullfile(H,'grapes.mat'), fullfile(H,'current.mat'), fullfile(H,'cop2.mat'), ...
      fullfile(H,'blanks_circ.mat'), fullfile(SE,'grapes_blanks.mat'), fullfile(SE,'Copy_of_grapes.mat'), ...
      fullfile(SE,'grapes_blanks_circ.mat'), fullfile(OLD,'RECLUSTER_VALIDATION_20260910_122564','handselect','grapes.mat')};
fprintf('--- grapes-type files (chan331 details)\n');
for i = []
    if ~exist(gf{i}, 'file'), continue; end
    G = load(gf{i}, 'rasters'); d = G.rasters.chan331.details;
    f = fieldnames(d); f = f(startsWith(f, 'class') & ~endsWith(f, 'rescue'));
    v = cellfun(@(x) d.(x), f);
    fprintf('%-95s %s\n', strrep(gf{i}, '/mnt/data0/', ''), strjoin(compose('%s=%d', string(f), v), ' '));
end
% --- rebuild grapes_blanks class 4 times (post-onset ticks, every stimulus)
S = load(fullfile(R,'stimulus.mat')); F = load(fullfile(R,'finalevents.mat')); E = load(fullfile(R,'experiment_properties_online3.mat'));
N = min(numel(S.stimulus), numel(E.scr_end_cell));
allstim = unique(cell2mat(cellfun(@(c) c.pics2use(:)', E.scr_config_cell(1:N), 'UniformOutput', false)));
om = build_all_stim_onset_map(S, F, N, allstim, E);
G = load(fullfile(H, 'grapes_blanks.mat'), 'rasters');
st = G.rasters.chan331.class4.stim;
rt = [];
for s = allstim
    if s > numel(st) || isempty(st{s}) || ~isKey(om, s), continue; end
    tr = st{s}; if ~iscell(tr), tr = num2cell(tr, 2); end
    O = om(s); if numel(O) ~= numel(tr), continue; end
    for k = 1:numel(tr)
        r = tr{k}(:)'; r = r(isfinite(r) & r >= 0 & r < 1000);   % first second after onset: no overlap with the next trial's pre-window
        rt = [rt, r + O(k)]; %#ok<AGROW>
    end
end
rt = unique(round(rt(:) * 100) / 100);
fprintf('\nrebuilt grapes_blanks class4: %d spike times (0-1000 ms after each onset only)\n', numel(rt));
% --- candidate times files
tf = {fullfile(H,'times_mLFUS02 raw_331.mat'), fullfile(SE,'times_mLFUS02 raw_331c.mat'), ...
      fullfile(SE,'times_mLFUS02 raw_331 not good probably.mat'), ...
      fullfile(OLD,'RECLUSTER_VALIDATION_20260910_122564','handselect','times_mLFUS02 raw_331.mat'), ...
      fullfile(OLD,'RECLUSTER_VALIDATION_20260910_122564','handselect','reconstructed_cc_mLFUS02 raw_331.mat')};
for s = 1:3
    d = dir(fullfile('/mnt/data0/MCWs Sessions/matlab validation/complete bundles', sprintf('Session %d', s), 'times_*', 'times_mLFUS02 raw_331.mat'));
    tf = [tf, fullfile({d.folder}, {d.name})]; %#ok<AGROW>
end
tf = tf(6:end);
for pat = {'/mnt/data0/MCWs Sessions/matlab validation/incomplete bundles/pt 9 emu 3 (2)/**/times_mLFUS02 raw_331*.mat', ...
           '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 9 emu 3 ch 331 check/**/times_mLFUS02 raw_331*.mat', ...
           '/mnt/data0/MCWs Sessions/matlab validation/analysis trials old/new trial pt 9 emu 3 ch 331/**/times_mLFUS02 raw_331*.mat', ...
           '/mnt/data0/MCWs Sessions/combinato/Session 1/**/times_mLFUS02 raw_331*.mat', ...
           [SE '/ch331*.mat'], [SE '/FR331.mat']}
    d = dir(pat{1});
    tf = [tf, fullfile({d.folder}, {d.name})]; %#ok<AGROW>
end
tf = unique(tf, 'stable');
skip = contains(tf, {'hand selected/times_', 'handselect/times_', 'python_recluster', 'raw_331c', 'not good probably'});
tf = tf(~skip);
fprintf('\n--- candidate times files: class sizes, and share of the rebuilt grapes class4 spikes found in each class (0.5 ms)\n');
for i = 1:numel(tf)
    if ~exist(tf{i}, 'file'), continue; end
    w = whos('-file', tf{i}); nm = {w.name};
    if ~any(strcmp(nm, 'cluster_class')), fprintf('%s: no cluster_class (%s)\n', strrep(tf{i}, '/mnt/data0/', ''), strjoin(nm, ' ')); continue; end
    T = load(tf{i}, 'cluster_class'); cc = T.cluster_class;
    K = max(cc(:,1));
    n = arrayfun(@(c) nnz(cc(:,1) == c), 0:K);
    frac = zeros(1, K + 1);
    for c = 0:K
        t = unique(cc(cc(:,1) == c, 2));
        if numel(t) < 2, continue; end
        j = interp1(t, 1:numel(t), rt, 'nearest', 'extrap');
        frac(c+1) = mean(abs(t(j) - rt) <= 0.5);
    end
    [b, bi] = max(frac);
    has = find(n == 3538) - 1;
    if ~any(n == 3538) && b < 0.9, fprintf('%s: classes %s, best %d%% (class %d)\n', strrep(tf{i}, '/mnt/data0/MCWs Sessions/matlab validation/', ''), mat2str(n), round(100*b), bi-1); continue; end
    fprintf('\n%s\n  n per class 0..%d: %s\n  %% of grapes class4 in each class: %s\n  best: class %d (%.0f%%, n=%d)%s\n', ...
        strrep(tf{i}, '/mnt/data0/', ''), K, mat2str(n), mat2str(round(100*frac)), bi-1, 100*b, n(bi), ...
        iff(~isempty(has), sprintf('   <-- class %s has exactly 3538', mat2str(has)), ''));
end
function v = iff(c, a, b), if c, v = a; else, v = b; end, end
