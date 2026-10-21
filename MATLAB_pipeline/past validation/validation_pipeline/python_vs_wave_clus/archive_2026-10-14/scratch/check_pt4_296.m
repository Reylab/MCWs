addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/real_response_analysis');
% --- ch296 in the BAD session
B = '/mnt/data0/sEEG_DATA/MCW-FH_010/EMU/EMU-007_subj-MCW-FH_010_task-RSVPDynamicScr_run-02 BAD';
fprintf('== pt10 emu7 ch296 (BAD session), expected class 2 = 2054\n');
T = load(fullfile(B, 'times_mLFUS08 raw_296.mat'), 'cluster_class');
fprintf('  times_mLFUS08 raw_296.mat: %s\n', mat2str(arrayfun(@(c) nnz(T.cluster_class(:,1)==c), 1:max(T.cluster_class(:,1)))));
for g = {'grapes_blanks.mat', 'grapes.mat', 'Copy_of_grapes.mat', 'grapes NO.mat', 'grapes_error.mat'}
    try
        R = load(fullfile(B, g{1}), 'rasters'); d = R.rasters.chan296.details;
        f = fieldnames(d); f = f(~cellfun(@isempty, regexp(f, '^class\d+$', 'once')));
        c = zeros(1, numel(f)); for k = 1:numel(f), c(str2double(f{k}(6:end))) = d.(f{k}); end
        fprintf('  %s: %s\n', g{1}, mat2str(c));
    catch ME, fprintf('  %s: %s\n', g{1}, ME.message); end
end
% --- pt4 emu7: grapes raster class vs times class, by spike time
R4 = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 4 emu 7 ch 322';
SE4 = '/mnt/data0/sEEG_DATA/MCW-FH_004/EMU/EMU-007_subj-MCW-FH_004_task-RSVPdynamic_scr_run-02';
S = load(fullfile(R4, 'stimulus.mat')); F = load(fullfile(R4, 'finalevents.mat')); E = load(fullfile(R4, 'experiment_properties_online3.mat'));
N = min(numel(S.stimulus), numel(E.scr_end_cell));
allstim = unique(cell2mat(cellfun(@(c) c.pics2use(:)', E.scr_config_cell(1:N), 'UniformOutput', false)));
om = build_all_stim_onset_map(S, F, N, allstim, E);
cases = {322, 'mLHIPH02 raw_322', fullfile(R4, 'grapes.mat'), 3
         261, 'mLPBTG05 raw_261', fullfile('/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 4 emu 7 ch 261 check', 'grapes.mat'), 1};
for i = 1:size(cases, 1)
    [ch, nm, gf, cls] = cases{i, :};
    G = load(gf, 'rasters'); st = G.rasters.(sprintf('chan%d', ch)).(sprintf('class%d', cls)).stim;
    rt = [];
    for s = allstim
        if s > numel(st) || isempty(st{s}) || ~isKey(om, s), continue; end
        tr = st{s}; if ~iscell(tr), tr = num2cell(tr, 2); end
        O = om(s); if numel(O) ~= numel(tr), continue; end
        for k = 1:numel(tr), r = tr{k}(:)'; r = r(isfinite(r) & r >= 0 & r < 1000); rt = [rt, r + O(k)]; end %#ok<AGROW>
    end
    rt = unique(round(rt(:) * 100) / 100);
    T = load(fullfile(SE4, ['times_' nm '.mat']), 'cluster_class'); cc = T.cluster_class;
    fprintf('\n== pt4 emu7 ch%d: grapes class %d (n=%d) rebuilt -> %d spike times (0-1000 ms after onsets)\n', ch, cls, G.rasters.(sprintf('chan%d', ch)).details.(sprintf('class%d', cls)), numel(rt));
    fprintf('  times file class counts: %s\n', mat2str(arrayfun(@(c) nnz(cc(:,1)==c), 0:max(cc(:,1)))));
    for c = 0:max(cc(:,1))
        t = unique(cc(cc(:,1) == c, 2)); if numel(t) < 2, continue; end
        j = interp1(t, 1:numel(t), rt, 'nearest', 'extrap');
        fprintf('  %% of rebuilt grapes class %d spikes in times class %d: %.1f%%\n', cls, c, 100 * mean(abs(t(j) - rt) <= 0.5));
    end
    % reverse: share of the times class's in-window spikes that the grapes class has
    t = unique(cc(cc(:,1) == cls, 2)); onset = sort(cell2mat(om.values));
    inwin = false(size(t)); for k = 1:numel(onset), inwin = inwin | (t >= onset(k) & t < onset(k) + 1000); end
    tw = t(inwin);
    j = interp1(rt, 1:numel(rt), tw, 'nearest', 'extrap');
    fprintf('  times class %d spikes 0-1000 ms after onsets: %d; found in rebuilt grapes class %d: %.1f%%\n', cls, numel(tw), cls, 100 * mean(abs(rt(j) - tw) <= 0.5));
end
