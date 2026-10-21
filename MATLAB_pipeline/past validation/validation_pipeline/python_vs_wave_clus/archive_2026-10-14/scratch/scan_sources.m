% For every single-channel folder: its times/grapes class counts, its source
% session in sEEG_DATA (matched by stimulus onsets), and every times/grapes
% file for that channel in the source session. READ ONLY.
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
SE = '/mnt/data0/sEEG_DATA';
out_csv = fullfile(fileparts(mfilename('fullpath')), 'scan_sources_v2.csv');

D = dir(SC); D = D([D.isdir] & ~startsWith({D.name}, '.'));
rows = {};
fp_cache = containers.Map();
for d = D'
    root = fullfile(SC, d.name);
    tok = regexp(d.name, 'pt (\d+) emu (\d+) ch (\d+)', 'tokens', 'once');
    if isempty(tok), continue; end
    [pt, emu, ch] = deal(str2double(tok{1}), str2double(tok{2}), str2double(tok{3}));
    nc5 = dir(fullfile(root, '*.NC5'));
    nm = ''; if ~isempty(nc5), [~, nm] = fileparts(nc5(1).name); end
    % --- this folder
    hs = fullfile(root, 'hand selected'); if ~exist(hs, 'dir'), hs = root; end
    t = dir(fullfile(hs, ['times_' nm '.mat']));          % the file named for THIS channel
    if isempty(t), t = dir(fullfile(hs, ['times_' nm '*.mat'])); end
    t_name = ''; t_counts = [];
    if ~isempty(t)
        t_name = t(1).name;
        T = load(fullfile(hs, t_name), 'cluster_class');
        t_counts = counts(T.cluster_class(:,1));
    end
    g = dir(fullfile(hs, '*grapes*.mat')); g = g(~contains({g.name}, {'response_metrics', 'circ'}));   % circ-shift grapes never used
    g_desc = {}; g_match = false; g_has = false; g_counts_all = {};
    for gi = 1:numel(g)
        c = grapes_counts(fullfile(hs, g(gi).name), ch);
        if isempty(c), g_desc{end+1} = sprintf('%s: no chan%d', g(gi).name, ch); continue; end %#ok<SAGROW>
        g_has = true; g_counts_all{end+1} = c; %#ok<SAGROW>
        g_desc{end+1} = sprintf('%s %s', g(gi).name, mat2str(c)); %#ok<SAGROW>
        if isequal(c, t_counts), g_match = true; end
    end
    % --- source session: same stimulus onsets, same patient
    src = '';
    fe = fullfile(root, 'finalevents.mat');
    if exist(fe, 'file')
        fp = onset_fp(fe);
        S = dir(fullfile(SE, sprintf('MCW-FH_%03d', pt), 'EMU', '*', 'finalevents.mat'));
        for s = S'
            k = fullfile(s.folder, s.name);
            if ~isKey(fp_cache, k), fp_cache(k) = onset_fp(k); end
            if isequal(fp_cache(k), fp) && ~contains(s.folder, 'BAD'), src = s.folder; break; end   % never the BAD session
        end
    end
    % --- files for this channel in the source session
    s_times = {}; s_grapes = {}; times_for_grapes = {}; grapes_for_times = {};
    if ~isempty(src) && ~isempty(nm)
        st = dir(fullfile(src, ['times_' nm '*.mat']));
        for k = 1:numel(st)
            try
                T2 = load(fullfile(src, st(k).name), 'cluster_class');
                c = counts(T2.cluster_class(:,1));
                s_times{end+1} = sprintf('%s %s', st(k).name, mat2str(c)); %#ok<SAGROW>
                if any(cellfun(@(x) isequal(x, c), g_counts_all)), times_for_grapes{end+1} = st(k).name; end %#ok<SAGROW>
            catch
                s_times{end+1} = sprintf('%s (unreadable)', st(k).name); %#ok<SAGROW>
            end
        end
        sg = dir(fullfile(src, '*grapes*.mat')); sg = sg(~contains({sg.name}, {'response_metrics', 'circ'}));
        for k = 1:numel(sg)
            if sg(k).bytes > 3e9, s_grapes{end+1} = sprintf('%s (skipped, %.1f GB)', sg(k).name, sg(k).bytes/1e9); continue; end %#ok<SAGROW>
            c = grapes_counts(fullfile(src, sg(k).name), ch);
            if isempty(c), continue; end
            s_grapes{end+1} = sprintf('%s %s', sg(k).name, mat2str(c)); %#ok<SAGROW>
            if isequal(c, t_counts), grapes_for_times{end+1} = sg(k).name; end %#ok<SAGROW>
        end
    end
    rows(end+1, :) = {pt, emu, ch, nm, d.name, strrep(hs, [root filesep], ''), t_name, mat2str(t_counts), ...
        strjoin(g_desc, ' ; '), double(g_has), double(g_match), strrep(src, [SE '/'], ''), ...
        strjoin(s_times, ' ; '), strjoin(s_grapes, ' ; '), strjoin(times_for_grapes, ' ; '), strjoin(grapes_for_times, ' ; ')}; %#ok<SAGROW>
    fprintf('%s: match %d, has_chan %d, src %s\n', d.name, g_match, g_has, strrep(src, [SE '/'], ''));
end
T = cell2table(rows, 'VariableNames', {'pt','emu','ch','chan_name','folder','gt_location','times_file','times_counts', ...
    'grapes_in_folder','grapes_has_channel','times_grapes_match','seeg_session','seeg_times_files','seeg_grapes_files', ...
    'seeg_times_matching_folder_grapes','seeg_grapes_matching_folder_times'});
writetable(T, out_csv);
fprintf('wrote %s\n', out_csv);

function c = counts(lab)
    lab = lab(:);
    if isempty(lab) || max(lab) < 1, c = []; return; end
    c = arrayfun(@(k) nnz(lab == k), 1:max(lab));
end

function c = grapes_counts(f, ch)
    c = [];
    try
        R = load(f, 'rasters');
    catch
        return
    end
    cf = sprintf('chan%d', ch);
    if ~isfield(R, 'rasters') || ~isfield(R.rasters, cf) || ~isfield(R.rasters.(cf), 'details'), return; end
    dd = R.rasters.(cf).details;
    fn = fieldnames(dd); fn = fn(~cellfun(@isempty, regexp(fn, '^class\d+$', 'once')));
    if isempty(fn), return; end
    c = zeros(1, numel(fn));
    for k = 1:numel(fn), c(str2double(fn{k}(6:end))) = dd.(fn{k}); end
end

function fp = onset_fp(f)
    F = load(f, 'pics_onset');
    p = F.pics_onset;
    if ~iscell(p), p = {p}; end
    fp = cellfun(@(x) [numel(x), sum(x(isfinite(x)))], p, 'UniformOutput', false);
end
