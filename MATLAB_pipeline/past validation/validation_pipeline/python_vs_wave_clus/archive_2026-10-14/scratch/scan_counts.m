% Every candidate times / grapes file per channel, with its class counts.
% READ ONLY. Excludes circ files and BAD sessions. Output: scan_counts.csv
% (one row per (channel, file)).
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
SE = '/mnt/data0/sEEG_DATA';
here = fileparts(mfilename('fullpath'));
L = readtable(fullfile(here, 'channels.csv'));     % pt, emu, ch
SCD = dir(SC); SCD = SCD([SCD.isdir] & ~startsWith({SCD.name}, '.'));
rows = {};
for i = 1:height(L)
    pt = L.pt(i); emu = L.emu(i); ch = L.ch(i);
    key = sprintf('pt %d emu %d ch %d', pt, emu, ch);
    hit = SCD(~cellfun(@isempty, regexp({SCD.name}, ['(^|\s)' key '(\s|$|\()'], 'once')));
    places = {};
    for h = hit'
        root = fullfile(SC, h.name);
        places{end+1} = {['folder:' h.name], root}; %#ok<SAGROW>
        sub = dir(root); sub = sub([sub.isdir] & startsWith({sub.name}, 'hand selected'));
        for s = sub', places{end+1} = {['folder:' h.name '/' s.name], fullfile(root, s.name)}; end %#ok<SAGROW>
    end
    ses = dir(fullfile(SE, sprintf('MCW-FH_%03d', pt), 'EMU', sprintf('EMU-%03d_*', emu)));
    ses = ses([ses.isdir] & ~contains({ses.name}, 'BAD'));
    for s = ses', places{end+1} = {['seeg:' s.name], fullfile(s.folder, s.name)}; end %#ok<SAGROW>
    for p = places
        loc = p{1}{1}; d = p{1}{2};
        is_seeg = startsWith(loc, 'seeg:');
        f = dir(fullfile(d, '*.mat'));
        for k = 1:numel(f)
            nm = f(k).name;
            if contains(lower(nm), 'circ') || contains(nm, 'response_metrics') || contains(nm, 'spikes'), continue; end
            if is_seeg && ~(contains(nm, sprintf('raw_%d', ch)) || contains(lower(nm), 'grapes')), continue; end
            fp = fullfile(d, nm);
            if f(k).bytes > 3e9
                rows(end+1, :) = {pt, emu, ch, loc, nm, 'skipped (>3GB)', ''}; %#ok<SAGROW>
                continue
            end
            try, w = whos('-file', fp); catch, continue; end
            v = {w.name};
            if any(strcmp(v, 'cluster_class')) && contains(nm, sprintf('raw_%d', ch)) || ...
               any(strcmp(v, 'cluster_class')) && ~is_seeg && ~contains(nm, 'raw_')
                T = load(fp, 'cluster_class');
                lab = T.cluster_class(:,1);
                c = arrayfun(@(x) nnz(lab == x), 1:max([lab; 0]));
                rows(end+1, :) = {pt, emu, ch, loc, nm, 'times', mat2str(c)}; %#ok<SAGROW>
            elseif any(strcmp(v, 'rasters'))
                R = load(fp, 'rasters');
                cf = sprintf('chan%d', ch);
                if ~isfield(R.rasters, cf) || ~isfield(R.rasters.(cf), 'details'), continue; end
                dd = R.rasters.(cf).details;
                fn = fieldnames(dd); fn = fn(~cellfun(@isempty, regexp(fn, '^class\d+$', 'once')));
                c = zeros(1, numel(fn));
                for q = 1:numel(fn), c(str2double(fn{q}(6:end))) = dd.(fn{q}); end
                rows(end+1, :) = {pt, emu, ch, loc, nm, 'grapes', mat2str(c)}; %#ok<SAGROW>
            end
        end
    end
    fprintf('%s: %d places, %d files so far\n', key, numel(places), size(rows, 1));
end
writetable(cell2table(rows, 'VariableNames', {'pt','emu','ch','location','file','kind','counts'}), fullfile(here, 'scan_counts.csv'));
disp('done');
