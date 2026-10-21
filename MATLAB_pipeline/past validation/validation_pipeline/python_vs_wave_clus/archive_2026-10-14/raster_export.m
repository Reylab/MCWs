function raster_export()
% RASTER_EXPORT  Per-stimulus response rows for every GT unit of the tuned
% trial (channels_tuned.json), one row per (unit, sort, stimulus), from the
% data_table.mat each validation's raster stage saved:
%   GT              validation_gt_vs_<tuned>/stimulus_rasters/gt
%   python as run   validation_gt_vs_<python_before>/stimulus_rasters/python_sd1
%   python tuned    validation_gt_vs_<tuned>/stimulus_rasters/python_sd1
%   wave_clus       <wc>/validation_gt_vs_<wc> (or the channel-level copy)/stimulus_rasters/<wc>
% Each sort's cluster is its own GT match (the validation's arm class).
% -> <out_dir>/raster/per_stim.csv   (read by raster_compare.py)
here = fileparts(mfilename('fullpath'));
C = jsondecode(fileread(fullfile(here, 'channels_tuned.json')));
out = fullfile(C.out_dir, 'raster'); if ~exist(out, 'dir'), mkdir(out); end
keep = {'stim_number','stim_name','ntrials','zscore','median_post','mean_post','mu_base','std_base','p_value_sign', ...
        'p_test','good_lat','onset','dura','mean_fr_base_hz','mean_fr_post_hz','resp_delta_fr_hz','frac_active_trials', ...
        'first_spk_lat_med','auroc','p_ranksum','peak_ifr','n_response','cluster_total'};
rows = {};
for u = C.units'
    root = fullfile(C.single_channel, u.folder);
    w = cellstr(u.wave_clus); w = w{1};
    wcv = fullfile(root, w, ['validation_gt_vs_' w u.tok]);
    if ~exist(fullfile(wcv, 'stimulus_rasters'), 'dir'), wcv = fullfile(root, ['validation_gt_vs_' w u.tok]); end
    src = {'GT',            fullfile(root, ['validation_gt_vs_' u.python_arm u.tok], 'stimulus_rasters', 'gt'); ...
           'python as run', fullfile(root, ['validation_gt_vs_' u.python_before u.tok], 'stimulus_rasters', 'python_sd1'); ...
           'python tuned',  fullfile(root, ['validation_gt_vs_' u.python_arm u.tok], 'stimulus_rasters', 'python_sd1'); ...
           'wave_clus',     fullfile(wcv, 'stimulus_rasters', w)};
    for s = 1:size(src, 1)
        f = fullfile(src{s, 2}, 'data_table.mat');
        if ~exist(f, 'file'), fprintf('MISSING %s %s: %s\n', u.id, src{s, 1}, f); continue; end
        T = load(f); T = T.data_table;
        k = keep(ismember(keep, T.Properties.VariableNames));
        T = T(:, k);
        T.unit = repmat(string(u.id), height(T), 1);
        T.sort = repmat(string(src{s, 1}), height(T), 1);
        T.run = repmat(string(ternary(s == 4, w, ternary(s == 2, u.python_before, ternary(s == 3, u.python_arm, 'hand selected')))), height(T), 1);
        rows{end+1} = T; %#ok<AGROW>
        fprintf('%s %-14s %3d stimuli  %s\n', u.id, src{s, 1}, height(T), f);
    end
end
writetable(vertcat(rows{:}), fullfile(out, 'per_stim.csv'));
fprintf('-> %s\n', fullfile(out, 'per_stim.csv'));
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
