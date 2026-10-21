function ppt_redo_rasters(session, multi)
% PPT_REDO_RASTERS  Re-draw only the raster pages of one GT unit's for ppt report (e.g. after adding the
% stimulus pictures to <channel>/pics_used). Every other file in the report (waveform pages, venns, csv,
% metrics, decks) is backed up first and put back unchanged afterwards.
%   ppt_redo_rasters('pt4emu7_ch322_class3', false)
if nargin < 2, multi = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; root = cfg.root;
rep = fullfile(root, 'for ppt'); if multi, rep = fullfile(rep, sprintf('GT c%d', g)); end
M = load(fullfile(rep, sprintf('membership_ch%d.mat', cfg.ch)), 'classes');
d = dir(fullfile(root, 'python_recluster_*_second_pass')); py = d(end).name; wc = fullfile(root, 'wave_clus notch');

% 1. back up every non-raster file of the report
f = dir(fullfile(rep, '**', '*')); f = f(~[f.isdir]);
keep = ~arrayfun(@(x) is_raster(x.folder, x.name), f);
bk = tempname; for i = find(keep)'
    dst = fullfile(bk, erase(f(i).folder, rep)); if ~exist(dst, 'dir'), mkdir(dst); end
    copyfile(fullfile(f(i).folder, f(i).name), dst);
end

% 2. the three full comparison runs: re-draw their raster pages (their windows / numbers do not change)
for S = {fullfile(root, sprintf('validation_gt_vs_%s_class%d', py, g)), ...
         fullfile(wc, sprintf('validation_wave_clus_vs_python_c%d', M.classes.python)), ...
         fullfile(wc, sprintf('validation_gt_vs_wave_clus_notch_class%d', g))}
    st = load_validation_state(S{1}); c = st.cfg; c.out_dir = S{1}; c.stages = {'class_match', 'rasters'};
    run_validation(c);
end

% 3. the report's raster pages (same code as the report; non-raster outputs are restored below)
o.session = session; o.runs = struct('label', 'notch', 'folder', 'wave_clus notch');
if multi, o.out = rep; end
build_ppt_report(o); ppt_python_subsets(session, multi); ppt_gt_subsets(session, multi);

% 4. put every non-raster file back exactly as it was
for i = find(keep)'
    copyfile(fullfile(bk, erase(f(i).folder, rep), f(i).name), fullfile(f(i).folder, f(i).name));
end
rmdir(bk, 's');
fprintf('[ppt_redo_rasters] %s: raster pages re-drawn, %d other files restored unchanged\n', session, nnz(keep));
end

function r = is_raster(folder, name)
% raster outputs: raster page pngs and the files plot_subset_rasters / plot_class_rasters write beside them,
% plus the full-run raster / summary / sign-test copies in 08 and 09 summary
r = contains(name, 'resp_window') || contains(name, '_gt_vs_arm_summary') || contains(name, '_sign_test') || ...
    any(strcmp(name, {'data_table.mat', 'grapes_blanks.mat', 'grapes_response_metrics_ch' })) || ...
    startsWith(name, 'grapes_response_metrics_ch');
end
