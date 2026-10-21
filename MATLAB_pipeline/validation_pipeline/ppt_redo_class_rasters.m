function ppt_redo_class_rasters(session, multi)
% PPT_REDO_CLASS_RASTERS  Re-draw only the 06_all_classes raster pages of one GT unit's report with the windows
% of each sort's MATCHED class (GT: GT class g; python: python's matched class; wave_clus: its matched class) --
% the rule build_ppt_report now uses. Old 06 folders are moved to <channel>/archive_<date>/06_own_class_windows.
%   ppt_redo_class_rasters('pt5emu4_ch271_class2', false)
if nargin < 2, multi = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; root = cfg.root;
rep = fullfile(root, 'for ppt'); if multi, rep = fullfile(rep, sprintf('GT c%d', g)); end
M = load(fullfile(rep, sprintf('membership_ch%d.mat', cfg.ch)));
d = dir(fullfile(root, 'python_recluster_*_second_pass')); py = d(end).name;
S_gt = fullfile(root, sprintf('validation_gt_vs_%s_class%d', py, g));                                  % ref GT, arm python
S_wc = fullfile(root, 'wave_clus notch', sprintf('validation_wave_clus_vs_python_c%d', M.classes.python)); % ref wave_clus
arch = fullfile(root, ['archive_' datestr(now, 'yyyy-mm-dd')], '06_own_class_windows', erase(rep, fullfile(root, 'for ppt')));
jobs = {M.gt, 'gt', S_gt, 'gt', fullfile(rep, '06_all_classes'); M.python, 'python', S_gt, 'arm', fullfile(rep, '06_all_classes'); ...
        M.wc_notch, 'wave_clus', S_wc, 'gt', fullfile(rep, 'wave_clus notch', '06_all_classes')};
for j = unique(jobs(:, 5))'                                  % move the old pages aside (not deleted)
    if exist(j{1}, 'dir'), dst = fullfile(arch, erase(j{1}, rep)); mkdir(fileparts(dst)); movefile(j{1}, dst); end
end
for i = 1:size(jobs, 1)
    [T, nm, st, side, out] = jobs{i, :};
    w = load(fullfile(st, 'validation_state.mat'), 'ref_tables_default'); two = isfield(w, 'ref_tables_default') && ~isempty(w.ref_tables_default);
    for k = unique(T.class(T.class > 0))'
        t = T.t(T.class == k); name = sprintf('%s_c%d', nm, k);
        plot_subset_rasters([], 'dir', st, 'times', t, 'name', name, 'side', side, 'out_dir', out);
        if two, plot_subset_rasters([], 'dir', st, 'times', t, 'name', [name '_default_window'], 'side', side, 'out_dir', out, 'window', 'default'); end
    end
end
fprintf('[ppt_redo_class_rasters] %s: 06_all_classes re-drawn with matched-class windows\n', session);
end
