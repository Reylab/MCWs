function ppt_resume_unit(session, multi)
% PPT_RESUME_UNIT  Finish one GT unit's for ppt report, doing only what is missing:
%   main report (ppt_session) | the notch run's GT venn (venn_wave_clus, if it was moved aside) |
%   03b / 03c / 03d (ppt_python_subsets) | 09 GT comparisons (ppt_gt_subsets)
here = fileparts(mfilename('fullpath')); addpath(here);
cfg = validation_sessions(session); g = cfg.gt.class;
o = fullfile(cfg.root, 'for ppt'); if multi, o = fullfile(o, sprintf('GT c%d', g)); end
w = fullfile(o, 'wave_clus notch'); wc = fullfile(cfg.root, 'wave_clus notch');
if ~exist(fullfile(w, '08_rasters_wave_clus_vs_python'), 'dir')
    ppt_session(session, multi); fprintf('[resume] %s: full report made\n', session); return
end
if ~exist(fullfile(wc, sprintf('venn_gt_vs_wave_clus_class%d', g), 'arm_classes.csv'), 'file')
    venn_wave_clus(session, 'wave_clus notch');
end
need = {};
if ~exist(fullfile(w, '03b_python_only', 'python_only_spikes.csv'), 'file'), need{end + 1} = '03b'; end
if ~exist(fullfile(w, '03c_rescued_only', 'rescued_only_spikes.csv'), 'file'), need{end + 1} = '03c'; end
if ~exist(fullfile(w, '03d_quarantine_only', 'counts.csv'), 'file'), need{end + 1} = '03d'; end
if ~isempty(need), ppt_python_subsets(session, multi, '', need); end
if ~exist(fullfile(o, '09_gt_vs_python', 'counts.csv'), 'file') || ~exist(fullfile(w, '09_gt_vs_wave_clus', 'counts.csv'), 'file')
    ppt_gt_subsets(session, multi);
end
fprintf('[resume] %s: done (python subsets made: %s)\n', session, strjoin(need, ' '));
end
