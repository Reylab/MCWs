function ppt_session(session, multi)
% PPT_SESSION  Everything the for ppt report of one GT unit needs, then the report:
%   grapes_blanks for <channel>/wave_clus notch (build_wave_clus_arm, once per channel), the venns of the
%   notch run (venn_wave_clus, with GT vs python), then build_ppt_report (only the notch run is compared; the
%   no-notch run is not used and nothing about it goes in for ppt). MULTI = the channel has more than one GT unit -> report in
%   <channel>/for ppt/GT c<g>/.
here = fileparts(mfilename('fullpath')); addpath(here);
cfg = validation_sessions(session); root = cfg.root; g = cfg.gt.class;
pp = ppt_paths(cfg, nargin > 1 && multi);   % python arm, validation and report folders (run tag: ppt_paths)
wc = fullfile(root, 'wave_clus notch');
assert(exist(fullfile(wc, cfg.gt.times_file), 'file') == 2, 'ppt_session: no notch run in %s', root);
if ~exist(fullfile(wc, 'grapes_blanks.mat'), 'file'), build_wave_clus_arm(root, cfg.ch, wc); end
% venns made before this run folder got its name or for another python arm: kept aside, remade below.
% A venn of THIS run and THIS python arm (e.g. another GT unit's, same channel) is kept.
py_now = fullfile(root, pp.py);
for v = dir(fullfile(pp.wcval, 'venn_*'))'
    st = fullfile(v.folder, v.name, 'validation_state.mat'); keep = false;
    if exist(st, 'file')
        w = load(st, 'cfg'); u = {w.cfg.gt.dir, w.cfg.arms.dir};
        keep = any(strcmp(u, wc)) && (startsWith(v.name, 'venn_gt_vs_wave_clus') || any(strcmp(u, py_now)));
    end
    if ~keep
        if ~exist(fullfile(pp.wcval, 'old_venns'), 'dir'), mkdir(fullfile(pp.wcval, 'old_venns')); end
        dst = fullfile(pp.wcval, 'old_venns', v.name); k = 1;
        while exist(dst, 'dir'), k = k + 1; dst = fullfile(pp.wcval, 'old_venns', sprintf('%s_%d', v.name, k)); end
        movefile(fullfile(v.folder, v.name), dst);
    end
end
venn_wave_clus(session, 'wave_clus notch', 'gt_vs_python', true);
o.session = session;
o.runs = struct('label', 'notch', 'folder', 'wave_clus notch');
% the no-notch run is not compared here (no 07_no_notch_vs_notch in for ppt)
o.out = pp.rep;
build_ppt_report(o);
ppt_python_subsets(session, nargin > 1 && multi);   % 03b_python_only, 03c_rescued_only, 03d_quarantine_only
ppt_gt_subsets(session, nargin > 1 && multi);       % 09_gt_vs_python, wave_clus notch/09_gt_vs_wave_clus
end
