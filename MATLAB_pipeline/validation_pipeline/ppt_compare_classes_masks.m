function ppt_compare_classes_masks(session, multi, py_k)
% PPT_COMPARE_CLASSES_MASKS  Never-rescued-by-mask breakdown for a ppt_compare_classes folder (alt_python_c<k>):
% the quarantined spikes python left in cluster 0, split by mask, for each comparison's quarantine regions:
%   <comparison>/regions/02_<ref>_x_quarantine/not_rescued_by_mask/   ref spikes (GT / wave_clus) python quarantined
%   <comparison>/regions/06_quarantine_only/not_rescued_by_mask/      python quarantined spikes in neither class
% Uses the comparison runs ppt_compare_classes made (their windows). ppt_compare_classes calls this at the end.
%   ppt_compare_classes_masks('pt6emu13_ch264_class3', false, 1)
if nargin < 2, multi = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; root = cfg.root; ch = cfg.ch;
pp = ppt_paths(cfg, multi); rep = pp.rep;
out = fullfile(rep, sprintf('alt_python_c%d', py_k));
M = load(fullfile(rep, sprintf('membership_ch%d.mat', ch))); P = M.python;
S_gp = fullfile(out, 'gt_vs_python'); S_wp = fullfile(out, 'wave_clus_vs_python');
st = load_validation_state(S_wp); wc_k = st.cfg.gt.class;
tm = struct('name', {sprintf('GT c%d', g), sprintf('wave_clus notch c%d', wc_k), sprintf('python c%d', py_k)}, ...
    'wave', {M.templates.gt(g, :), M.templates.wc_notch(wc_k, :), M.templates.python(py_k, :)}, ...
    'color', {[0 0 0], [0 .45 .7], [.9 .6 0]}, 'style', {'-', '--', ':'});
C = {S_gp, 'gt', g, 'GT'; S_wp, 'wc_notch', wc_k, 'wave_clus notch'};
for i = 1:size(C, 1)
    [sd, rn, rc, rlab] = C{i, :};
    w = load(fullfile(sd, 'validation_state.mat'), 'ref_tables_default'); two = isfield(w, 'ref_tables_default') && ~isempty(w.ref_tables_default);
    R = M.(rn); not_r = isnan(P.([rn '_class'])) | P.([rn '_class']) ~= rc;
    r1 = find(R.class == rc & R.python_quarantined & R.python_class == 0);            % ref spikes, python left in 0
    r2 = find(P.quarantined & P.class == 0 & not_r);                                   % python spikes in neither class
    ppt_mask_breakdown(R.t(r1), M.waves.(rn)(r1, :), R.python_idx(r1), P, tm, ...
        fullfile(sd, 'regions', sprintf('02_%s_x_quarantine', rn), 'not_rescued_by_mask'), sd, two, 'gt', ...
        sprintf('ch%d: %s c%d spikes python quarantined and never rescued (vs python c%d)', ch, rlab, rc, py_k));
    ppt_mask_breakdown(P.t(r2), M.waves.python(r2, :), r2, P, tm, ...
        fullfile(sd, 'regions', '06_quarantine_only', 'not_rescued_by_mask'), sd, two, 'arm', ...
        sprintf('ch%d: python quarantined spikes in neither %s c%d nor python c%d, never rescued', ch, rlab, rc, py_k));
    fprintf('[ppt_compare_classes_masks] %s vs python c%d: %d %s spikes never rescued, %d quarantine-only never rescued\n', ...
        rlab, py_k, numel(r1), rlab, numel(r2));
end
end
