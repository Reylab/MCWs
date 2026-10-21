function ppt_wc_py_destinations(session, multi)
% PPT_WC_PY_DESTINATIONS  By-destination breakdown of the wave_clus notch vs python venn regions (as the GT
% comparisons have): each region split by where its spikes are in the other sort, with waveforms and rasters
% (each sort's own matched-class windows from the wave_clus vs python run) -> <region folder>/by_destination/
%   03_wave_clus_only, 05_wave_clus_x_quarantine   (wave_clus spikes: where they are in python)
%   03b_python_only, 03c_rescued_only, 03d_quarantine_only   (python spikes: where they are in wave_clus)
%   ppt_wc_py_destinations('pt4emu7_ch322_class3', false)
if nargin < 2, multi = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; root = cfg.root; ch = cfg.ch;
pp = ppt_paths(cfg, multi); rep = pp.rep;
wr = fullfile(rep, 'wave_clus notch');
M = load(fullfile(rep, sprintf('membership_ch%d.mat', ch)));
W = M.wc_notch; P = M.python; py_c = M.classes.python; wc_c = M.classes.wc_notch;
state = fullfile(pp.wcval, sprintf('validation_wave_clus_vs_python_c%d', py_c));
w = load(fullfile(state, 'validation_state.mat'), 'ref_tables_default');
two = isfield(w, 'ref_tables_default') && ~isempty(w.ref_tables_default);
tm = struct('name', {sprintf('wave_clus notch c%d', wc_c), sprintf('python c%d', py_c), sprintf('GT c%d', g)}, ...
    'wave', {M.templates.wc_notch(wc_c, :), M.templates.python(py_c, :), M.templates.gt(g, :)}, ...
    'color', {[0 .45 .7], [.9 .6 0], [0 0 0]}, 'style', {'-', '--', ':'});

% the venn regions (as build_ppt_report / ppt_python_subsets / ppt_check_unit define them)
in_wc = W.class == wc_c; in_py = P.class == py_c; not_wc = isnan(P.wc_notch_class) | P.wc_notch_class ~= wc_c;
R = {'03_wave_clus_only', in_wc & (isnan(W.python_class) | (W.python_class ~= py_c & ~W.python_quarantined)), 'wc'; ...
     '05_wave_clus_x_quarantine', in_wc & W.python_quarantined & W.python_class ~= py_c, 'wc'; ...
     '03b_python_only', in_py & ~P.quarantined & not_wc, 'py'; ...
     '03c_rescued_only', in_py & P.quarantined & not_wc, 'py'; ...
     '03d_quarantine_only', P.quarantined & ~in_py & not_wc, 'py'};
for i = 1:size(R, 1)
    [nm, m, who] = R{i, :};
    if ~any(m), fprintf('[ppt_wc_py_destinations] %s: no spikes\n', nm); continue; end
    if strcmp(who, 'wc')
        ppt_by_destination(W.t(m), M.waves.wc_notch(m, :), W.python_class(m), py_c, 'python', tm, ...
            fullfile(wr, nm, 'by_destination'), state, two, 'gt', [0 .45 .7], ...
            sprintf('ch%d: %s (wave_clus notch c%d vs python c%d), by where they are in python', ch, strrep(nm(4:end), '_', ' '), wc_c, py_c));
    else
        ppt_by_destination(P.t(m), M.waves.python(m, :), P.wc_notch_class(m), wc_c, 'wave_clus notch', tm, ...
            fullfile(wr, nm, 'by_destination'), state, two, 'arm', [.9 .6 0], ...
            sprintf('ch%d: %s (wave_clus notch c%d vs python c%d), by where they are in wave_clus', ch, strrep(nm(4:end), '_', ' '), wc_c, py_c));
    end
end
fprintf('[ppt_wc_py_destinations] %s done -> %s\n', session, wr);
end
