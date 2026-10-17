function F = ppt_flags(session, multi)
% PPT_FLAGS  Things in a for ppt report that need a look, each with a likely explanation, -> <out>/flags.csv.
% Uses accounting.csv (ppt_check_unit), the membership and the full comparison runs. Rules:
%   size       a venn region is > 2% of its class (and >= 20 spikes): where those spikes are in the other sort
%   response   a region's in-window % differs from its class's by > 1.5x (>= 50 spikes): the region is
%              response-rich / response-poor compared with the class it comes from
%   quarantine one quarantine property explains > 50% of the wave_clus x quarantine spikes
%   zscore     paired sign test across the stim list on the full runs' per-stimulus z-score / sign-test p, p < 0.05
%   windows    stimuli with a late latency or a long response (default-window version exists)
%   check      any FAIL in accounting.csv
if nargin < 2, multi = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; ch = cfg.ch; root = cfg.root;
out = fullfile(root, 'for ppt'); if multi, out = fullfile(out, sprintf('GT c%d', g)); end
M = load(fullfile(out, sprintf('membership_ch%d.mat', ch)));
A = readtable(fullfile(out, 'accounting.csv'), 'TextType', 'string');
wc = fullfile(root, 'wave_clus notch'); d = dir(fullfile(root, 'python_recluster_*_second_pass')); py = d(end).name;
py_c = M.classes.python; wc_c = M.classes.wc_notch;
runs = {'wave_clus notch vs python', fullfile(wc, sprintf('validation_wave_clus_vs_python_c%d', py_c)), 'wave_clus notch', 'python'; ...
        'GT vs python', fullfile(root, sprintf('validation_gt_vs_%s_class%d', py, g)), 'GT', 'python'; ...
        'GT vs wave_clus notch', fullfile(wc, sprintf('validation_gt_vs_wave_clus_notch_class%d', g)), 'GT', 'wave_clus notch'};
F = cell(0, 5);   % severity | comparison | finding | likely explanation | figures

if any(A.check == "FAIL")
    F = addf(F, 'HIGH', 'all', sprintf('%d bookkeeping check(s) FAIL', nnz(A.check == "FAIL")), ...
        'a region does not match its venn or the regions do not add up -- figures are not consistent', 'accounting.csv');
end
regfig = containers.Map({'ref only', 'ref x quarantine', 'arm only', 'rescued only', 'quarantine only'}, ...
    {'03_wave_clus_only / 09 01_gt_only', '05_wave_clus_x_quarantine / 09 04_gt_x_quarantine', ...
     '03b_python_only / 09 02_*_only', '03c_rescued_only / 09 05_rescued_only', '03d_quarantine_only / 09 06_quarantine_only'});
for i = 1:size(runs, 1)
    cmp = runs{i, 1}; T = A(A.comparison == cmp, :);
    tot = T(endsWith(T.region, 'regions sum'), :); if height(tot) < 2, continue; end
    rn = runs{i, 3}; an = runs{i, 4};
    for r = ["ref only", "ref x quarantine", "arm only", "rescued only", "quarantine only"]
        row = T(T.region == r, :); if isempty(row), continue; end
        isref = startsWith(r, "ref"); ct = tot(1 + ~isref, :); who = ternary(isref, rn, an); other = ternary(isref, an, rn);
        frac = row.n_spikes / max(ct.n_spikes, 1);
        if row.n_spikes >= 20 && frac > 0.02 && r ~= "quarantine only"
            F = addf(F, ternary(frac > 0.10, 'HIGH', 'MED'), cmp, sprintf('%s: %d spikes = %.1f%% of %s''s class (%d in-window)', ...
                strrep(strrep(r, 'ref', rn), 'arm', an), row.n_spikes, 100 * frac, who, row.n_in_window), ...
                explain(r, who, other), regfig(char(r)));
        end
        if row.n_spikes >= 50
            rr = (row.n_in_window / row.n_spikes) / max(ct.n_in_window / ct.n_spikes, eps);
            if rr > 1.5 || rr < 1 / 1.5
                F = addf(F, 'MED', cmp, sprintf('%s: %.1f%% of its spikes in the response windows vs %.1f%% for the whole %s class', ...
                    strrep(strrep(r, 'ref', rn), 'arm', an), 100 * row.n_in_window / row.n_spikes, 100 * ct.n_in_window / ct.n_spikes, ...
                    ternary(isref, rn, an)), ternary(rr > 1, ...
                    'response-rich: these spikes carry response -- losing / adding them changes the response, look at their rasters', ...
                    'response-poor: mostly background firing (noise / other unit) -- removing them should sharpen the response'), regfig(char(r)));
            end
        end
    end
    % per-stimulus responsiveness of the two full runs
    st = load_validation_state(runs{i, 2}); R1 = st.ref_tables{1}; R2 = st.ref_tables{2};
    [~, a, b] = intersect(R1.stim_number, R2.stim_number);
    for f = {'zscore', 'p_value_sign'}
        if ~all(ismember(f{1}, [R1.Properties.VariableNames, R2.Properties.VariableNames])), continue; end
        x = R1.(f{1})(a); y = R2.(f{1})(b); k = isfinite(x) & isfinite(y);
        if nnz(k) >= 5, p = signtest(x(k), y(k)); else, p = NaN; end
        if p < 0.05
            up = nnz(y(k) > x(k)); F = addf(F, 'MED', cmp, sprintf('%s differs across the %d stimuli: %s higher in %d, lower in %d (sign test p=%.2g)', ...
                strrep(f{1}, '_', ' '), nnz(k), an, up, nnz(k) - up, p), ternary(strcmp(f{1}, 'zscore'), ...
                'different class size / baseline: more spikes raise baseline mu & sd, which lowers z even when the response is the same -- compare mu base / std base on the summary page', ...
                'the response is weaker / stronger in one sort per stimulus -- look at the stimuli that flip'), ...
                '08 summary / 09 summary');
        end
    end
end
% quarantine property behind most of wave_clus x quarantine
q = fullfile(out, 'wave_clus notch', '05_wave_clus_x_quarantine', 'counts.csv');
if exist(q, 'file')
    Q = readtable(q, 'TextType', 'string'); n_all = Q.n_spikes(1); Q = Q(2:end, :); Q = Q(~startsWith(Q.group, ["rescued", "not rescued"]), :);
    [mx, j] = max(Q.n_spikes);
    if n_all > 0 && mx / n_all > 0.5
        F = addf(F, 'MED', 'wave_clus notch vs python', sprintf('%.0f%% of the %d wave_clus x quarantine spikes fail "%s"', 100 * mx / n_all, n_all, Q.group(j)), ...
            sprintf(['python''s %s mask removes spikes wave_clus keeps in its class -- if they are in-window and look like the ' ...
            'template, the mask is too strict for this unit'], Q.group(j)), '05_wave_clus_x_quarantine/by property');
    end
end
% late latency / long duration
st = load_validation_state(runs{2, 2});
if isfield(st, 'ref_tables_default') && ~isempty(st.ref_tables_default)
    T = st.ref_tables{1}; fl = T.dura > st.cfg.max_dura | T.onset > st.cfg.max_lat;
    F = addf(F, 'LOW', 'GT windows', sprintf('stim %s: latency > %d ms or duration > %d ms', mat2str(T.stim_number(fl)'), st.cfg.max_lat, st.cfg.max_dura), ...
        'late / very long GT-estimated responses (or an off-response) -- the *_default_window rasters show the stimulus-locked view', ...
        '*_default_window folders');
end
F = cell2table(F, 'VariableNames', {'severity', 'comparison', 'finding', 'likely_explanation', 'figures'});
writetable(F, fullfile(out, 'flags.csv'));
fprintf('[ppt_flags] %s: %d flag(s) -> %s\n', session, height(F), fullfile(out, 'flags.csv'));
end

function F = addf(F, sev, cmp, f, th, fig)
F(end + 1, :) = {sev, cmp, f, th, fig};
end

function s = explain(r, who, other)
switch char(r)
    case 'ref only'
        s = sprintf(['%s keeps these in its class but %s has them in another class / cluster 0 / not detected -- see the ' ...
            '"where they are" panels: not detected = detection difference; cluster 0 or another class = a clustering boundary'], who, other);
    case 'ref x quarantine'
        s = sprintf('%s keeps them; %s quarantined them (by property) -- mask disagreement, check the property breakdown', who, other);
    case 'arm only'
        s = sprintf(['%s assigns these to its class but %s does not -- if they match the template and are in-window, %s ' ...
            'under-assigns; if they look different, %s over-merges'], who, other, other, who);
    case 'rescued only'
        s = sprintf('rescued into %s''s class by template matching but %s lacks them -- rescue may be pulling in neighbours', who, other);
    otherwise
        s = '';
end
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
