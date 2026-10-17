function M = build_sort_membership(sorts, varargin)
% BUILD_SORT_MEMBERSHIP  Where every spike of every sort went in every other sort.
%
%   sorts = struct('name', {'gt', 'wave_clus', 'python'}, ...
%                  'file', {gt_times_file, wc_times_file, py_times_file});
%   M = build_sort_membership(sorts, 'tol', 0.5, 'out_file', fullfile(out, 'membership_ch259.mat'));
%
% For each sort, M.<name> is a table with one row per spike (the times file's cluster_class order):
%   t, class                       spike time (ms) and its class in this sort (0 = unassigned)
%   quarantined, rescued, forced, mahal_reassigned, q_<property>
%                                  python-layout files only (quarantine.all, quarantine.rescued,
%                                  forced, mahal_reassigned, quarantine_property_flags per property)
%   <other>_idx                    row of the paired spike in sort <other> (0 = no spike there)
%   <other>_class                  its class in <other> (NaN = no spike there, 0 = unassigned)
%   <other>_quarantined            (python <other> only) the paired spike was quarantined
%   <other>_dt                     paired spike time - this spike time (ms)
% Pairing: one-to-one, closest first, |dt| <= tol, over ALL spikes of both sorts (every class).
% M.waves.<name> = the stored waveforms (spikes) of that sort, rows as in the table.
% M.templates.<name> = class means (row k = class k) from those waveforms.
%
% Typical subsets:
%   gt spikes in python class X:      M.gt(M.gt.python_class == X, :)
%   wave_clus c1 spikes python lacks:  M.wave_clus(M.wave_clus.class == 1 & isnan(M.wave_clus.python_class), :)
%   python c1 spikes in wave_clus c2: M.python(M.python.class == 1 & M.python.wave_clus_class == 2, :)
p = inputParser;
addParameter(p, 'tol', 0.5);
addParameter(p, 'out_file', '');
parse(p, varargin{:}); o = p.Results;
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(fileparts(here), 'real_response_analysis'));

n = numel(sorts); D = cell(1, n);
for i = 1:n
    T = load(sorts(i).file);
    d = table(double(T.cluster_class(:, 2)), double(T.cluster_class(:, 1)), 'VariableNames', {'t', 'class'});
    if isfield(T, 'quarantine') && isfield(T.quarantine, 'all') && any(T.quarantine.all)
        d.quarantined = logical(T.quarantine.all(:));
        r = false(height(d), 1); if isfield(T.quarantine, 'rescued'), r = logical(T.quarantine.rescued(:)); end
        d.rescued = r;
        if isfield(T, 'forced'), d.forced = logical(T.forced(:)); end
        if isfield(T, 'mahal_reassigned'), d.mahal_reassigned = logical(T.mahal_reassigned(:)); end
        [Q, labels, active] = quarantine_property_flags(T);
        for j = find(active)
            d.(['q_' matlab.lang.makeValidName(strrep(labels{j}, ' ', '_'))]) = Q(:, j);
        end
    end
    D{i} = d;
    if isfield(T, 'spikes'), M.waves.(sorts(i).name) = double(T.spikes); else, M.waves.(sorts(i).name) = []; end
    W = M.waves.(sorts(i).name); cls = d.class; K = max(cls);
    tm = nan(max(K, 0), size(W, 2));
    for k = 1:K, if any(cls == k) && ~isempty(W), tm(k, :) = mean(W(cls == k, :), 1); end, end
    M.templates.(sorts(i).name) = tm;
end
for a = 1:n
    for b = 1:n
        if a == b, continue; end
        [ia, ib] = pair_closest(D{a}.t, D{b}.t, o.tol);
        idx = zeros(height(D{a}), 1); idx(ia) = ib;
        c = nan(height(D{a}), 1); c(ia) = D{b}.class(ib);
        dt = nan(height(D{a}), 1); dt(ia) = D{b}.t(ib) - D{a}.t(ia);
        nb = sorts(b).name;
        D{a}.([nb '_idx']) = idx; D{a}.([nb '_class']) = c; D{a}.([nb '_dt']) = dt;
        if ismember('quarantined', D{b}.Properties.VariableNames)
            q = false(height(D{a}), 1); q(ia) = D{b}.quarantined(ib); D{a}.([nb '_quarantined']) = q;
        end
    end
end
for i = 1:n, M.(sorts(i).name) = D{i}; end
M.sorts = sorts; M.tol = o.tol;
if ~isempty(o.out_file)
    save(o.out_file, '-struct', 'M', '-v7.3');
    fprintf('[build_sort_membership] -> %s\n', o.out_file);
end
end

function [ia, ib] = pair_closest(a, b, tol)
% one-to-one, closest pairs first, |b - a| <= tol; a and b need not be sorted
[as, oa] = sort(a(:)); [bs, ob] = sort(b(:));
pa = zeros(0, 1); pb = pa; pd = pa; j0 = 1; nb = numel(bs);
ca = cell(numel(as), 1); cb = ca;
for i = 1:numel(as)
    while j0 <= nb && bs(j0) < as(i) - tol, j0 = j0 + 1; end
    j = j0;
    while j <= nb && bs(j) <= as(i) + tol, j = j + 1; end
    if j > j0, ca{i} = repmat(i, j - j0, 1); cb{i} = (j0:j - 1)'; end
end
pa = vertcat(pa, ca{:}); pb = vertcat(pb, cb{:}); pd = abs(bs(pb) - as(pa));
[~, s] = sort(pd);
ua = false(numel(as), 1); ub = false(nb, 1); keep = false(numel(s), 1);
for k = 1:numel(s)
    i = pa(s(k)); j = pb(s(k));
    if ~ua(i) && ~ub(j), ua(i) = true; ub(j) = true; keep(k) = true; end
end
ia = oa(pa(s(keep))); ib = ob(pb(s(keep)));
end
