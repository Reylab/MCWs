function [classes_full, info] = template_classify_from_grapes(spikes_in, grapes_file, ch, root, varargin)
% TEMPLATE_CLASSIFY_FROM_GRAPES  Extend a grapes ground-truth clustering
% (which only labels spikes INSIDE stimulus windows) to every spike in a
% spikes file, via nearest-template waveform matching.
%
% Spikes that fall inside a grapes stimulus window keep their TRUE grapes
% class, exactly (no approximation). Every other spike (e.g. blank-screen /
% inter-trial spikes grapes never saw) gets the class of its nearest
% template in raw waveform space - APPROXIMATE.
%
% NOT wired into any pipeline yet. Standalone utility: returns a full-length
% classes vector in the same shape update_grapes / update_grapes_blanks
% expect as their `classes{chi}` input, for later use (e.g. building an
% approximate grapes_blanks for a hand-select ground truth that has no
% matching-scheme full-session clustering of its own).
%
% USAGE
%   classes_full = template_classify_from_grapes(spikes_in, grapes_file, ch, root)
%
% INPUTS
%   spikes_in    EITHER a path to a *_spikes.mat / times_*.mat file with
%                `spikes` (N x nsamples waveforms) + `index` or
%                `cluster_class(:,2)` (spike times, ms) -- OR an already-
%                loaded struct with those same fields.
%   grapes_file  path to a grapes*.mat (ground truth raster). Plain or
%                blanks flavour give identical results (only post-onset
%                spikes -- rel>=0 -- are used, and those are unaffected by
%                the blanks/circshift machinery).
%   ch           channel number.
%   root         session root (needs stimulus.mat, finalevents.mat,
%                experiment_properties_online3.mat -- to replay the same
%                stimulus-onset lookup grapes itself was built from).
%
%   Name/value:
%     'match_tol_ms'    0.5   grapes<->spikes_in time-match tolerance (ms)
%     'min_class_n'     5     classes with fewer matched examples don't get
%                              a template (their own members still keep
%                              their true grapes label; nothing else can be
%                              assigned TO that class)
%     'reject_sdnum'    Inf   if finite, spikes farther than
%                              reject_sdnum * (that class's own within-class
%                              std of distance-from-mean) from every
%                              template are left unclassified (class 0)
%                              instead of forced to the nearest one.
%
% OUTPUT
%   classes_full   1 x N vector, N = size(spikes,1), same row order as
%                  spikes_in: grapes-covered spikes = true grapes class;
%                  everything else = nearest-template class (or 0, if
%                  rejected / no template available).
%   info           struct: n_total, n_grapes_labeled, n_template_labeled,
%                  n_unclassified, per_class_counts, templates, template_sd,
%                  gt_classes.
%
% CAVEAT: nearest-template classification of spikes grapes never labeled is
% an approximation, not a real sort -- it has no ground truth to validate
% against by construction. Good enough for a baseline-rate estimate; not a
% substitute for actually clustering those spikes.

    p = inputParser;
    addParameter(p, 'match_tol_ms', 0.5);
    addParameter(p, 'min_class_n', 5);
    addParameter(p, 'reject_sdnum', Inf);
    parse(p, varargin{:});
    o = p.Results;

    addpath(genpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline'));

    % ---- load / accept spikes_in -------------------------------------------
    if ischar(spikes_in) || isstring(spikes_in)
        S = load(char(spikes_in));
    else
        S = spikes_in;
    end
    assert(isfield(S, 'spikes'), 'spikes_in needs a ''spikes'' field (waveforms).');
    spikes = double(S.spikes);
    if isfield(S, 'index')
        sp_time = double(S.index(:))';
    elseif isfield(S, 'cluster_class')
        sp_time = double(S.cluster_class(:,2))';
    else
        error('spikes_in needs ''index'' or ''cluster_class(:,2)'' for spike times.');
    end
    n_total = size(spikes, 1);
    assert(numel(sp_time) == n_total, 'spikes/time length mismatch (%d vs %d)', n_total, numel(sp_time));

    % ---- ground-truth (post-onset only, exact) -----------------------------
    omap = build_onset_map(root);
    gt = reconstruct_gt_cc(grapes_file, ch, omap);
    fprintf('[tmplcls] ch%d ground truth: %d spikes, classes %s\n', ...
            ch, size(gt,1), mat2str(unique(gt(:,1))'));

    % ---- match ground-truth spike times onto spikes_in ---------------------
    gm = nearest_match(gt(:,2), sp_time, o.match_tol_ms);
    matched = gm > 0;
    classes_full = zeros(1, n_total);
    classes_full(gm(matched)) = gt(matched, 1)';
    n_gt_labeled = nnz(classes_full > 0);
    fprintf('[tmplcls] matched to spikes_in: %d / %d ground-truth spikes (%.1f%%)\n', ...
            nnz(matched), size(gt,1), 100*nnz(matched)/size(gt,1));

    % ---- build per-class templates from the trusted subset -----------------
    gt_classes = unique(classes_full(classes_full > 0));
    templates = zeros(numel(gt_classes), size(spikes,2));
    template_sd = zeros(1, numel(gt_classes));
    keep_class = true(1, numel(gt_classes));
    for i = 1:numel(gt_classes)
        k = gt_classes(i);
        wf = spikes(classes_full == k, :);
        if size(wf,1) < o.min_class_n
            keep_class(i) = false;
            continue
        end
        templates(i,:) = mean(wf, 1);
        template_sd(i) = sqrt(sum(var(wf, 1)));   % same convention as build_templates.m
    end
    gt_classes = gt_classes(keep_class);
    templates  = templates(keep_class, :);
    template_sd = template_sd(keep_class);
    fprintf('[tmplcls] templates built for classes: %s  (min_class_n=%d)\n', ...
            mat2str(gt_classes), o.min_class_n);

    % ---- nearest-template classification for everything else ---------------
    unlabeled = find(classes_full == 0);
    n_template_labeled = 0;
    if ~isempty(unlabeled) && ~isempty(templates)
        wf = spikes(unlabeled, :);
        % euclidean distance, n_unlabeled x n_templates
        D = sqrt(max(bsxfun(@plus, sum(wf.^2,2), sum(templates.^2,2)') - 2*(wf*templates'), 0));
        [dmin, jmin] = min(D, [], 2);
        ok = true(size(dmin));
        if isfinite(o.reject_sdnum)
            radius = o.reject_sdnum * template_sd(jmin)';
            ok = dmin <= radius(:);
        end
        classes_full(unlabeled(ok)) = gt_classes(jmin(ok));
        n_template_labeled = nnz(ok);
    end
    n_unclassified = nnz(classes_full == 0);

    fprintf(['[tmplcls] ch%d done: %d grapes-exact + %d template-approx = %d labeled, ' ...
             '%d left unclassified (class 0) / %d total\n'], ...
            ch, n_gt_labeled, n_template_labeled, n_gt_labeled+n_template_labeled, ...
            n_unclassified, n_total);

    info = struct('n_total', n_total, 'n_grapes_labeled', n_gt_labeled, ...
                  'n_template_labeled', n_template_labeled, 'n_unclassified', n_unclassified, ...
                  'gt_classes', gt_classes, 'templates', templates, 'template_sd', template_sd);
    info.per_class_counts = arrayfun(@(k) nnz(classes_full==k), [0, gt_classes]);
end

% ----------------------------------------------------------------------
function omap = build_onset_map(root)
% snum -> [stimulus onset times, ms], replaying update_grapes's own lookup
% (2D pics_onset quirk included, for consistency with the production code).
    omap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    S = load(fullfile(root, 'stimulus.mat'));       stimulus = S.stimulus;
    F = load(fullfile(root, 'finalevents.mat'));    pics_onset = F.pics_onset;
    E = load(fullfile(root, 'experiment_properties_online3.mat'), 'scr_end_cell');
    Nscr = min(numel(stimulus), numel(E.scr_end_cell));
    for scri = 1:Nscr
        po = pics_onset{scri};
        st = stimulus{scri};
        for j = 1:numel(st)
            snum  = st(j).ID;
            terna = st(j).terna_onset;
            if ndims(po) == 3
                ts = po(sub2ind(size(po), terna(:,1), terna(:,2), terna(:,3)));
            else
                M  = po(terna(:,1), terna(:,3));
                ts = M(1:length(M));
            end
            ts = ts(:).';
            if ~isKey(omap, snum), omap(snum) = []; end
            omap(snum) = [omap(snum), ts];
        end
    end
end

% ----------------------------------------------------------------------
function cc = reconstruct_gt_cc(grapes_file, ch, omap)
% [class, time_ms] for every POST-ONSET (rel>=0) spike in a grapes raster,
% de-duplicated across overlapping stimulus windows. Identical for plain vs
% blanks flavour (blanks/circshift never touch rel>=0).
    cc = zeros(0,2);
    G = load(grapes_file);
    cf = ['chan' num2str(ch)];
    if ~isfield(G, 'rasters') || ~isfield(G.rasters, cf), return; end
    R = G.rasters.(cf);
    fn = fieldnames(R);
    cls = sort(cellfun(@(x) sscanf(x, 'class%d'), fn(~cellfun('isempty', ...
             regexp(fn, '^class\d+$', 'once')))))';
    allt = []; allc = [];
    for ci = 1:numel(cls)
        C = sprintf('class%d', cls(ci));
        stimc = R.(C).stim;
        for snum = 1:numel(stimc)
            tr = stimc{snum};
            if isempty(tr), continue; end
            if ~iscell(tr), tr = num2cell(tr, 2); end
            if ~isKey(omap, snum), continue; end
            O = omap(snum);
            if numel(O) ~= numel(tr), continue; end
            for t = 1:numel(tr)
                rel = tr{t}(:).';
                rel = rel(rel >= 0 & rel < 9000);
                if isempty(rel), continue; end
                allt = [allt, rel + O(t)];
                allc = [allc, repmat(cls(ci), 1, numel(rel))];
            end
        end
    end
    if isempty(allt), return; end
    [allt, ord] = sort(allt); allc = allc(ord);
    keep = [true, ~(diff(allt) <= 0.05 & diff(allc) == 0)];
    cc = [allc(keep)', allt(keep)'];
end

% ----------------------------------------------------------------------
function gm = nearest_match(gt_t, ar_t, tol)
% gm(i) = index into ar_t of the nearest spike to gt_t(i) within tol, else 0.
    gm = zeros(numel(gt_t), 1);
    [as, order] = sort(ar_t(:));
    for i = 1:numel(gt_t)
        k = find(as >= gt_t(i) - tol, 1, 'first');
        best = 0; bestd = tol + 1;
        while ~isempty(k) && k <= numel(as) && as(k) <= gt_t(i) + tol
            dd = abs(as(k) - gt_t(i));
            if dd < bestd, bestd = dd; best = k; end
            k = k + 1;
        end
        if best > 0, gm(i) = order(best); end
    end
end
