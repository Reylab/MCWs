function [inside,outside,stim_of_inside] = find_response_spikes_dynamic(class_num,grapes,times_cc,omap,per_stim,stimlist,ch,buffer,fixed_width,other_stim_list,other_onset,other_dura)
    % FIND_RESPONSE_SPIKES_DYNAMIC  Same idea as find_response_spikes/
    % classify_class_spikes, but instead of one fixed [-pre,+post] window
    % applied to every trial, each STIMULUS gets its own window ANCHORED at
    % its own measured response latency:
    %   [onset - buffer, onset + width + buffer]
    % where width = that stimulus's own measured dura (default), OR a single
    % fixed_width applied to every stimulus if you pass one -- useful when a
    % stimulus's own duration estimate isn't trustworthy but you still want
    % the window to start at the right place per stimulus, and want one knob
    % to tune the width quickly rather than adjusting it per side/per stim.
    %
    % "onset" here is the response LATENCY (ms after stimulus flash, from
    % get_latency_BCM via create_responses_data/characterize_grapes_responses),
    % NOT the trial's absolute onset time -- it's on the same relative-to-
    % stimulus-onset clock as every spike time in the raster (see
    % get_latency_BCM.m: it searches ejex, the raster's own time axis,
    % between 80-1000ms post-stimulus). The absolute window bound is that
    % latency added to the trial's real onset time (from omap), same as the
    % fixed-window version.
    %
    % class_num   - which class to score (e.g. 1)
    % grapes      - loaded grapes_blanks*.mat struct (this arm's raster)
    % times_cc    - cluster_class loaded from the times_*.mat file, [class time_ms]
    % omap        - containers.Map from the onset-map build, scoped to stimlist
    % per_stim    - table with (at least) stim_number/onset/dura, one row per
    %               scored stimulus for THIS class (get_class_metrics's m.per_stim)
    % stimlist    - the stim IDs you're restricting to (must match what built omap)
    % ch          - channel number (for the field name lookup)
    % buffer      - ms padding added on each side of [onset, onset+width]
    % fixed_width - OPTIONAL. Omit or pass [] to use each stimulus's own dura
    %               (fully dynamic, the original behavior). Pass a number to
    %               use that width for every stimulus instead of dura -- still
    %               anchored per-stimulus at its own onset, still gated on
    %               good_lat (onset itself still needs a confident detection).
    % other_stim_list - OPTIONAL. Omit or pass [] to score every stim in
    %               `stimlist` off its own per_stim row (original behavior).
    %               Pass a list of stim IDs to make THOSE specific stims skip
    %               their own row/good_lat/onset/dura check entirely and use
    %               other_onset/other_dura instead -- for stimuli that were
    %               never expected to elicit a response (so their own
    %               latency estimate is meaningless or absent), letting you
    %               still borrow a comparable, trustworthy window (e.g. the
    %               median onset/dura from your real responsive stimuli)
    %               anchored at each such stimulus's own trial onset time.
    %               Every OTHER stim in `stimlist` (not in other_stim_list)
    %               still uses its own row exactly as before -- this builds
    %               ONE unified set of windows across both kinds of stimuli
    %               in a single call, so `outside` stays a true "not in ANY
    %               window" baseline and a spike can't be double-counted
    %               between two separate calls (split `inside` back out by
    %               ismember(stim_of_inside, other_stim_list)).
    % other_onset/other_dura - the fixed latency/width applied to every stim
    %               in other_stim_list. Required together if other_stim_list
    %               is non-empty; ignored otherwise.
    % stim_of_inside - OUTPUT, same length/order as `inside`: which stim ID's
    %               window actually matched each inside-window spike -- use this
    %               for per-stimulus bucketing, not a separately re-derived
    %               nearest-onset guess (which can disagree when windows have
    %               different widths per stimulus).
    if nargin < 9,  fixed_width = []; end
    if nargin < 10, other_stim_list = []; end
    if nargin < 11, other_onset = []; end
    if nargin < 12, other_dura = []; end

    cf = ['chan' num2str(ch)];
    cls = ['class' num2str(class_num)];

    %% verify times/grapes agreement (identical to find_response_spikes.m)
    recon_times = reconstruct_class_times(grapes.rasters.(cf).(cls).stim,omap);

    tt = sort(times_cc(times_cc(:,1) == class_num,2));
    tol  = 0.5;   % ms -- 1:1 spike-time match tolerance (same convention used elsewhere in the repo)
    min_hit_frac = 0.70;
    hit = 0; j = 1; nt = numel(tt);
    for i = 1:numel(recon_times)
        rt = recon_times(i);
        while j <= nt && tt(j) < (rt-tol)
            j = j+1;
        end
        if j <= nt && abs(tt(j) - rt) <= tol
            hit = hit +1;
        end
    end
    hit_frac = hit / max(numel(recon_times), 1);
    times_matches_raster = hit_frac >= min_hit_frac;

    if times_matches_raster
        class_abs_times = times_cc(times_cc(:,1) == class_num, 2);
        fprintf('[classify-dynamic] class%d: 1:1 time match confirmed (%.0f%% of %d recon spikes matched within %.1fms) -- using cluster_class directly.\n', ...
                class_num, 100*hit_frac, numel(recon_times), tol);
    else
        class_abs_times = recon_times;
        fprintf('[classify-dynamic] class%d: time match FAILED (%.0f%%, need %.0f%%) -- falling back to raster reconstruction (post-onset only, no outside-window coverage).\n', ...
                class_num, 100*hit_frac, 100*min_hit_frac);
    end

    %% build per-trial dynamic windows: [trial_onset + latency - buffer, trial_onset + latency + dura + buffer]
    win_lo = [];
    win_hi = [];
    win_stim = [];   % which stim ID each window row belongs to -- kept so "inside" spikes can report
                      % their TRUE match provenance directly, instead of a second, separately-computed
                      % nearest-onset guess (assign_spikes_to_stim) that can disagree with this window
                      % match whenever windows have different widths per stimulus.
    for s = stimlist(:)'
        if ~isKey(omap,s), continue; end
        if ismember(s, other_stim_list)
            % Borrowed window: this stim was never expected to have its own
            % confident detection, so skip the row/good_lat/onset/dura check
            % entirely and anchor the caller-supplied fixed onset+dura here instead.
            lat = other_onset;
            dura = other_dura;
        else
            row = per_stim(per_stim.stim_number == s, :);
            if isempty(row), continue; end            % this stim has no metrics row (not scored for this class)
            if ~row.good_lat(1), continue; end        % no CONFIDENT detection -- dura falls back to the single longest
                                                        % above-threshold IFR stretch even if it never met the real
                                                        % sustained-response criteria, which can be much wider than a
                                                        % genuine response. create_responses_data.m gates on good_lat
                                                        % for the same reason; skip rather than trust a loose fallback.
            lat = row.onset(1);
            if isnan(lat), continue; end   % get_latency_BCM found no bout at all -- nothing to anchor a window on
            if isempty(fixed_width)
                dura = row.dura(1);
                if isnan(dura), continue; end
            else
                dura = fixed_width;        % override this stimulus's own dura -- width is now fixed, only the anchor (onset) varies per stim
            end
        end
        trial_times = omap(s);
        win_lo = [win_lo, trial_times + lat - buffer];
        win_hi = [win_hi, trial_times + lat + dura + buffer];
        win_stim = [win_stim, repmat(s, 1, numel(trial_times))];
    end
    [win_lo, ord] = sort(win_lo);
    win_hi = win_hi(ord);
    win_stim = win_stim(ord);

    %% classify each spike: inside if it falls within ANY trial's dynamic window
    n = numel(class_abs_times);
    is_inside = false(n,1);
    stim_of = nan(n,1);
    for i = 1:n
        t = class_abs_times(i);
        k = find(win_lo <= t,1,'last');
        cands = max(k-1,1):min(k+1,numel(win_lo));
        hit = cands(t >= win_lo(cands) & t <= win_hi(cands));
        if ~isempty(hit)
            is_inside(i) = true;
            % A spike inside any stim_list window is a stim_list response;
            % "other" (borrowed-window) stimuli only get spikes outside every
            % stim_list window. Previously the earliest-starting window won,
            % so the preceding "other" stimulus's window tail took stim_list
            % responses (GT ch328: 850 in stim_list windows, 824 reported).
            if numel(hit) > 1 && ~isempty(other_stim_list)
                pref = hit(~ismember(win_stim(hit), other_stim_list));
                if ~isempty(pref), hit = pref; end
            end
            stim_of(i) = win_stim(hit(1));
        end
    end
    inside  = class_abs_times(is_inside);
    outside = class_abs_times(~is_inside);
    stim_of_inside = stim_of(is_inside);
    fprintf('[classify-dynamic] class%d: %d inside dynamic window, %d outside (of %d total)\n', ...
            class_num, numel(inside), numel(outside), n);

    function t = reconstruct_class_times(stimc, omap)
        t = [];
        for s = 1:numel(stimc)
            if isempty(stimc{s}) || ~isKey(omap, s), continue; end
            tr = stimc{s};
            if ~iscell(tr), tr = num2cell(tr,2); end
            O = omap(s);
            if numel(O) ~= numel(tr), continue; end
            for k = 1:numel(tr)
                rel = tr{k}(:).';
                rel = rel(rel >= 0 & rel < 9000);      % post-onset only // blanks caveat
                if ~isempty(rel), t = [t, rel + O(k)]; end
            end
        end
        t = sort(t(:));
    end
end
