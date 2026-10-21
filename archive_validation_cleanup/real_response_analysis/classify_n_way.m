function T = classify_n_way(display_sets, full_sets, arm_names, tol)
% CLASSIFY_N_WAY  Classify every spike across GT + N arms into a membership
% category, WITHOUT deduplicating across arms -- each arm's own spike keeps
% its own recorded time and gets its own row, tagged with which OTHER
% arm(s) also have a matching counterpart within tol. Generalizes
% classify_three_way.m (fixed at GT+sd1+sd2, 7 hardcoded categories) to an
% arbitrary number of arms with data-driven category labels instead of a
% hardcoded enum, so N=1 arm (3 categories: gt / arm1 / gt+arm1) and N=3
% arms (15 categories) both fall out of the same code.
%
% display_sets{a} are the DISPLAY sets: response-window-restricted, per
% stimulus -- these decide WHICH spikes get a row/tick and at what time, one
% row per arm's own detection. Whether a category counts as "only"
% (exclusive) is judged against each OTHER arm's FULL session population
% (full_sets{a}), NOT the other arm's window-restricted set -- a spike that
% exists in another arm's data but simply falls outside that arm's own
% (somewhat arbitrary, per-stimulus) response window is not a real
% detection failure, it's a window-boundary artifact (see
% classify_three_way.m's header for the stim513 case this was confirmed on).
%
% VARIABLES
%   display_sets - input: cell array, one per arm (same order as arm_names),
%                  each a vector of this arm's own DISPLAY spike times
%   full_sets    - input: cell array, one per arm, FULL session population
%                  times (class-filtered, not window-restricted) -- used
%                  only to test cross-arm presence/absence
%   arm_names    - input: cellstr, one name per arm (arm_names{1} is
%                  conventionally 'gt', but nothing here requires that)
%   tol          - input: match tolerance (ms)
%   n            - number of arms
%   a/da         - loop variable: which arm, and its display set
%   present      - per-spike-in-da boolean matrix, one column per arm:
%                  does this spike have a counterpart (within tol) in that
%                  arm's FULL population? (own arm's column is trivially true)
%   time/arm/category - growing columns, one row per (arm, own display spike)
%   T            - output table: time, arm, category (category = '+'-joined
%                  names of every arm this spike was found in, e.g. 'gt',
%                  'gt+rescue_sd1', or with 3 arms 'gt+rescue_sd1+rescue_sd2')
    n = numel(arm_names);
    time = []; arm = {}; category = {};

    for a = 1:n
        da = display_sets{a};
        if isempty(da), continue; end
        present = false(numel(da), n);
        present(:,a) = true;
        for b = 1:n
            if b == a, continue; end
            present(:,b) = nearest_match(da, full_sets{b}, tol) > 0;
        end
        for i = 1:numel(da)
            c = strjoin(arm_names(present(i,:)), '+');
            time(end+1,1) = da(i); arm{end+1,1} = arm_names{a}; category{end+1,1} = c;
        end
    end

    T = table(time, arm, category);
end
