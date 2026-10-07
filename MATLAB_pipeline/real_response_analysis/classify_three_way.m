function T = classify_three_way(gt_t, sd1_t, sd2_t, gt_full_t, sd1_full_t, sd2_full_t, tol)
% CLASSIFY_THREE_WAY  Classify every spike across GT/SD1/SD2 into one of 7
% membership categories, WITHOUT deduplicating across arms -- each arm's
% own spike keeps its own recorded time and gets its own row, tagged with
% which other arm(s) have a matching counterpart within tol. This is built
% for the colored union raster (one tick per arm's OWN detection, at that
% arm's own time, colored by consensus) -- for a deduplicated headline
% gained/lost COUNT, use membership_sets (pairwise, one-to-one) instead;
% this function intentionally does NOT enforce one-to-one matching since a
% raster tick belongs to a specific arm's specific recorded time regardless
% of whether another arm's nearby spike was "claimed" by something else.
%
% gt_t/sd1_t/sd2_t are the DISPLAY sets: response-window-restricted, per
% stimulus -- these decide WHICH spikes get a row/tick on the raster and at
% what time, one row per arm's own detection. But whether a category counts
% as "only" (exclusive) is judged against each OTHER arm's FULL session
% population (gt_full_t/sd1_full_t/sd2_full_t), NOT the other arm's
% window-restricted set. Reason: a spike that exists in another arm's data
% but simply falls outside the (somewhat arbitrary, per-arm, per-stimulus)
% response window WE defined is not a real detection failure -- it's a
% window-boundary artifact. Confirmed directly against stim513: all 20
% "GT only" spikes in GT's extra window zone (not reached by SD1/SD2's
% narrower windows) turned out to have a match somewhere in SD2's full
% 18238-spike population -- zero were genuinely absent. So "only" here means
% "not found anywhere in that arm's data," not "not found in that arm's
% window."
%
% category codes (7 total):
%   'gt_only'  'sd1_only'  'sd2_only'  'all3'
%   'gt_sd1'  (gt+sd1, not sd2)
%   'gt_sd2'  (gt+sd2, not sd1 -- expected rare/near-empty; kept distinct
%              rather than folded into gt_sd1, since folding would silently
%              misclassify any that do occur)
%   'sd1_sd2' (sd1+sd2, not gt)
%
% VARIABLES
%   gt_t/sd1_t/sd2_t - inputs: each arm's DISPLAY spike times for one
%                       stimulus (response-window-restricted) -- one row
%                       per entry, this decides what's plotted
%   gt_full_t/sd1_full_t/sd2_full_t - inputs: each arm's FULL session spike
%                       times (class-filtered, not window-restricted) --
%                       used only to test cross-arm presence/absence
%   tol - match tolerance (ms)
%   time/arm/category    - growing columns, one triplet per output row
%   in_sd1_of_gt/in_sd2_of_gt - per-GT-spike: does it have a counterpart
%                       anywhere in sd1/sd2's FULL population
%   in_gt_of_sd1/in_sd2_of_sd1 - per-SD1-spike: does it have a counterpart
%                       anywhere in gt/sd2's FULL population
%   in_gt_of_sd2/in_sd1_of_sd2 - per-SD2-spike: does it have a counterpart
%                       anywhere in gt/sd1's FULL population
%   i/c - loop index / this row's category string
%   T   - output table: time, arm, category

    time = []; arm = {}; category = {};

    in_sd1_of_gt = nearest_match(gt_t, sd1_full_t, tol) > 0;
    in_sd2_of_gt = nearest_match(gt_t, sd2_full_t, tol) > 0;
    for i = 1:numel(gt_t)
        if in_sd1_of_gt(i) && in_sd2_of_gt(i)
            c = 'all3';
        elseif in_sd1_of_gt(i)
            c = 'gt_sd1';
        elseif in_sd2_of_gt(i)
            c = 'gt_sd2';
        else
            c = 'gt_only';
        end
        time(end+1,1) = gt_t(i); arm{end+1,1} = 'gt'; category{end+1,1} = c;
    end

    in_gt_of_sd1  = nearest_match(sd1_t, gt_full_t, tol) > 0;
    in_sd2_of_sd1 = nearest_match(sd1_t, sd2_full_t, tol) > 0;
    for i = 1:numel(sd1_t)
        if in_gt_of_sd1(i) && in_sd2_of_sd1(i)
            c = 'all3';
        elseif in_gt_of_sd1(i)
            c = 'gt_sd1';
        elseif in_sd2_of_sd1(i)
            c = 'sd1_sd2';
        else
            c = 'sd1_only';
        end
        time(end+1,1) = sd1_t(i); arm{end+1,1} = 'sd1'; category{end+1,1} = c;
    end

    in_gt_of_sd2  = nearest_match(sd2_t, gt_full_t, tol) > 0;
    in_sd1_of_sd2 = nearest_match(sd2_t, sd1_full_t, tol) > 0;
    for i = 1:numel(sd2_t)
        if in_gt_of_sd2(i) && in_sd1_of_sd2(i)
            c = 'all3';
        elseif in_gt_of_sd2(i)
            c = 'gt_sd2';
        elseif in_sd1_of_sd2(i)
            c = 'sd1_sd2';
        else
            c = 'sd2_only';
        end
        time(end+1,1) = sd2_t(i); arm{end+1,1} = 'sd2'; category{end+1,1} = c;
    end

    T = table(time, arm, category);
end
