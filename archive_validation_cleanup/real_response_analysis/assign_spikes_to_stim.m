function stim_of = assign_spikes_to_stim(times, omap, stim_list)
% ASSIGN_SPIKES_TO_STIM  For each spike ABSOLUTE time in `times`, find which
% stimulus's trial onset it's nearest to (same nearest-onset logic used
% throughout this pipeline, e.g. find_response_spikes_dynamic) and return
% that stim_number. Used to bucket a flat classified spike list (e.g.
% find_response_spikes_dynamic's `inside` output, which is pooled across
% every stimulus in stim_list) back into per-stimulus groups for the
% per-stimulus comparison page.
%
% VARIABLES
%   times     - input: absolute spike times to bucket
%   omap      - containers.Map from the onset-map build, stim ID -> onset times
%   stim_list - which stim IDs to consider (must match what built omap)
%   stim_of   - output: stim_number nearest to each entry of times (NaN if
%               omap is empty/has no entries at all)
%   all_onsets/all_stims - every (onset time, stim_number) pair across
%               stim_list, pooled and sorted by time together
%   ord       - sort permutation applied to both all_onsets and all_stims together
%   n/i       - loop count / index over `times`
%   t         - current spike time being assigned
%   j         - position in all_onsets where onset <= t most recently became true
%   cands     - the 1-3 nearby onset indices actually compared against t
%   best      - which of cands is closest to t
    all_onsets = [];
    all_stims = [];
    for s = stim_list(:)'
        if ~isKey(omap, s), continue; end
        ts = omap(s);
        all_onsets = [all_onsets, ts];
        all_stims  = [all_stims, repmat(s, 1, numel(ts))];
    end
    [all_onsets, ord] = sort(all_onsets);
    all_stims = all_stims(ord);

    n = numel(times);
    stim_of = nan(n,1);
    if isempty(all_onsets), return; end
    for i = 1:n
        t = times(i);
        j = find(all_onsets <= t, 1, 'last');
        cands = max(j-1,1):min(j+1,numel(all_onsets));
        if isempty(cands), continue; end
        [~, best] = min(abs(all_onsets(cands) - t));
        stim_of(i) = all_stims(cands(best));
    end
end
