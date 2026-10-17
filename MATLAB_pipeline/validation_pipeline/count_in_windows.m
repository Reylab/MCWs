function [n, hit] = count_in_windows(t, stims, lo, hi, onset_map)
% COUNT_IN_WINDOWS  Spikes (absolute times T, ms) inside the raster windows:
% for each stimulus STIMS(i), every trial onset in ONSET_MAP plus
% [LO(i), HI(i)] (ms from onset; NaN = no window). Each spike counts once,
% however many windows it falls in -- the "n response spikes" in a raster
% page title, from the same green windows the page draws.
%   n   - spikes inside any window
%   hit - logical, same size as T
    t = t(:);
    hit = false(size(t));
    for i = 1:numel(stims)
        if ~isKey(onset_map, stims(i)) || ~isfinite(lo(i)) || ~isfinite(hi(i)), continue; end
        t0 = onset_map(stims(i));
        for tr = 1:numel(t0)
            hit = hit | (t >= t0(tr) + lo(i) & t <= t0(tr) + hi(i));
        end
    end
    n = nnz(hit);
end
