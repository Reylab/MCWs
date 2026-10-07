function W = get_waveforms_for_times(times, times_cc_col2, spikes_matrix, tol)
% GET_WAVEFORMS_FOR_TIMES  Match each absolute spike time in `times` to its
% row in a times_*.mat's cluster_class (column 2, absolute ms) and return
% the corresponding row of the row-aligned waveform matrix (times_*.mat's
% own `spikes` field). Same nearest_match-based join used throughout this
% pipeline (e.g. gt_waveforms in analyze_cluster_membership.m), just
% generalized to any arm's times file instead of only GT's.
%
% VARIABLES
%   times          - input: absolute spike times to fetch waveforms for
%   times_cc_col2  - input: the SOURCE times file's cluster_class(:,2), same
%                    row order as spikes_matrix
%   spikes_matrix  - input: the SOURCE times file's own `spikes` (waveform) matrix
%   tol            - match tolerance (ms), same convention as elsewhere
%   idx            - nearest_match result: row in times_cc_col2 for each entry of times
%   keep           - which entries of times actually found a match within tol
%   W              - output: matched waveform rows, one per matched time
    idx = nearest_match(times, times_cc_col2, tol);
    keep = idx > 0;
    W = spikes_matrix(idx(keep), :);
end
