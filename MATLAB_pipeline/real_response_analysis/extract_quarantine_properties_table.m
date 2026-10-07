function T = extract_quarantine_properties_table(spikes_struct, idx_all, subset_times)
% EXTRACT_QUARANTINE_PROPERTIES_TABLE  Pulls EVERY field of a spikes file's
% quarantine_properties (within_channel.m's per-spike metrics: width,
% prominence_ratio, peak_sample_20, num_peaks, the fail_* flags, the
% composed reason string, etc.) for a given subset of spike times, aligned
% row-for-row -- so you can look at the ACTUAL numbers behind a "low
% amplitude" or "bad width" flag and judge for yourself whether it's a real
% violation or a bug (e.g. a flagged spike whose width/amplitude actually
% looks fine would mean something's wrong upstream).
%
% Uses idx_all (from classify_spike_provenance.m's tolerance-based lookup)
% rather than re-deriving the match itself, so this can't disagree with
% which spike the rest of the provenance pipeline thinks is which.
%
% VARIABLES
%   spikes_struct - input: the arm's loaded *_spikes.mat (needs
%                   quarantine_properties; pass [] or omit the field to get
%                   just spike_time back with no property columns)
%   idx_all       - input: per-subset-spike index into spikes_struct.index_all,
%                   0 = no match within tol (classify_spike_provenance.m's output)
%   subset_times  - input: the same spike times idx_all was computed against
%   T             - output: table, one row per subset_times entry, column
%                   'spike_time' plus one column per quarantine_properties
%                   field (NaN, or '' for cellstr fields, where idx_all==0)
    subset_times = subset_times(:);
    n = numel(subset_times);
    T = table(subset_times, 'VariableNames', {'spike_time'});
    if isempty(spikes_struct) || ~isfield(spikes_struct, 'quarantine_properties')
        return;
    end
    qp = spikes_struct.quarantine_properties;
    fns = fieldnames(qp);
    hit = idx_all > 0;
    for i = 1:numel(fns)
        fn = fns{i};
        val = qp.(fn);
        if iscell(val)
            col = repmat({''}, n, 1);
            col(hit) = val(idx_all(hit));
        elseif islogical(val)
            col = false(n, 1);
            col(hit) = val(idx_all(hit));
        else
            col = nan(n, 1);
            col(hit) = double(val(idx_all(hit)));
        end
        T.(fn) = col;
    end
end
