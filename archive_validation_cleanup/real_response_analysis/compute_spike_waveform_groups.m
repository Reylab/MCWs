function groups = compute_spike_waveform_groups(times_struct, spikes_struct, subset_times, tol)
% COMPUTE_SPIKE_WAVEFORM_GROUPS  Splits a subset of a rescue arm's own spike
% times into named groups by provenance, for visually inspecting what each
% category's waveforms actually look like (see plot_provenance_waveform_pages.m):
%   'original'          - neither forced nor rescued
%   'forced'             - force-classified AND NOT rescued. Rescue takes
%                          priority: a spike that's both force-classified and
%                          rescued is a 'rescued: <mask>' spike, not 'forced'
%                          -- otherwise it would be double-counted in both
%                          'forced' and its rescued-mask group, and
%                          original+forced+rescued wouldn't partition the
%                          subset. 'forced' and 'rescued: <mask>' are
%                          mutually exclusive because of this.
%   'rescued: <mask>'    - one entry per quarantine mask actually present
%                          (see classify_spike_provenance.m), restricted to
%                          the RESCUED spikes that failed THAT mask --
%                          non-disjoint FROM EACH OTHER (a spike failing 2
%                          masks appears in both mask groups -- same
%                          "involvement" convention as
%                          compute_spike_provenance_breakdown.m's mask_counts),
%                          but every rescued spike belongs to at least one of
%                          them and NONE of 'original'/'forced'.
%
% Shares classify_spike_provenance.m with compute_spike_provenance_breakdown.m
% so the counts (row 2/3 bars) and these waveform groups can't silently
% disagree about which spike belongs where.
%
% Each group also carries .properties (extract_quarantine_properties_table.m):
% every within_channel.m per-spike metric (width, prominence_ratio,
% peak_sample_20, num_peaks, the fail_* flags, the composed reason string,
% etc.) for that group's own spikes -- so a flagged spike's ACTUAL numbers
% can be checked directly (a real violation should show e.g. a genuinely low
% peak_sample_20 for a "low amplitude" group; one that looks normal is a
% sign something upstream is mismatched, not that the spike is bad).
%
% VARIABLES
%   times_struct/spikes_struct/subset_times/tol - input: see classify_spike_provenance.m
%   groups        - output: struct array, one row per category, with .name,
%                   .times (subset of subset_times in that category), and
%                   .properties (table, same row order as .times)
    subset_times = subset_times(:);
    groups = struct('name', {}, 'times', {}, 'properties', {});
    if isempty(subset_times)
        return;
    end
    pc = classify_spike_provenance(times_struct, spikes_struct, subset_times, tol);

    sel = ~pc.is_forced & ~pc.is_rescued;
    groups(end+1) = mk_group('original', subset_times, pc.idx_all, sel, spikes_struct);
    sel = pc.is_forced & ~pc.is_rescued;
    groups(end+1) = mk_group('forced', subset_times, pc.idx_all, sel, spikes_struct);

    for m = 1:numel(pc.mask_labels)
        sel = pc.is_rescued & pc.mask_fail(:,m);
        groups(end+1) = mk_group(sprintf('rescued: %s', pc.mask_labels{m}), subset_times, pc.idx_all, sel, spikes_struct); %#ok<AGROW>
    end
end

function g = mk_group(name, subset_times, idx_all, sel, spikes_struct)
    t = subset_times(sel);
    props = extract_quarantine_properties_table(spikes_struct, idx_all(sel), t);
    g = struct('name', name, 'times', t, 'properties', props);
end
