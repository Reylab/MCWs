function times_out = adapt_python_times_for_matlab(times_py)
% ADAPT_PYTHON_TIMES_FOR_MATLAB  Normalizes a loaded Python-pipeline
% times_*.mat struct (mcws's own save format) so this MATLAB pipeline's
% arm-comparison code (do_structure_rescue_BCM_online3.m,
% match_arms_to_ground_truth.m, classify_spike_provenance.m,
% rescue_mask_breakdown.m, classify_quarantined_not_rescued.m, ...) can use
% it as a normal arm without any of that code needing to change.
%
% Field-by-field comparison against a real MATLAB times file showed
% cluster_class/spikes/forced/inspk are already the same dtype (float64/
% uint8) -- no casting needed there. Three REAL structural gaps, not just
% naming, handled here:
%
% 1. POPULATION SCOPE / class 0. Python's cluster_class includes EVERY raw
%    spike, with class 0 reserved for quarantined-or-never-rescued spikes
%    (Python's own convention -- class 0 is never a real cluster). MATLAB's
%    cluster_class/cluster_class_pre_rescue never contain class-0 rows at
%    all. Every consumer in this pipeline filters by an explicit target
%    class number (cca(cca(:,1)==class,2), see match_arms_to_ground_
%    truth.m), so leftover class-0 rows in cluster_class itself are low
%    risk -- left untouched here. The two fields THIS function synthesizes
%    (cluster_class_pre_rescue, below) DO strip class-0, since nothing
%    downstream filters those by class before treating them as "the
%    accepted population."
%
% 2. cluster_class_pre_rescue does not exist in Python's output -- Python
%    tracks rescue via quarantine.rescued (a per-spike boolean, aligned to
%    cluster_class's rows) instead of a separate pre/post array pair.
%    Reconstructed as cluster_class, minus whichever rows quarantine.
%    rescued flags true (exactly the rows rescue ADDED) minus class-0 rows.
%
% 3. rescue_mask / quarantine_masks_used don't exist as top-level fields --
%    Python's equivalent lives nested in quarantine.rescued, and its
%    rejector names (bundle_collision/within_channel/refractory_period) are
%    a DIFFERENT VOCABULARY than MATLAB's three mask names (mask_nonart/
%    mask_non_quarantine/mask_non_refract), not a 1:1 rename -- see the
%    mapping below (same pairing used by adapt_python_spikes_for_matlab.m,
%    which builds the *_spikes.mat-equivalent this pipeline's mask-breakdown
%    code also needs).
%
% VARIABLES
%   times_py   - input: struct from load(times_*.mat) as saved by mcws's
%                own pipeline (needs cluster_class; quarantine.rescued used
%                if present -- a plain/no-rescue Python run legitimately
%                omits it entirely, same as three of six real arm files
%                checked here, and is treated as "nothing rescued" rather
%                than an error)
%   times_out  - output: times_py, unmodified, PLUS cluster_class_
%                pre_rescue, rescue_mask, quarantine_masks_used added.
%                rescue_mask is aligned to cluster_class's OWN rows here
%                (not a separate raw-spikes file, unlike MATLAB) -- pair
%                with adapt_python_spikes_for_matlab.m's output, built from
%                the SAME times_py, for anything expecting a *_spikes.mat
%                struct with a matching index_all.
    times_out = times_py;
    cc = times_py.cluster_class;
    n = size(cc, 1);

    if isfield(times_py, 'quarantine') && isfield(times_py.quarantine, 'rescued')
        rescued = logical(times_py.quarantine.rescued(:));
        if numel(rescued) ~= n
            error('adapt_python_times_for_matlab: quarantine.rescued length (%d) != cluster_class rows (%d)', numel(rescued), n);
        end
    else
        rescued = false(n, 1);
    end

    real_class = cc(:,1) ~= 0;

    % #1 + #2
    times_out.cluster_class_pre_rescue = cc(real_class & ~rescued, :);

    % #3: rescue_mask, aligned to cluster_class's own rows (see
    % adapt_python_spikes_for_matlab.m for the matching index_all).
    % LOGICAL row vector, like rescue_spikes.m writes (false(1,n_all)) --
    % consumers index with it directly (make_cluster_report's breakdown pages),
    % and a uint8 0/1 vector there is read as index numbers (index 0 errors).
    % Same for forced: logical in MATLAB times files, uint8 in Python's.
    times_out.rescue_mask = reshape(rescued, 1, []);
    if isfield(times_out, 'forced') && ~isempty(times_out.forced)
        times_out.forced = logical(times_out.forced(:));
    end

    % Conceptual pairing, not a literal rename -- mask_nonart = cross-
    % channel/bundle artifact, mask_non_quarantine = within-channel
    % waveform QC, mask_non_refract = refractory-violation chain. Only
    % listed if the corresponding quarantine.<field> is actually present.
    py_to_matlab = struct( ...
        'bundle_collision',  'mask_nonart', ...
        'within_channel',    'mask_non_quarantine', ...
        'refractory_period', 'mask_non_refract');
    py_names = fieldnames(py_to_matlab);
    used = {};
    for i = 1:numel(py_names)
        if isfield(times_py, 'quarantine') && isfield(times_py.quarantine, py_names{i})
            used{end+1} = py_to_matlab.(py_names{i}); %#ok<AGROW>
        end
    end
    times_out.quarantine_masks_used = used;

    % Selected-temperature index: Python stores a 0-based row of the SPC tree,
    % MATLAB (find_temp / gui_status.current_temp / Temp) a 1-based one.
    % Verified on ch328: Python current_temp=10, and its SPC class sizes
    % (4702, 17) are tree row 11 in MATLAB indexing, not row 10.
    if isfield(times_out, 'gui_status') && isfield(times_out.gui_status, 'current_temp')
        times_out.gui_status.current_temp = double(times_out.gui_status.current_temp) + 1;
    end
    % Same 0-based -> 1-based shift for the other indices the temperature
    % diagram reads. Verified on ch328: Python original_classes are 0 / 1 for
    % final class 1 / 2, whose sizes are tree columns 1 / 2 in MATLAB terms;
    % rescued spikes carry no SPC cluster, so they become 0 ("null") as in MATLAB.
    if isfield(times_out, 'gui_status') && isfield(times_out.gui_status, 'original_classes') ...
            && numel(times_out.gui_status.original_classes) == n
        oc = double(times_out.gui_status.original_classes(:));
        spc_assigned = real_class & ~rescued;
        oc(spc_assigned)  = oc(spc_assigned) + 1;
        oc(~spc_assigned) = 0;
        times_out.gui_status.original_classes = oc;
    end
    if isfield(times_out, 'gui_status') && isfield(times_out.gui_status, 'auto_sort_info') ...
            && isfield(times_out.gui_status.auto_sort_info, 'elbow')
        times_out.gui_status.auto_sort_info.elbow = double(times_out.gui_status.auto_sort_info.elbow) + 1;
    end
    if isfield(times_out, 'Temp') && ~isempty(times_out.Temp)
        times_out.Temp = double(times_out.Temp) + 1;
    end
end
