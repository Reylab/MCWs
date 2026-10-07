function [M, labels, active] = quarantine_property_flags(T)
% QUARANTINE_PROPERTY_FLAGS  Per-spike fail flags for EVERY potential
% quarantine mask, for a Python-layout times file, always in the same order
% so every figure shows the same set even when a mask caught nothing:
%
%   bundle collision      bundle_collision            (MATLAB mask_nonart)
%   collision             MATLAB separate_collisions only (mask_non_collision)
%   structural            \
%   low amplitude          | within_channel, fanned out per check and rebuilt
%   bad width              | from Python's raw metrics by
%   bad prominence ratio  /  adapt_python_spikes_for_matlab (MATLAB mask_non_quarantine)
%   multipeak             within_channel check, opt-in in MATLAB, never a
%                         Python quarantine reason
%   refractory            refractory_period           (MATLAB mask_non_refract)
%   task-specific         task_specific               (MATLAB mask_taskspks)
%
%   M      - n_spikes x 9 logical, failed that mask AND the mask fed this
%            run's quarantine (all-false column for an inactive mask)
%   labels - the 9 names above
%   active - 1 x 9 logical: the mask fed this run's quarantine. The union of
%            the active columns is quarantine.all.
    labels = {'bundle collision', 'collision', 'structural', 'low amplitude', 'bad width', ...
              'bad prominence ratio', 'multipeak', 'refractory', 'task-specific'};
    sp = adapt_python_spikes_for_matlab(T);
    n = numel(sp.index_all);
    M = false(n, numel(labels));
    active = false(1, numel(labels));
    q = struct();
    if isfield(T, 'quarantine'), q = T.quarantine; end

    if isfield(sp, 'mask_nonart')
        M(:, 1) = ~sp.mask_nonart(:); active(1) = true;
    end
    % 2: no collision rejector in Python
    if isfield(q, 'within_channel')
        qp = struct();
        if isfield(sp, 'quarantine_properties'), qp = sp.quarantine_properties; end
        wc = {'fail_structural', 'fail_amp', 'fail_width', 'fail_prom_ratio'};
        for i = 1:numel(wc)
            if isfield(qp, wc{i})
                M(:, 2+i) = logical(qp.(wc{i})(:)); active(2+i) = true;
            end
        end
        if ~any(active(3:6))
            warning('quarantine_property_flags: within_channel present but its per-check flags could not be rebuilt.');
        end
    end
    % 7: multipeak is computed by within_channel but never quarantines in Python
    if isfield(sp, 'mask_non_refract')
        M(:, 8) = ~sp.mask_non_refract(:); active(8) = true;
    end
    if isfield(q, 'task_specific') && numel(q.task_specific) == n
        M(:, 9) = logical(q.task_specific(:)); active(9) = true;
    end
end
