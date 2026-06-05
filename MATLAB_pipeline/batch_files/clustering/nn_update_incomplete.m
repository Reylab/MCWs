function index = nearest_neighbor(spike_x, tmplt_vect, maxdist, par_or_pointdist, varargin)
    % nearest_neighbor - Backward-compatible nearest-neighbor with algo-specific weighting.
    %
    % Supports two call styles used in this repo:
    % 1) New style: nearest_neighbor(spike, templates, maxdist, par, ...)
    % 2) Legacy style: nearest_neighbor(spike, templates, maxdist, pointdist, pointlimit, k)

    % Legacy API path used by template_type 'nn'.
    if ~isstruct(par_or_pointdist)
        pointdist = par_or_pointdist;
        pointlimit = Inf;
        k = [];
        if ~isempty(varargin), pointlimit = varargin{1}; end
        if numel(varargin) >= 2, k = varargin{2}; end
        index = legacy_nearest_neighbor(spike_x, tmplt_vect, maxdist, pointdist, pointlimit, k);
        return
    end

    par = par_or_pointdist;
    if ~isfield(par, 'pk_weight'), par.pk_weight = 50; end
    if ~isfield(par, 'amp_dir'), par.amp_dir = 'neg'; end
    if ~isfield(par, 'xor_weight'), par.xor_weight = 5; end

    % Support positional algo argument, e.g. nearest_neighbor(..., par, 'algo3').
    algo = 'algo0';
    extra = varargin;
    if ~isempty(extra) && (ischar(extra{1}) || (isstring(extra{1}) && isscalar(extra{1})))
        first = char(extra{1});
        if startsWith(first, 'algo')
            algo = first;
            extra = extra(2:end);
        end
    end

    p = inputParser;
    addParameter(p, 'pointdist', [], @(x) isnumeric(x));
    addParameter(p, 'pointlimit', Inf, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'k', [], @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'algo', algo, @(x) ischar(x) || (isstring(x) && isscalar(x)));
    addParameter(p, 'template_weight', [], @(x) isnumeric(x) || isempty(x));
    parse(p, extra{:});

    pointdist = p.Results.pointdist;
    pointlimit = p.Results.pointlimit;
    k = p.Results.k;
    algo = char(p.Results.algo);
    template_weight = p.Results.template_weight;

    distances = compute_algo_distances(spike_x, tmplt_vect, par, algo, template_weight);
    conforming = find(distances < maxdist);

    % Optional pointwise filter for compatibility.
    if ~isempty(pointdist)
        pointwise_conforming = [];
        for i = 1:size(tmplt_vect, 1)
            if sum(abs(spike_x - tmplt_vect(i, :)) > pointdist(i, :)) < pointlimit
                pointwise_conforming = [pointwise_conforming i]; %#ok<AGROW>
            end
        end
        conforming = intersect(conforming, pointwise_conforming);
    end

    if isempty(conforming)
        index = 0;
    else
        if ~isempty(k)
            [~, i] = sort(distances(conforming));
            i = i(1:min(length(i), k));
        else
            [~, i] = min(distances(conforming));
        end
        index = conforming(i);
    end
end

function index = legacy_nearest_neighbor(spike_x, tmplt_vect, maxdist, pointdist, pointlimit, k)
    diff = tmplt_vect - repmat(spike_x, size(tmplt_vect, 1), 1);
    distances = sqrt(sum(diff.^2, 2))';
    conforming = find(distances < maxdist);

    if ~isempty(pointdist)
        pointwise_conforming = [];
        for i = 1:size(tmplt_vect, 1)
            if sum(abs(spike_x - tmplt_vect(i, :)) > pointdist(i, :)) < pointlimit
                pointwise_conforming = [pointwise_conforming i]; %#ok<AGROW>
            end
        end
        conforming = intersect(conforming, pointwise_conforming);
    end

    if isempty(conforming)
        index = 0;
    else
        if ~isempty(k)
            [~, i] = sort(distances(conforming));
            i = i(1:min(length(i), k));
        else
            [~, i] = min(distances(conforming));
        end
        index = conforming(i);
    end
end

function distances = compute_algo_distances(spike_x, tmplt_vect, par, algo, template_weight)
    n_templates = size(tmplt_vect, 1);
    distances = zeros(1, n_templates);
    spike_width = get_peak_width(spike_x, par.amp_dir);

    for i = 1:n_templates
        template = tmplt_vect(i, :);
        template_width = get_peak_width(template, par.amp_dir);
        [normConst, weights] = compute_weight_vector(length(spike_x), spike_width, template_width, algo, par, template_weight);
        diff = spike_x - template;
        distances(i) = normConst * sqrt(sum(weights .* (diff .^ 2)));
    end
end

function [normConst, weights] = compute_weight_vector(n, spike_width, template_width, algo, par, template_weight)
    spike_mask = false(1, n);
    template_mask = false(1, n);

    if ~isnan(spike_width.left) && ~isnan(spike_width.right)
        spike_mask(spike_width.left:min(spike_width.right, n)) = true;
    end
    if ~isnan(template_width.left) && ~isnan(template_width.right)
        template_mask(template_width.left:min(template_width.right, n)) = true;
    end

    spike_wt = par.pk_weight;
    if isempty(template_weight)
        xor_wt = par.xor_weight;
    else
        xor_wt = template_weight;
    end

    weights = ones(1, n);
    overlap = spike_mask & template_mask;
    xor_region = (spike_mask | template_mask) & ~overlap;

    % Preserve the algo switch behavior used by template matching experiments.
    switch algo
        case 'algo0'
            % Baseline: no extra weighting.
        case 'algo1'
            weights(spike_mask) = spike_wt;
        case 'algo2'
            weights(overlap) = spike_wt;
        case 'algo3'
            weights(overlap) = spike_wt;
            weights(xor_region) = xor_wt;
        case 'algo4'
            weights(spike_mask | template_mask) = spike_wt;
        case 'algo5'
            weights(template_mask) = spike_wt;
        otherwise
            % Unknown mode falls back to baseline behavior.
    end

    normConst = sqrt(n) / sqrt(sum(weights));
end

function width_struct = get_peak_width(spike_x, amp_dir)
    if strcmp(amp_dir, 'neg')
        wav = -spike_x;
    else
        wav = spike_x;
    end
    [pks, locs, ~, p] = findpeaks(wav);

    if ~isempty(pks)
        [pk, peak_loc] = max(pks);
        pk_loc = locs(peak_loc);
        prom_max = p(peak_loc);

        level = pk - prom_max / 2;
        above_level = find(wav >= level);

        if ~isempty(above_level)
            diff_above = diff(above_level);
            breaks = find(diff_above > 1);
            segments = {};
            start_idx = 1;
            for b = 1:length(breaks)
                segments{end + 1} = above_level(start_idx:breaks(b)); 
                start_idx = breaks(b) + 1;
            end
            segments{end + 1} = above_level(start_idx:end); 

            peak_segment = [];
            for s = 1:length(segments)
                if any(segments{s} == pk_loc)
                    peak_segment = segments{s};
                    break;
                end
            end

            if ~isempty(peak_segment)
                left_width = min(peak_segment);
                right_width = max(peak_segment);
            else
                left_width = NaN;
                right_width = NaN;
            end
        else
            left_width = NaN;
            right_width = NaN;
        end

        if left_width <= 0
            left_width = 1;
        end
        if right_width > length(spike_x)
            right_width = length(spike_x);
        end
    else
        left_width = NaN;
        right_width = NaN;
    end

    width_struct.left = left_width;
    width_struct.right = right_width;
end