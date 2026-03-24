function index = nearest_neighbor(spike_x, tmplt_vect, maxdist, par, varargin)
    % nearest_neighbor - Find nearest neighbor(s) within a distance threshold
    %
    % Required:
    %   spike_x    - row vector query point (1 x n_features)
    %   tmplt_vect - template matrix (n_templates x n_features)
    %   maxdist    - maximum distance threshold (scalar or 1 x n_templates vector)
    %   par        - parameter struct; relevant fields:
    %                  par.pk_weight  (default 1)   - weight boost at spike peak
    %                  par.amp_dir    (default 'neg')- polarity of peak ('neg'|'pos')
    %
    % Optional name-value pairs:
    %   'pointdist'  - per-template std-dev matrix, same size as tmplt_vect
    %                  (default: [], disables pointwise filter)
    %   'pointlimit' - max number of per-dim violations allowed (default: Inf)
    %   'k'          - number of nearest neighbors to return (default: [], returns 1)

    p = inputParser;
    addParameter(p, 'pointdist',  [],  @(x) isnumeric(x));
    addParameter(p, 'pointlimit', Inf, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'k',          [],  @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'algo', 'algo0', @ischar);
    parse(p, varargin{:});

    pointdist  = p.Results.pointdist;
    pointlimit = p.Results.pointlimit;
    k          = p.Results.k;
    algo       = p.Results.algo;

    % if ~isfield(par, 'pk_weight'), par.pk_weight = 1;     end
    if ~isfield(par, 'amp_dir'),   par.amp_dir   = 'neg'; end

    % [normConst, w] = get_weight_vector(spike_x, par.pk_weight, par.amp_dir);
    % w_resize  = ones(size(tmplt_vect, 1), 1) * w;
    % distances = normConst * sqrt(sum(w_resize .* (ones(size(tmplt_vect,1),1)*spike_x - tmplt_vect).^2, 2)');

    % distances = sqrt(sum((ones(size(tmplt_vect,1),1)*spike_x - tmplt_vect).^2, 2)');
    spike_width = compute_peak_width(spike_waveform, par.amp_dir);

    for i = 1:size(tmplt_vect, 1)
        template_width(i) = compute_peak_width(tmplt_vect(i,:), par.amp_dir);
        distances = compute_weighted_distance(spike_x, tmplt_vect, spike_width, template_width(i), algo, par,50,25);

    end
    conforming = find(distances < maxdist);

    % Pointwise distance filter (optional)
    if ~isempty(pointdist)
        pointwise_conforming = [];
        for i = 1:size(tmplt_vect, 1)
            if sum(abs(spike_x - tmplt_vect(i,:)) > pointdist(i,:)) < pointlimit
                pointwise_conforming = [pointwise_conforming i]; %#ok<AGROW>
            end
        end
        conforming = intersect(conforming, pointwise_conforming);
    end

    if isempty(conforming)
        index = 0;
    else
        if ~isempty(k)
            [~, i] = sort(distances(conforming));   % k-nearest neighbors
            i = i(1:min(length(i), k));
        else
            [~, i] = min(distances(conforming));
        end
        index = conforming(i);
    end
end

function distance = compute_weighted_distance(spike,template,spike_width,template_width,algo,par, varargin)    % get_weight_vector - Compute weight vector for distance calculation
    %
    % Inputs:
    %   algo      - algorithm type (string)
     if length(varargin) >= 2
        spike_wt = varargin{1};  % weight for spike region
        xor_wt = varargin{2};    % weight for XOR region (only used in spike_AND_vs_XOR)
    else
        spike_wt = 1;
        xor_wt = 3;
    end
    
    n = length(spike);
    spike_mask = false(1, n);
    template_mask = false(1, n);
    
    % Create masks from width struct
    if isstruct(spike_width) && ~isnan(spike_width.left) && ~isnan(spike_width.right)
        spike_mask(spike_width.left:min(spike_width.right, n)) = true;
    end
    if isstruct(template_width) && ~isnan(template_width.left) && ~isnan(template_width.right)
        template_mask(template_width.left:min(template_width.right, n)) = true;
    end
    
    % Initialize weights
    weights = ones(1, n);
    
    % Apply weighting based on variant
    % Key: each variant applies weights differently, but ALL use the same approach as get_weight_matrix
    switch algo
        case 'algo0'
            % Only weight spike region
            
        case 'algo1'
            % Only weight template region
            weights(spike_mask) = spike_wt;
            
        case 'algo2'
            % Only weight overlap region
            overlap = spike_mask & template_mask;
            weights(overlap) = spike_wt;
            
        case 'algo3'
            % Weight union of spike and template regions
            overlap = spike_mask & template_mask;
            xor_region = (spike_mask | template_mask) & ~overlap;
            weights(overlap) = spike_wt;       % AND region
            weights(xor_region) = xor_wt;      % XOR region
                        
        case 'algo4'
            overlap = spike_mask | template_mask;
            weights(overlap) = spike_wt;

        case 'algo5'
            weights(template_mask) = spike_wt;
            
        otherwise
            % No weighting
    end

end

function width_struct = compute_peak_width(spike_x, amp_dir)
    if strcmp(amp_dir, 'neg')
        wav = -spike_x;
    else
        wav = spike_x;
    end
    [pks, locs, w, p] = findpeaks(wav);
    
    if ~isempty(pks)
        [pk, peak_loc] = max(pks);
        pk_loc = locs(peak_loc);
        prom_max = p(peak_loc);

        level = pk - prom_max/2;
        above_level = find(wav >= level);
        if ~isempty(above_level)
            % Find connected components
            diff_above = diff(above_level);
            breaks = find(diff_above > 1);
            segments = {};
            start_idx = 1;
            for b = 1:length(breaks)
                segments{end+1} = above_level(start_idx:breaks(b));
                start_idx = breaks(b) + 1;
            end
            segments{end+1} = above_level(start_idx:end);
            % Find segment containing peak_loc
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