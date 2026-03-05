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
    parse(p, varargin{:});

    pointdist  = p.Results.pointdist;
    pointlimit = p.Results.pointlimit;
    k          = p.Results.k;

    if ~isfield(par, 'pk_weight'), par.pk_weight = 1;     end
    if ~isfield(par, 'amp_dir'),   par.amp_dir   = 'neg'; end

    % [normConst, w] = get_weight_vector(spike_x, par.pk_weight, par.amp_dir);
    % w_resize  = ones(size(tmplt_vect, 1), 1) * w;
    % distances = normConst * sqrt(sum(w_resize .* (ones(size(tmplt_vect,1),1)*spike_x - tmplt_vect).^2, 2)');

    distances = sqrt(sum((ones(size(tmplt_vect,1),1)*spike_x - tmplt_vect).^2, 2)');
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

    % function [normConst, weights] = get_weight_vector(spike_x, pk_weight, amp_dir)
    %     if strcmp(amp_dir, 'neg')
    %         wav = -spike_x;
    %     else
    %         wav = spike_x;
    %     end
    %     [pks, locs, w, ~] = findpeaks(wav);
    %     w_vect = ones(1, length(spike_x));

    %     if ~isempty(pks)
    %         [~, peak_loc] = max(pks);
    %         width_max = w(peak_loc);
    %         pk_loc    = locs(peak_loc);

    %         left_width  = max(1,              round(pk_loc - width_max/2));
    %         right_width = min(length(spike_x), round(pk_loc + width_max/2));

    %         w_vect(left_width:right_width) = pk_weight;
    %     end

    %     weights   = w_vect;
    %     normConst = sqrt(length(weights)) / sqrt(sum(weights));
    % end

end