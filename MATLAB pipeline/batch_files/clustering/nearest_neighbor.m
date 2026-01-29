function index = nearest_neighbor(x, vectors, maxdist, varargin)
    % Use the parser to avoid the varargin index confusion
    p = inputParser;
    addOptional(p, 'pointdist', []);
    addOptional(p, 'pointlimit', Inf);
    addOptional(p, 'k', 1); 
    addOptional(p, 'weights', ones(size(vectors)));
    parse(p, varargin{:});

    % Distance logic
    diffs_sq = (ones(size(vectors,1),1)*x - vectors).^2;
    % Weights applied here
    distances = sqrt(sum(diffs_sq .* p.Results.weights, 2)');

    conforming = find(distances < maxdist);

    % Pointwise logic
    if ~isempty(p.Results.pointdist)
        pw_conforming = [];
        for i = 1:size(vectors,1)
            if (sum(abs(x - vectors(i,:)) > p.Results.pointdist(i,:)) < p.Results.pointlimit)
                pw_conforming = [pw_conforming i];
            end
        end
        conforming = intersect(conforming, pw_conforming);
    end

    if isempty(conforming)
        index = 0;
    else
        [~, sorted_idx] = sort(distances(conforming));
        % This line ensures we only get ONE index for 'center' mode
        % because k defaults to 1
        index = conforming(sorted_idx(1:min(length(sorted_idx), p.Results.k)));
    end
end