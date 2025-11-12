function [k_select,k_lSel,kDist_vec] = processKvKnee(k_sort, k_sortIdx)
% processMedDistKnee:
% Applies knee detection to each cell of medDist_sort using the same algorithm
% as processIdistKnee. Assumes each medDist_sort{k} is sorted ascending.

    numCells = numel(k_sort);
    k_lSel   = zeros(numCells, 1);

    % expands data into 2D vector, col 1 is medDist
    % and col 2 is coeff num (only used for plotting)
  % Calculate total number of elements for proper pre-allocation
    totalElements = sum(cellfun(@length, k_sort));
    k_vector = zeros(totalElements, 3);
    count = 0;
    
    % Expand all medDist to one vector
    for k = 1:numCells
        kV_k = k_sort{k};
        kv_Gidx = k_sortIdx{k};
        numElements = length(kV_k);
        
        % Fill the appropriate section
        indices = count + (1:numElements);
        k_vector(indices, 1) = kV_k;          % medDist values
        k_vector(indices, 2) = k;                  % cell indices
        k_vector(indices,3) = kv_Gidx;
        count = count + numElements;
    end
    
    % Sort by the first column (medDist values)
    % remove zeros k_vector
    k_vector = k_vector(k_vector(:, 1) ~= 0, :); % Remove rows with zero values
    [kVector_sort, kVectorIdx] = sort(k_vector(:, 1));
    kVector_sortCoeff = k_vector(kVectorIdx, 2);
    kVector_sortGauss = k_vector(kVectorIdx, 3);    
    
    kDist_vec = [kVector_sort,kVector_sortCoeff, ...
        kVector_sortGauss];
    
    sorted_vals = kVector_sort;
    sorted_idx  = kVector_sortCoeff;
    sorted_gauss = kVector_sortGauss;

    minVal = 1;
    maxVal = length(kVector_sort);
    top_candidates = sorted_vals(end-maxVal+1:end);
    top_indices    = sorted_idx(end-maxVal+1:end);
    top_gauss      = sorted_gauss(end-maxVal+1:end);
    ncoeff = length(sorted_vals);
    maxA = max(sorted_vals);
    
    nd = max(10,round(maxVal/10));
    if ncoeff >= nd
        d_vals = (top_candidates(nd:end) - top_candidates(1:end-nd+1)) ...
                 / maxA * ncoeff / nd;
        all_above1 = find(d_vals >= 1);
    else
        all_above1 = [];
    end
    
    if numel(all_above1) >= 2
        aux2 = diff(all_above1);
        temp_bla = conv(aux2(:), [1 1 1]/3); % smooth differences
        temp_bla = temp_bla(2:end-1);
        temp_bla(1) = aux2(1);
        temp_bla(end) = aux2(end);
    
        % First position where 3 consecutive differences == 1
        thr_knee_diff = all_above1(find(temp_bla(2:end) == 1, 1)) + round(nd/2);
    
        % Number of inputs to select
        inputs = maxVal - thr_knee_diff + 1;

    end
    
    leng_select = length(top_indices(end:-1:end-inputs+1));
    k_select = zeros(leng_select, 3);

    coeff_idx = top_indices(end:-1:end-inputs+1);
    gauss_idx = top_gauss(end:-1:end-inputs+1);
    k_select(:,2) = coeff_idx;           % coefficient indices
    k_select(:,1) = kVector_sort(end:-1:end-inputs+1);% corresponding values
    k_select(:,3) = gauss_idx;
    k_lSel = length(coeff_idx);

end
