function corrMat = calculateOverlapRowCorrelation(overlapMatFull, GaussIDs)
% CALCULATEOVERLAPROWCORRELATION calculates the correlation between rows of the
% overlap matrix, skipping columns based on Gaussian IDs and zero values.
    
    n = size(overlapMatFull, 1);
    corrMat = eye(n); % Initialize with 1s on the diagonal
    k_indices = 1:n;
    
    for i = 1:n
        Ri = overlapMatFull(i, :);
        for j = i+1:n
            Rj = overlapMatFull(j, :);
            
            
            % We want to correlate the *pattern* of overlap with *other* % components, not the known diagonal '1's.
            mask_indices = (k_indices ~= i) & (k_indices ~= j);

            mask_non_zero = (Ri ~= 0) | (Rj ~= 0);
            
         
            valid_k = mask_indices & mask_non_zero;
            

            % Check if there are enough points for correlation (min 2)
            if sum(valid_k) > 1
                
                % Extract the relevant components for correlation
                Ri_masked = Ri(valid_k);
                Rj_masked = Rj(valid_k);
                
                % Check for zero variance, which also causes NaN
                if var(Ri_masked) == 0 || var(Rj_masked) == 0
                    corr_val = NaN; % One of the vectors is constant
                else
                    % Calculate Pearson correlation coefficient
                    C = corr([Ri_masked', Rj_masked']);
                    corr_val = C(1, 2);
                end
            else
                % Not enough valid columns to compute a meaningful correlation
                corr_val = NaN; 
            end
            
            corrMat(i, j) = corr_val;
           % corrMat(j, i) = corr_val;
        end
    end
end