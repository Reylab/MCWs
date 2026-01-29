function [templates, maxdist, pointdist, weights] = build_templates(classes, features, spike_type, peak_weight)
% function [templates maxdist pointdist weights] = build_templates(classes, features, spike_type, peak_weight)

if nargin < 3, spike_type = 'neg'; end
if nargin < 4, peak_weight = 2; end

max_class = max(classes);
feature_dim = size(features,2);
templates = zeros(max_class, feature_dim);
maxdist   = zeros(1,max_class);
pointdist = zeros(max_class,feature_dim);
weights   = ones(max_class, feature_dim); % Initialize all weights to 1

for i=1:max_class,
    fi = features(classes==i,:);
    avg_wave = mean(fi, 1);
    templates(i,:) = avg_wave;
    
    % --- Determine weights based on peak width ---
    % Invert for findpeaks if the input is negative spikes
    search_wave = avg_wave;
    if strcmp(spike_type, 'neg')
        search_wave = -avg_wave;
    end
    
    % Find the peak and its width
    [pks, locs, w] = findpeaks(search_wave);
    
    if ~isempty(pks)
        % Target the highest peak
        [~, max_idx] = max(pks);
        peak_loc = locs(max_idx);
        peak_width = w(max_idx);
        
        % Calculate start/end indices for the weighting
        idx_start = max(1, round(peak_loc - peak_width/2));
        idx_end   = min(feature_dim, round(peak_loc + peak_width/2));
        
        % Apply the scalar weight to the specific peak region
        weights(i, idx_start:idx_end) = peak_weight;
    end
    % ---------------------------------------------
    
    maxdist(i)     = sqrt(sum(var(fi,1)));
    pointdist(i,:)   = sqrt(var(fi,1));
end