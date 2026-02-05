function index = nearest_neighbor(spike_x,tmplt_vect,maxdist,par,varargin)
    % function index = nearest_neigbor(x,vectors,maxdist,pointdist*,pointlimit*,k*)
    % spike_x is a row vector
    % pointdist (optional) - vector of standard deviations
    % pointlimit (optional) - upper bound on number of points outside pointdist
    % k (optional) - number of points used for nearest neighbor

    % Find the distance to all neighbors. Consider only those neighbors where
    % the point falls in the radius of possibility for that point. Find the
    % nearest possible neighbor.
    % Return 0 if there is no possible nearest neighbor.
    if ~isfield(par,'pk_weight')
        par.pk_weight = 1;
    end
    if ~isfield(par,'amp_dir')
        par.amp_dir = 'neg';
    end
    [normConst,w] =get_weight_vector(spike_x,par.pk_weight,par.amp_dir);
    w_resize = ones(size(tmplt_vect,1),1)*w;
    distances = normConst* sqrt(sum(w_resize.*(ones(size(tmplt_vect,1),1)*spike_x - tmplt_vect).^2,2)');
    conforming = find(distances < maxdist);
    if( length(varargin) > 0 )
        pointdist = varargin{1};
        if( length(varargin) > 1 )
            pointlimit = varargin{2};
        else
            pointlimit = Inf;
        end
        pointwise_conforming = [];
        for i=1:size(tmplt_vect,1),
            if( sum( abs(spike_x-tmplt_vect(i,:)) > pointdist(i,:) ) < pointlimit )  % number of deviations from pointdist allowed.
                pointwise_conforming = [pointwise_conforming i];
            end
        end
        conforming = intersect(conforming, pointwise_conforming);
    end
    if( length( conforming ) == 0 )
        index = 0;
    else
        if( length(varargin) > 2 )
            k = varargin{3};
            [y i] = sort(distances(conforming)); % k-nearest neighbors
            i = i(1:min(length(i),k));
        else
            [y i] = min(distances(conforming));   
        end
        index = conforming(i);
    end

    function [normConst,weights] = get_weight_vector(spike_x,pk_weight,amp_dir)
        if amp_dir == 'neg'
            wav = -spike_x;
        else
            wav = spike_x;
        end
        [pks,locs,w,p] = findpeaks(wav);
        w_vect = ones(size(spike_x,2),1);
        
        [pk,peak_loc] = max(pks);
        width_max = w(peak_loc);
        peak_loc = locs(peak_loc);
    
        left_width = round(peak_loc - width_max/2);
        if left_width <= 0
            left_width = 1;
        end
        right_width = round(peak_loc+ width_max/2);
        if right_width > size(spike_x,2)
            right_width = size(spike_x,2);
        end

        % if width_max < length(spike_x)/2.5
        %     w_vect(left_width:right_width) = pk_weight;
        % end
        
        weights = w_vect';
        normConst = sqrt(length(weights))/sqrt(sum(weights));
        
    end

end