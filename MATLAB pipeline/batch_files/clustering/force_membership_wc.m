function class_out = force_membership_wc(f_in, class_in, f_out, par)
% ... (original comments) ...
%                    o par.spike_type     - 'pos' or 'neg' (default 'neg')
%                    o par.peak_weight    - scaling factor for peak width (default 1)

% Set defaults for new weight parameters
if ~isfield(par, 'spike_type'), par.spike_type = 'neg'; end
if ~isfield(par, 'peak_weight'), par.peak_weight = 3; end

nspk = size(f_out,1);
class_out = zeros(1,size(f_out,1));

switch par.template_type
    case 'nn'
        % (Leaving nn as is for now unless you want weights here too)
        sdnum = par.template_sdnum;
        k     = par.template_k;
        k_min = par.template_k_min;
        sd    = sqrt(sum(var(f_in,1)))*ones(1,size(f_in,1));
        for i=1:nspk,
            nn = nearest_neighbor(f_out(i,:),f_in,sdnum*sd,Inf*ones(size(f_in)),Inf,k);
            if( nn )
                winner = mode(class_in(nn));
                if nnz(class_in(nn)==winner)<k_min
                    class_out(i) = 0;
                else
                    class_out(i) = winner;
                end
            else
                class_out(i) = 0;
            end
        end
      
    case 'center'
        % 1. Get templates and the peak-based weight matrix
        [centers, sd, pd, weights] = build_templates(class_in, f_in, par.spike_type, par.peak_weight); 
        
        sdnum = par.template_sdnum;
        for i=1:nspk,
            % 2. Pass weights as the 4th optional argument (varargin{4})
            % Note: we pass empty [] for pointdist, pointlimit, and k to reach the 4th slot
            class_out(i) = nearest_neighbor(f_out(i,:), centers, sdnum*sd, 'weights', weights);
        end
        
    case 'ml'
        % (original ml logic)
        [mu, inv_sigma] = fit_gaussian(f_in,class_in);
        for i=1:nspk,
            class_out(i) = ML_gaussian(f_out(i,:),mu,inv_sigma);
        end
        
    case 'mahal'
        % (original mahal logic)
        classes = unique(class_in);
        mdistance = zeros(length(classes), nspk);
        maxdist   = zeros(1, length(classes));
        for ci = 1:length(classes)
           i = classes(ci);
           mdistance(i,:) = mahal(f_out, f_in(class_in ==i, :));
           maxdist(i) = sqrt(mean(mahal(f_in(class_in ==i, :), f_in(class_in ==i, :))));
        end
        sdnum = par.template_sdnum;
        for i = 1:nspk
             [d, winner] = min(mdistance(:,i));
             if sqrt(d) < sdnum*maxdist(winner)
                 class_out(i) = classes(winner);
             end
        end
        
    otherwise
        sprintf('force_membership(): <%s> is not a known template type.\n',par.template_type);
        
end