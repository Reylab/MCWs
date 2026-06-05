function [g,xg,pg,coeffs,medDist,summary_out] = GMM_1channel(spikes,par)
    coeffs = haar_coeffs(spikes,par);
 

    m = size(coeffs,2);

   
    medDist = cell(2,m);

    g      = cell(1,m); 
    xg     = cell(1,m);           
    pg     = cell(1,m);           

    summary_gmm = cell(10,m);

    coeff_interest = 1;
    for j = 1:m
        Mj = GMM_basic1D(coeffs(:,j),par);

        [kj_max,kj_idx] = sort(Mj.dist_kj,'descend');
        [idist_max,idist_idx] = sort(Mj.Dij,'descend');


        summary_gmm{1,j} = j;
        summary_gmm{2,j} = Mj.dist_kj;
        summary_gmm{3,j} = Mj.Dij;
        summary_gmm{4,j} = Mj.med_dist;
        summary_gmm{5,j} = Mj.mu_mpdf;
        summary_gmm{6,j} = Mj.s_mpdf;
        summary_gmm{7,j} = Mj.g.mu;
        summary_gmm{8,j} = Mj.g.Sigma;
        summary_gmm{9,j} = Mj.Mcompon;
        summary_gmm{10,j} = Mj.polyID;


        medDist{1,j} = Mj.med_dist;
        medDist{2,j} = Mj.comp_medMax;


        g{j}       = Mj.g;
        xg{j}      = Mj.xg;
        pg{j}      = Mj.pg;

    end
    medDist = medDist';

    summary_gmm = summary_gmm';
    summary_out = cell2table(summary_gmm,'VariableNames', {'coeff num', 'kj', ...
        'Dij', ...
        'med idist','mpdf mu','mpdf std','mu gmm','std gmm','M_comp', ...
        'polyID'});
    
end

function M = GMM_basic1D(x,par)
    % One 1-D feature vector x (n x 1) -> metrics + peak/inflection locations
    x = x(:);

    opts  = statset('MaxIter',3000,'TolFun',1e-5,'Display','off');
    K     = 16;
    Mgrid = 100;                 % 100 for paper try 256 if missing def.

    g = fitgmdist(x, K, 'Replicates', 5, ...
        'CovarianceType','diagonal', ...
        'RegularizationValue', 1e-5, ...    
        'Start','plus', 'Options', opts);

    % Grid + pdf


    mu = g.mu(:);                       
    s  = sqrt(squeeze(g.Sigma(:)));      
    a  = g.ComponentProportion(:);    
    polyID = (1:numel(mu))';   % assign original indices from g

    
    
    xg = linspace(min(x), max(x), Mgrid).';
    pg = pdf(g, xg);

    [pk_vals, pk_locs, pk_widths] = findpeaks(pg, xg, 'WidthReference', 'halfheight');
    [~, tallest_idx] = max(pk_vals);
    main_peak_center = pk_locs(tallest_idx);
    main_peak_width  = pk_widths(tallest_idx);
    left_bound  = main_peak_center - main_peak_width/2;
    right_bound = main_peak_center + main_peak_width/2;

    
    
    % [a_sort,a_idx] = sort(a,'descend');
    % dropThreshold = 4;
    % a_drops = (a_sort(1:end-1)-a_sort(2:end))*100;
    % a_cutoff = find(a_drops > dropThreshold,1);
    % 
    % if isempty(a_cutoff) || isnan(a_cutoff)
    %     % if no cutoff found, take top 3 instead
    %     topN = min(3, numel(a_sort));
    %     a_top_idx = a_idx(1:topN);
    % else
    %     a_top_idx = a_idx(1:min(a_cutoff, numel(a_sort)));
    % end
 %   mu_top = mu(a_top_idx);


    in_main_peak = (mu >= left_bound) & (mu <= right_bound);
    a_top_idx = find(in_main_peak);
    mu_sort = mu(a_top_idx);
    s_sort = s(a_top_idx);
    polyID_M = polyID(a_top_idx);
    M_pdf = 0;
    a_top = a(a_top_idx);
    % combine biggest functions for consolidation metric

    scale_a = a_top/sum(a_top);
    for i =1:length(a_top)

        Npdf_i = normpdf(xg,mu_sort(i),s_sort(i)); %scale w a_top(i)?
        M_pdf = M_pdf + scale_a(i)*Npdf_i;     
     end


    pgmax = max(pg);
    dx = mean(diff(xg));

    % local maxima for peak detection (alternative) but can return
    % prominence, and input minProminence metric. paper uses crit pts of
    % pdf
    
    % promScale = 0.01;
    % s_comp   = sqrt(squeeze(g.Sigma(:)));
    % s_med    = median(s_comp);                     % typical component width
    % minDist  = max(1, round(0.5 * s_med / max(dx,realmin)));
    % 
    % [pks, locs] = findpeaks(pg, 'MinPeakProminence', promScale*max(pg), ...
    %                               'MinPeakDistance',  minDist);
    % isPeak = false(size(pg)); isPeak(locs) = true;

    % derivative to find critical pts for inflection/peaks
    p1  = gradient(pg, dx);
    p2  = gradient(p1, dx);
    
    s1 = p1(1:end-1);
    s2 = p1(2:end);
    crossPosToNeg = (s1 > 0  & s2 <= 0) | (s1 >= 0 & s2 < 0);

    crossLoc = find(crossPosToNeg);  
    % choose the endpoint closer to the zero 
    pick_loc = abs(p1(crossLoc+1)) < abs(p1(crossLoc));
    idx = crossLoc + pick_loc;

    isPeak = false(size(pg));
    validPeak = false(size(idx));
    if ~isempty(idx)
        validPeak = (p2(idx) < 0);             % concave down
        isPeak(idx(validPeak)) = true;
    end

    % inflections look for sign change or 0
    t1 = p2(1:end-1);
    t2 = p2(2:end);
    infl = (t1 >= 0 & t2 <= 0) | (t1 <= 0 & t2 >= 0);

    isInfl = false(size(pg));
    isInfl(2:end) = infl;                      % mark right endpoint

    Knew = numel(mu);               % actual surviving components


    [II,JJ] = find(triu(true(Knew),1));

    mu_M = sum(xg .* M_pdf) / sum(M_pdf);  % weighted mean over xg
    std_M = sqrt(sum((xg - mu_M).^2 .* M_pdf) / sum(M_pdf));
    kj = abs(mu - mu_M)/std_M;
    
    Dij = (abs(mu(II) - mu(JJ))).* sqrt(a(II) .* a(JJ)) ...
            ./sqrt(s(II) .* s(JJ));
    dist_per_comp = accumarray([II; JJ], [Dij; Dij], [K 1], @(v){v}, {});
    %avg_dist = cellfun(@mean, dist_per_comp);
    med_dist = cellfun(@median,dist_per_comp);
    %formally idist = median(dij)
    % [~, comp_max] = max(avg_dist);
    % Idist = mean(maxk(avg_dist,2));
    %[Idist, comp_max] = max(avg_dist);
    [Idist,comp_medMax] = max(med_dist);



    % if Idist_med > 1.55*min(med_dist)
    %     coeff_oInterMed = 1;
    % else
    %     coeff_oInterMed = 0;
    % end
    % Package extras


    %output idist max and poly from max idist

    M = struct('Dij',Dij, ...
           'g',g,'xg',xg, 'pg',pg, ...
            'med_dist',med_dist, ...
            'comp_medMax',comp_medMax, ...
            'dist_kj',kj,'mu_mpdf',mu_M,'s_mpdf',std_M,'Mcompon', polyID_M, ...
            'polyID',polyID);
end


function Xwav = haar_coeffs(spikes,par)
    scales = par.scales;
    feature = par.features;
    nspk = size(spikes,1);
    ls = size(spikes,2);
    cc=zeros(nspk,ls);
		try
			spikes_l = reshape(spikes',numel(spikes),1);
			if exist('wavedec')
				[c_l,l_wc] = wavedec(spikes_l,scales,'haar');
			else
				[c_l,l_wc] = fix_wavedec(spikes_l,scales);
			end
			wv_c = [0;l_wc(1:end-1)];
			nc = wv_c/nspk;
			wccum = cumsum(wv_c);
			nccum = cumsum(nc);
			for cf = 2:length(nc)
				cc(:,nccum(cf-1)+1:nccum(cf)) = reshape(c_l(wccum(cf-1)+1:wccum(cf)),nc(cf),nspk)';
			end
		catch
		    if exist('wavedec')                             % Looks for Wavelets Toolbox
				for i=1:nspk                                % Wavelet decomposition
					[c,l] = wavedec(spikes(i,:),scales,'haar');
					cc(i,1:ls) = c(1:ls);
				end
			else
				for i=1:nspk                                % Replaces Wavelets Toolbox, if not available
					[c,l] = fix_wavedec(spikes(i,:),scales);
					cc(i,1:ls) = c(1:ls);
				end
			end
		
		end
        Xwav = cc;
end

function [mu_new,s_new,a_new,polyID_new] = mergeNearby(mu,s,a,polyID,tol)
    % pairwise distance between means
    n = numel(mu);
    merged = false(1,n);
    mu_new = []; s_new = []; a_new = []; polyID_new = {};

    for i = 1:n
        if merged(i), continue; end
        % find group close to mu(i)
        group = find(abs(mu - mu(i)) < tol*mean(s));
        merged(group) = true;

        % combine weights
        a_sum = sum(a(group));
        % weighted mean
        mu_c = sum(a(group).*mu(group)) / a_sum;
        % pooled variance
        var_c = sum(a(group).*(s(group).^2 + mu(group).^2))/a_sum - mu_c^2;
        s_c = sqrt(max(var_c,eps));

        % store
        mu_new(end+1,1) = mu_c;
        s_new(end+1,1)  = s_c;
        a_new(end+1,1)  = a_sum;
        polyID_new{end+1,1} = polyID(group);
    end
end


