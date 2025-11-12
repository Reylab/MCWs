function [inspk,coeffs] = gmm_extract(spikes,par)
    
    inspk = [];
    coeffs = [];

    filename = par.filename;
    files_in_dir = dir();
    split_file = strfind(filename,'_');
    basename = extractBefore(filename,split_file(end));
    
    try
        % remove _spikes
        folder = pwd;
        folderName = fullfile(folder, 'gmm');
        if ~exist(folderName, 'dir')
            mkdir(folderName);
        end
        fileNameGMM = append(basename,'_gmm.mat');
        file_gmm = fullfile(folderName,fileNameGMM);
        if ~exist(file_gmm, 'file')
            [g,xg,pg,wd_coeffs,medDist,summary_table] = GMM_1channel(spikes,par);
            save(file_gmm, "summary_table", "g", "pg", "xg", ...
                 "wd_coeffs")
        else
            load(file_gmm);
        end
        g_init = cell(1,length(g));
        medDist_cell = cell(1,length(g));
        kj_mat = cell(1,length(g));
        M_comp = cell(1,length(g));
        polyID = cell(1,length(g));
        medD_sortAll = cell(1,length(g));
        medD_sIdxAll = cell(1,length(g));
        kj_sortAll = cell(1,length(g));
        kj_sIdxAll = cell(1,length(g));
        idistVals = zeros(length(g),1);
        max_k = zeros(length(g),1);
        medDist_init = summary_table.("med idist");
        Mcomp_init = summary_table.M_comp;
        kj_init = summary_table.kj;
        g_init = g;
        for i = 1:length(g)
            g_in = g{i};
            mu = g_in.mu(:);                       
            s  = sqrt(squeeze(g_in.Sigma(:)));      
            a  = g_in.ComponentProportion(:);
            polyID_in = (1:numel(mu))';
          
            keep = a > 0.005;
            mu = mu(keep);
            s  = s(keep);
            a  = a(keep);
            polyID{i} = polyID_in(keep);
            Knew = numel(mu);
            a_new = a / sum(a);
            s_new = reshape(s.^2, 1, 1, Knew);
            
            medDist_cell{i} = medDist_init{i}(keep);
            kj_mat{i} = kj_init{i}(keep);
            g{i} = gmdistribution(mu,s_new,a_new);
            M_comp{i} = Mcomp_init{i}(ismember(Mcomp_init{i},polyID{i}));
            pg2 = pdf(g{i}, xg{i});
            [medD_sortAll{i},medD_sIdxAll{i}] = sort(medDist_init{i},'Descend');
            [kj_sortAll{i},kj_sIdxAll{i}] = sort(kj_init{i},'Descend');
            
            if Knew > 0
                idistVals(i) = max(medDist_cell{i});
                max_k(i) = max(kj_mat{i});
            end
        end
        %%
        [maxK_sort(:,2),maxK_sort(:,1)] = sort(max_k,"descend");
        
        [idist_sort(:,2),idist_sort(:,1)] = sort(idistVals,"descend");
        k_sort = cell(size(kj_mat));
        polyID_M = polyID;
        k_sortIdx = cell(size(kj_mat));
        for r = 1:numel(k_sort)
            vals = kj_mat{r};
            ids  = polyID_M{r};
            
            mask = ~isnan(vals);
            vals_valid = vals(mask);
            ids_valid  = ids(mask);
            
            [vals_sorted, idx] = sort(vals_valid, 'descend');
            k_sort{r} = vals_sorted;
            k_sortIdx{r} = ids_valid(idx);
        end
        medDist_sort = cell(size(medDist_cell));
        medDist_sortIdx = cell(size(medDist_cell));
        
        for r = 1:numel(medDist_cell)
            vals = medDist_cell{r};
            ids  = polyID_M{r};
            
            mask = ~isnan(vals);
            vals_valid = vals(mask);
            ids_valid  = ids(mask);
            
            [vals_sorted, idx] = sort(vals_valid, 'descend');
            
            medDist_sort{r} = vals_sorted;
            medDist_sortIdx{r} = ids_valid(idx);
        end
        %% exclude medDist values that are part of the combined pdf
        numRows = numel(M_comp);
        
        medDist_sort_masked    = cell(size(medDist_sort));
        medDist_sortIdx_masked = cell(size(medDist_sortIdx));
        
        k_sort_masked = cell(size(k_sort));
        k_sortIdx_masked = cell(size(k_sortIdx));
        for r = 1:numRows
            vals = medDist_sort{r};
            ids  = medDist_sortIdx{r};
            idx_exclude = M_comp{r};
            
            mask = ~ismember(ids, idx_exclude) & ~isnan(vals);
            
            medDist_sort_masked{r}    = vals(mask);
            medDist_sortIdx_masked{r} = ids(mask);
        end
        
        for r = 1:numRows
            vals = k_sort{r};
            ids  = k_sortIdx{r};
            idx_exclude = M_comp{r};
            
            mask = ~ismember(ids, idx_exclude) & ~isnan(vals);
            
            k_sort_masked{r}    = vals(mask);
            k_sortIdx_masked{r} = ids(mask);
        end
        % Replace originals
        k_sort = k_sort_masked;
        k_sortIdx = k_sortIdx_masked;
        medDist_sort    = medDist_sort_masked;
        medDist_sortIdx = medDist_sortIdx_masked;
        %% calculate knees
        [k_select,k_lSel,kDist_vec] = processKvKnee(k_sort, k_sortIdx);
        [medDist_select, medDist_lSel,medDist_vec] = processMedDistKnee(medDist_sort, medDist_sortIdx);
        %% boundary creation 
        [select_gauss,select_gauss1pct,select_gauss2p5pct,...
        select_gauss_5pct, select_gauss_10pct, select_all, ...
         x_vals, y_vals, gauss_ids, x_inter, y_inter] = lineExclusion(medDist_vec, ...
            medDist_select, k_select, kDist_vec, basename);
        %% redundancy removal
        threshQ2_10pct = select_all.trim10_0pct.threshQ2;
        threshQ4_10pct = select_all.trim10_0pct.threshQ4;
        
        select_spike_match = spikeMatch(select_gauss_10pct, ...
            threshQ2_10pct, threshQ4_10pct, g_init, wd_coeffs, ...
            basename, 5, 0.9);
        coeffs = unique(select_spike_match(:,3));
        inspk = wd_coeffs(:,coeffs);
    catch ME
        fprintf(2, '\n*** FAILED DATASET: %s ***\n', basename);
        fprintf(2, 'Error: %s\n\n', ME.message);
    end
end