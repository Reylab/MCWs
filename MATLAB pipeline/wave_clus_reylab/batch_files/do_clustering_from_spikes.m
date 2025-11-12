function [outfilename, status] = do_clustering_from_spikes(channame,spikes, index,min_spikes4SPC, par_input,save_spikes)
    outfilename = ['times_' channame '.mat'];
    status = false;
    par = set_parameters();
    par = update_parameters(par,par_input,'clus');
    par = update_parameters(par,par_input,'batch_plot');

    par.filename = channame;
    par.reset_results = true;
    safechanname = strrep(channame,' ','_');
    par.fname_in = ['tmp_data_wc' safechanname];     % temporary filename used as input for SPC
    par.fname = ['data_' channame];
    par.nick_name = channame;
    par.fnamespc = ['data_wc' safechanname];
        

    % LOAD SPIKES
    nspk = size(spikes,1);
    naux = min(par.max_spk,size(spikes,1));

    if nspk < min_spikes4SPC
        warning('MyComponent:noValidInput', 'Not enough spikes in the file');
        return
    end

    % CALCULATES INPUTS TO THE CLUSTERING ALGORITHM.
    inspk = wave_features(spikes,par);     %takes wavelet coefficients.
    par.inputs = size(inspk,2);                       % number of inputs to the clustering

	if par.permut == 'n'
        % GOES FOR TEMPLATE MATCHING IF TOO MANY SPIKES.
        if size(spikes,1)> par.max_spk;
            % take first 'par.max_spk' spikes as an input for SPC
            inspk_aux = inspk(1:naux,:);
        else
            inspk_aux = inspk;
        end
	else
        % GOES FOR TEMPLATE MATCHING IF TOO MANY SPIKES.
        if size(spikes,1)> par.max_spk;
            % random selection of spikes for SPC
            ipermut = randperm(length(inspk));
            ipermut(naux+1:end) = [];
            inspk_aux = inspk(ipermut,:);
        else
            ipermut = randperm(size(inspk,1));
            inspk_aux = inspk(ipermut,:);
        end
	end
    %INTERACTION WITH SPC
    save(par.fname_in,'inspk_aux','-ascii');
    try
        [clu, tree] = run_cluster(par,true);
		if exist([par.fnamespc '.dg_01.lab'],'file')
			movefile([par.fnamespc '.dg_01.lab'], [par.fname '.dg_01.lab'], 'f');
			movefile([par.fnamespc '.dg_01'], [par.fname '.dg_01'], 'f');
		end
    catch
        warning('MyComponent:ERROR_SPC', 'Error in SPC');
        return
    end

    [clust_num temp auto_sort] = find_temp(tree,clu,par);

    if par.permut == 'y'
        clu_aux = zeros(size(clu,1),2 + size(spikes,1)) -1;  %when update classes from clu, not selected elements go to cluster 0
        clu_aux(:,ipermut+2) = clu(:,(1:length(ipermut))+2);
        clu_aux(:,1:2) = clu(:,1:2);
        clu = clu_aux;
        clear clu_aux
    end

    classes = zeros(1,size(clu,2)-2);
    for c =1: length(clust_num)
        aux = clu(temp(c),3:end) +1 == clust_num(c);
        classes(aux) = c;
    end

    if par.permut == 'n'
        classes = [classes zeros(1,max(size(spikes,1)-par.max_spk,0))];
    end

    Temp = [];
    % Classes should be consecutive numbers
    classes_names = nonzeros(sort(unique(classes)));
    for i= 1:length(classes_names)
       c = classes_names(i);
       if c~= i
           classes(classes == c) = i;
       end
       Temp(i) = temp(i);
    end

    % IF TEMPLATE MATCHING WAS DONE, THEN FORCE
    if (size(spikes,1)> par.max_spk || ...
            (par.force_auto))
        f_in  = spikes(classes~=0,:);
        f_out = spikes(classes==0,:);
        class_in = classes(classes~=0);
        class_out = force_membership_wc(f_in, class_in, f_out, par);
        forced = classes==0;
        classes(classes==0) = class_out;
        forced(classes==0) = 0;
    else
        forced = zeros(1, size(spikes,1));
    end

    gui_status = struct();
    gui_status.current_temp =  max(temp);
    gui_status.auto_sort_info = auto_sort;
    gui_status.original_classes = zeros(size(classes));

    for i=1:max(classes)
        gui_status.original_classes(classes==i) = clust_num(i);
    end

    current_par = par;
    par = struct;
    par = update_parameters(par, current_par, 'relevant');
    par = update_parameters(par,current_par,'batch_plot');

    par.sorting_date = datestr(now);
    cluster_class = zeros(nspk,2);
    cluster_class(:,2)= index;
    cluster_class(:,1)= classes';
    
    vars = {'cluster_class', 'par','inspk','forced','Temp','gui_status'};
    if exist('ipermut','var')
        vars{end+1} = 'ipermut';
    end
    if save_spikes
        vars{end+1} = 'spikes';
    else
        spikes_file = filename;
        vars{end+1} = 'spikes_file';
    end
    
    try
      save(outfilename,vars{:});
    catch
      save(outfilename,vars{:},'-v7.3');
    end
    status = true;
end