function Do_clustering(input, varargin)

% PROGRAM Do_clustering.
% Does clustering on all files in Files.txt
% Runs after Get_spikes, which prepares '*_spikes.mat' files.
%
% function Do_clustering(input, par_input)
% Saves spikes, spike times (in ms), coefficients used (inspk), used
% parameters, random spikes selected for clustering (ipermut) and
% results (cluster_class).
%
%input must be:
%               A .txt file with the names of the '*_spikes.mat' files to use.
%               A matlab cell with the names of the '*_spikes.mat' files to use.
%               A vector with channel numbers. In this case the function 
%                   will proccess all the '*_spikes.mat' files located in 
%                   the folder with those channel numbers (e.g., CSC1_spikes.mat or NSX4_spikes.mat)
%               'all', in this case the functions will process all the
%                   '*_spikes.mat' files in the folder.
%               A string with a filename ended in '_spikes.mat'
% optional argument 'par' and the next input must be a struct with some of
%       the SPC parameters (the detection parameters are taken from the
%       XXX_spikes file or the set_parameters file. All the parameters included
%       in the structure par will overwrite the parameters loaded from set_parameters()
% optional argument 'parallel' : true for use parallel computing.
% optional argument 'make_times': true for computing the sorting from the XXX_spikes files.
% optional argument 'make_plots': true for plotting the results based on the save XXX_times files.
% optional argument 'resolution': resolution string used by the plots (default: '-r150').
% optional argument 'save_spikes': false for disable saving all the spikes
% waveforms on the times* file.
% See also
% Get_spikes


% Example
% param.min_clus = 60;
% param.max_spk = 50000;
% par.maxtemp = 0.251;
% Do_clustering(1:16,'parallel',true,'par',param)
%
% If the times were created but something went wrong with the plots. Then, the
% code can be called again just to create the plots:
% Do_clustering(1:16,'make_times',false,'par',param)
%
%
% Similarly, for don't create the plots:
% Do_clustering(1:16,'make_plots',false,'par',param)
%

% min_spikes4SPC = 16; % if are less that this number of spikes, clustering won't be made.
min_spikes4SPC = 64; % if are less that this number of spikes, clustering won't be made.

%default config
par_input = struct;
parallel = false;
make_times = true;
make_plots = true;
make_templates = false;
resolution = '-r150';
save_spikes = true;
%search for optional inputs
nvar = length(varargin);
for v = 1:nvar
    if strcmp(varargin{v},'par')
        if (nvar>=v+1) && isstruct(varargin{v+1})
            par_input = varargin{v+1};
        else
            error('Error in ''par'' optional input.')
        end
    elseif strcmp(varargin{v},'parallel')
        if (nvar>=v+1) && islogical(varargin{v+1})
            parallel = varargin{v+1};
        else
            error('Error in ''parallel'' optional input.')
        end
    elseif strcmp(varargin{v},'make_times')
        if (nvar>=v+1) && islogical(varargin{v+1})
            make_times = varargin{v+1};
        else
            error('Error in ''make_times'' optional input.')
        end
    elseif strcmp(varargin{v},'make_plots')
        if (nvar>=v+1) && islogical(varargin{v+1})
            make_plots = varargin{v+1};
        else
            error('Error in ''make_plots'' optional input.')
        end
    elseif strcmp(varargin{v},'make_templates')
        if (nvar>=v+1) && islogical(varargin{v+1})
            make_templates = varargin{v+1};
        else
            error('Error in ''make_templates'' optional input.')
        end
    elseif strcmp(varargin{v},'resolution')
        if (nvar>=v+1) && ischar(varargin{v+1})
            resolution = varargin{v+1};
        else
            error('Error in ''resolution'' optional input.')
        end
    elseif strcmp(varargin{v},'save_spikes')
        if (nvar>=v+1) && islogical(varargin{v+1})
            save_spikes = varargin{v+1};
        else
            error('Error in ''save_spikes'' optional input.')
        end

    end
end

run_par_for = parallel;
filenames = {};
% get a cell of filenames from the input
if isnumeric(input) || any(strcmp(input,'all'))  %cases for numeric or 'all' input
    
    filenames_all = {};
    dirnames = dir();
    dirnames = {dirnames.name};

    for i = 1:length(dirnames)
        fname = dirnames{i};

        if length(fname) < 12
            continue
        end
        if ~ strcmp(fname(end-10:end),'_spikes.mat')
            continue
        end
        if strcmp(input,'all')
            filenames = [filenames {fname}];
        else
            aux = regexp(fname(1:end-11), '\d+$', 'match');
            if ~isempty(aux) && ismember(str2num(aux{1}),input)
                filenames = [filenames {fname}];
            end
        end
        filenames_all = [filenames_all {fname}];
    end

elseif ischar(input) && length(input) > 4
    if  strcmp (input(end-3:end),'.txt')   %case for .txt input
        filenames =  textread(input,'%s');
    else
        filenames = {input};               %case for cell input
    end

elseif iscellstr(input)
    filenames = input;
else
    ME = MException('MyComponent:noValidInput', 'Invalid input arguments');
    throw(ME)
end

clustering_start_time = tic;
par_file = set_parameters();
if make_times
% open parallel pool, if parallel input is true
    if parallel == true
        if exist('matlabpool','file')
            try
                matlabpool('open');
            catch
                parallel = false;
            end
        else
            poolobj = gcp('nocreate'); % If no pool, do not create new one.
            if isempty(poolobj)
                parpool
            else
                parallel = false;
            end
        end
    end



    initial_date = now;
    Nfiles = length(filenames);
    if run_par_for == true
        parfor fnum = 1:Nfiles
            filename = filenames{fnum};
            begin_time = tic;
            do_clustering_single(filename,min_spikes4SPC, par_file, par_input,fnum,save_spikes);
            time_taken = toc(begin_time);
            fprintf('%d of %d ''times'' files (%s) done in %0.2f seconds.\n', ...
                count_new_times(initial_date, filenames),Nfiles, filename, time_taken)
        end
    else
        for fnum = 1:length(filenames)
            filename = filenames{fnum};
            begin_time = tic;
            do_clustering_single(filename,min_spikes4SPC, par_file, par_input,fnum,save_spikes);
            time_taken = toc(begin_time);
            fprintf('%d of %d ''times'' files (%s) done in %0.2f seconds.\n', ...
                count_new_times(initial_date, filenames),Nfiles, filename, time_taken)
        end
    end
    if parallel == true
        if exist('matlabpool','file')
            matlabpool('close')
        else
            poolobj = gcp('nocreate');
            delete(poolobj);
        end
    end

	log_name = 'spc_log.txt';
	f = fopen(log_name, 'w');
	for fnum = 1:length(filenames)
        filename = filenames{fnum};
        log_name = [filename 'spc_log.txt'];
        if exist(log_name, 'file')
			fi = fopen(log_name,'r');
			result = fread(fi);
			fwrite(f,result);
			fclose(fi);
			delete(log_name);
		end
    end
	fclose(f);

	time_taken = toc(clustering_start_time);
    time_taken_mins = floor(time_taken / 60);
    time_taken_secs = mod(time_taken, 60);
    fprintf('Clustering done in %d mins %0.0f seconds.\n', time_taken_mins, time_taken_secs)
end

make_templates_start_time = tic;
if make_templates
    cls_centers = cell(length(filenames_all),1);
    cls_maxdist = cell(length(filenames_all),1);
    channels = [];
    disp('Creating templates...')
    for fnum = 1:length(filenames_all)
        filename = filenames_all{fnum};
        
        par = struct;
        par.filename = filename;

        data_handler = readInData(par);
        par = data_handler.update_par(par);

        if ~data_handler.with_wc_spikes       			%data should have spikes
            continue
        end
        filename = data_handler.nick_name;
        timefile = ['times_' filename '.mat'];
        if ~exist(timefile,'file')
            cls_centers{fnum} = nan;
            cls_maxdist{fnum} = inf;
            channels(end+1) = str2num(filename(end-2:end));
            disp([timefile ' not found.'])
            continue
        end

        [~, ~, spikes, ~, ~, ~, classes, ~,~] = data_handler.load_results();

        f_in  = spikes(classes~=0,:);
        class_in = classes(classes~=0);

        [centers, sd, ~] = build_templates(class_in, f_in);
        cls_centers{fnum} = centers;
        cls_maxdist{fnum} = (par_input.max_std_templates*sd).^2;
        channels(end+1) = str2num(filename(end-2:end));
    end
    max_std = par_input.max_std_templates;
    save('templates_wc_offline.mat', 'cls_centers', 'cls_maxdist', 'channels', 'max_std')
    try
        copyfile('templates_wc_offline.mat', par_input.processing_rec_metadata)

    catch ME
        disp('Failed to copy templates to rec_metadata.')
        disp(ME.message)
    end
    make_templates_toc = toc(make_templates_start_time);
    fprintf("Templates creation took %s seconds. \n", num2str(make_templates_toc, '%2.2f'));
end



if make_plots
    make_plots_start_time = tic;
    disp('Creating figures...')
    numfigs = length(filenames);
    % curr_fig = figure('Visible','Off');
    % curr_fig2 = figure('Visible','Off');
    % set(curr_fig, 'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperPositionMode', 'auto','units','normalized','outerposition',[0 0 1 1],'RendererMode','manual','Renderer','painters')
    % set(curr_fig2, 'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperPositionMode', 'auto','units','normalized','outerposition',[0 0 1 1],'RendererMode','manual','Renderer','painters')
    % if isfield(get(curr_fig),'GraphicsSmoothing')
    %     set(curr_fig,'GraphicsSmoothing','off');
    %     set(curr_fig2,'GraphicsSmoothing','off');
    % end

    parfor fnum = 1:numfigs
        try
            fig_start_time = tic;
            curr_fig = figure('Visible','Off');
            curr_fig2 = figure('Visible','Off');
            set(curr_fig, 'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperPositionMode', 'auto','units','normalized','outerposition',[0 0 1 1],'RendererMode','manual','Renderer','painters')
            set(curr_fig2, 'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperPositionMode', 'auto','units','normalized','outerposition',[0 0 1 1],'RendererMode','manual','Renderer','painters')
            if isfield(get(curr_fig),'GraphicsSmoothing')
                set(curr_fig,'GraphicsSmoothing','off');
                set(curr_fig2,'GraphicsSmoothing','off');
            end
            % clf(curr_fig2)
            % clf(curr_fig)
            set(0, 'CurrentFigure', curr_fig)
            filename = filenames{fnum};
            par = struct;
            par.filename = filename;
    
            par.cont_segment = true;  %maybe true and save the sample in spikes
    
            data_handler = readInData(par);
            par = data_handler.update_par(par);
    
            par = update_parameters(par,par_file,'batch_plot');
            par = update_parameters(par,par_input,'batch_plot');
    
            if ~data_handler.with_wc_spikes       			%data should have spikes
                continue
            end
            filename = data_handler.nick_name;
    
            file_pos_names = {'','a','b','c','d','e','f'};
            for i=1:length(file_pos_names)
                new_file_name = ['fig2print_' filename file_pos_names{i} '.png'];
                if exist(new_file_name, 'file')==2
                    delete(new_file_name);
                end
            end
    
    
            subplot(3,1,1)
            if par.cont_segment && data_handler.with_psegment
                box off; hold on
                %these lines are for plotting continuous data
                [xd_sub, sr_sub] = data_handler.get_signal_sample();
                lx = length(xd_sub);
                plot((1:lx)/sr_sub,xd_sub)
                noise_std_detect = median(abs(xd_sub))/0.6745;
                xlim([0 lx/sr_sub])
                thr = par.stdmin * noise_std_detect;
                plotmax = 15 * noise_std_detect; %thr*par.stdmax/par.stdmin;
    
                if strcmp(par.detection,'pos')
                    line([0 length(xd_sub)/sr_sub],[thr thr],'color','r')
                    ylim([-plotmax/2 plotmax])
                elseif strcmp(par.detection,'neg')
                    line([0 length(xd_sub)/sr_sub],[-thr -thr],'color','r')
                    ylim([-plotmax plotmax/2])
                else
                    line([0 length(xd_sub)/sr_sub],[thr thr],'color','r')
                    line([0 length(xd_sub)/sr_sub],[-thr -thr],'color','r')
                    ylim([-plotmax plotmax])
                end
            end
            title([pwd '/' filename],'Interpreter','none','Fontsize',14)
    
            if ~data_handler.with_spc
                print2file = par_file.print2file;
                if isfield(par_input,'print2file')
                    print2file = par_input.print2file;
                end
                if print2file
                    % print(curr_fig,'-dpng',['fig2print_' filename '.png'],resolution);
                    F = getframe(curr_fig);
                    imwrite(F.cdata, ['fig2print_' filename '.png'])
                else
                    print(curr_fig)
                end
                % clear print2file
                fprintf('%d figs Done. ',fnum);
                continue
            end
    
            % LOAD SPIKES
            [clu, tree, spikes, index, inspk, ipermut, classes, forced,temp] = data_handler.load_results();
            nspk = size(spikes,1);
            Mclasses = max(classes);
            auto_sort_info = [];
            if data_handler.with_gui_status
                [original_classes, current_temp, auto_sort_info] = data_handler.get_gui_status();
                temperature = par.mintemp+current_temp*par.tempstep;
                org_class = zeros(1,Mclasses);
                for i = 1:Mclasses
                    org_class(i) = original_classes(find(classes==i,1,'first'));
                end
            else
                [temp] = find_temp_old(tree,par);
                temp = ones(max(classes),1)*temp;
                org_class = 1:max(classes);
                temperature = par.mintemp+temp*par.tempstep;
            end
    
    
            %PLOTS
            clus_pop = [];
            ylimit = [];
            subplot(3,5,11)
    
            color = [[0.0 0.0 1.0];[1.0 0.0 0.0];[0.0 0.5 0.0];[0.620690 0.0 0.0];[0.413793 0.0 0.758621];[0.965517 0.517241 0.034483];
            [0.448276 0.379310 0.241379];[1.0 0.103448 0.724138];[0.545 0.545 0.545];[0.586207 0.827586 0.310345];
            [0.965517 0.620690 0.862069];[0.620690 0.758621 1.]];
            maxc = size(color,1);
    
            hold on
            num_temp = floor((par.maxtemp -par.mintemp)/par.tempstep);     % total number of temperatures
            switch par.temp_plot
                    case 'lin'
                       if ~isempty(auto_sort_info)
                            [xp, yp] = find(auto_sort_info.peaks);
                            ptemps = par.mintemp+(xp)*par.tempstep;
                            psize = tree(sub2ind(size(tree), xp,yp+4));
                            plot(ptemps,psize,'xk','MarkerSize',7,'LineWidth',0.9);
                            area(par.mintemp+par.tempstep.*[auto_sort_info.elbow,num_temp],max(ylim).*[1 1],'LineStyle','none','FaceColor',[0.9 0.9 0.9]);
                        end
                        plot([par.mintemp par.maxtemp-par.tempstep], ...
                        [par.min_clus par.min_clus],'k:',...
                        par.mintemp+(1:num_temp)*par.tempstep, ...
                        tree(1:num_temp,5:size(tree,2)),[temperature temperature],[1 tree(1,5)],'k:')
                        for i=1:Mclasses
                            tree_clus = tree(temp(i),4+org_class(i));
                            tree_temp = tree(temp(i)+1,2);
                            plot(tree_temp,tree_clus,'.','color',color(mod(i-1,maxc)+1,:),'MarkerSize',20);
                        end
                    case 'log'
                        set(gca,'yscale','log');
                        if ~isempty(auto_sort_info)
                            [xp, yp] = find(auto_sort_info.peaks);
                            ptemps = par.mintemp+(xp)*par.tempstep;
                            psize = tree(sub2ind(size(tree), xp,yp+4));
                            semilogy(ptemps,psize,'xk','MarkerSize',7,'LineWidth',0.9);
                            area(par.mintemp+par.tempstep.*[auto_sort_info.elbow,num_temp],max(ylim).*[1 1],'LineStyle','none','FaceColor',[0.9 0.9 0.9],'basevalue',1);
                        end
                        semilogy([par.mintemp par.maxtemp-par.tempstep], ...
                        [par.min_clus par.min_clus],'k:',...
                        par.mintemp+(1:num_temp)*par.tempstep, ...
                        tree(1:num_temp,5:size(tree,2)),[temperature temperature],[1 tree(1,5)],'k:')
    
                        for i=1:Mclasses
                            tree_clus = tree(temp(i),4+org_class(i));
                            tree_temp = tree(temp(i)+1,2);
                            semilogy(tree_temp,tree_clus,'.','color',color(mod(i-1,maxc)+1,:),'MarkerSize',20);
                        end
            end
            xlim([par.mintemp, par.maxtemp])
            subplot(3,5,6)
            hold on
    
            tot_spikes = 0;
    
            class0 = find(classes==0);
                max_spikes=min(length(class0),par.max_spikes_plot);
                tot_spikes = tot_spikes + max_spikes;
                plot(spikes(class0(1:max_spikes),:)','k');
                xlim([1 size(spikes,2)]);
            subplot(3,5,10);
                hold on
                plot(spikes(class0(1:max_spikes),:)','k');
                plot(mean(spikes(class0,:),1),'c','linewidth',2)
                xlim([1 size(spikes,2)]);
                title(['Cluster 0: # ' num2str(length(class0))],'Fontweight','bold')
            subplot(3,5,15)
                plot_isi(index, class0, par.line_freq)
    
    
            for i = 1:max(classes)
                set(0, 'CurrentFigure', curr_fig)
                class = find(classes==i);
                subplot(3,5,6);
                max_spikes=min(length(class),par.max_spikes_plot);
                tot_spikes = tot_spikes + max_spikes;
                plot(spikes(class(1:max_spikes),:)','color',color(mod(i-1,maxc)+1,:));
                xlim([1 size(spikes,2)]);
    
                if i<=3
                    subplot(3,5,6+i);
                    hold on
                    plot(spikes(class(1:max_spikes),:)','color',color(mod(i-1,maxc)+1,:));
                    plot(mean(spikes(class,:),1),'k','linewidth',2)
                    xlim([1 size(spikes,2)]);
                    title(['Cluster ' num2str(i) ': # ' num2str(length(class)) ' (' num2str(nnz(classes(:)==i & ~forced(:))) ')'],'Fontweight','bold')
                    ylimit = [ylimit;ylim];
                    subplot(3,5,11+i)
                    plot_isi(index, class, par.line_freq)
    
                elseif i<=8
            	      set(0, 'CurrentFigure', curr_fig2)
                    subplot(3,5,2+i);
                    hold on
                    plot(spikes(class(1:max_spikes),:)','color',color(mod(i-1,maxc)+1,:));
                    plot(mean(spikes(class,:),1),'k','linewidth',2)
                    xlim([1 size(spikes,2)]);
                    title(['Cluster ' num2str(i) ': # ' num2str(length(class)) ' (' num2str(nnz(classes(:)==i & ~forced(:))) ')'],'Fontweight','bold')
                    ylimit = [ylimit;ylim];
                    subplot(3,5,7+i)
                    plot_isi(index, class, par.line_freq)
                end
            end
            numclus = max(classes);
            % Rescale spike's axis
            if ~isempty(ylimit)
                ymin = min(ylimit(:,1));
                ymax = max(ylimit(:,2));
                set(0, 'CurrentFigure', curr_fig)
                for i = 1:min(3,numclus)
                   subplot(3,5,6+i); ylim([ymin ymax]);
                end
                set(0, 'CurrentFigure', curr_fig2)
                for i = 4:min(8,numclus)
                   subplot(3,5,2+i); ylim([ymin ymax]);
                end
    
            end
    
            features_name = par.features;
    
            outfileclus='cluster_results.txt';
            fout=fopen(outfileclus,'at+');
            if isfield(par,'stdmin')
                stdmin = par.stdmin;
            else
                stdmin = NaN;
            end
            fprintf(fout,'%s\t %s\t %g\t %d\t %g\t', char(filename), features_name, temperature, numclus, stdmin);
            for ii=0:numclus
                fprintf(fout,'%d\t',nnz(classes==ii));
            end
            fclose(fout);
            figure_created_toc = toc(fig_start_time);
            save_start_time = tic;
            if par.print2file
                % https://www.mathworks.com/help/matlab/creating_plots/compare-ways-to-export-save-graphics-plots-from-figures.html
                % print(curr_fig,'-dpng',['fig2print_' filename '.png'],resolution);
                % saveas(curr_fig, ['fig2print_' filename '.png'])
                % exportgraphics(curr_fig, ['fig2print_' filename '.png'])
                F = getframe(curr_fig);
                imwrite(F.cdata, ['fig2print_' filename '.png'])
                if numclus>3
                    F = getframe(curr_fig2);
                    imwrite(F.cdata, ['fig2print_' filename 'a.png'])
                end
            else
                print(curr_fig)
                if numclus>3
                    print(curr_fig2);
                end
            end
            close(curr_fig)
            close(curr_fig2)
            fig_save_toc = toc(save_start_time);
            fig_toc = toc(fig_start_time);
            % fprintf("Fig %d/%d (%s)(%d)(%d) took %s seconds. Creation(%s) Saving(%s) \n", fnum, numfigs, filename, tot_spikes, Mclasses, num2str(fig_toc, '%2.2f'),num2str(figure_created_toc, '%2.2f'),num2str(fig_save_toc, '%2.2f'));
        catch ME
            print(ME.stack)
        end
    end
    % close(curr_fig)
    % close(curr_fig2)
    disp(' ')
    make_plots_toc = toc(make_plots_start_time);
    fprintf("Clustering plots (%d) took %s seconds. \n", numfigs, num2str(make_plots_toc, '%2.2f'));
end

end

function do_clustering_single(filename,min_spikes4SPC, par_file, par_input,fnum,save_spikes)

    par = struct;
    par = update_parameters(par,par_file,'clus');
    par = update_parameters(par,par_file,'batch_plot');
    par = update_parameters(par,par_input,'clus');
    par = update_parameters(par,par_input,'batch_plot');

    par.filename = filename;
    par.reset_results = true;

    data_handler = readInData(par);
    par = data_handler.par;
    check_WC_params(par)
%     if isfield(par,'channels') && ~isnan(par.channels)
%         par.max_inputs = par.max_inputs * par.channels;
%     end

    par.fname_in = ['tmp_data_wc' num2str(fnum)];                       % temporary filename used as input for SPC
    par.fname = ['data_' data_handler.nick_name];
    par.nick_name = data_handler.nick_name;
    par.fnamespc = ['data_wc' num2str(fnum)];



    if data_handler.with_spikes            			%data have some time of _spikes files
%         [spikes, index] = data_handler.load_spikes();
        [spikes, index,spikes_all,index_all] = data_handler.load_spikes_withCollisions();
    else
        warning('MyComponent:noValidInput', 'File: %s doesn''t include spikes', filename);
        return
    end

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
    cluster_class(:,2)= index';
    cluster_class(:,1)= classes';
    vars = {'cluster_class', 'par','inspk','forced','Temp','gui_status'};
%     if exist('index_all','var')
    if ~isempty(index_all)
        cluster_class_withcollision = zeros(numel(index_all),2);
        cluster_class_withcollision(:,2)=index_all';
        no_coll = ismember(cluster_class_withcollision(:,2),index);
        cluster_class_withcollision(no_coll,1)=classes';
        max_class = max(classes)+1;
        cluster_class_withcollision(~no_coll,1)=max_class;
         vars = {'cluster_class', 'cluster_class_withcollision', 'par','inspk','forced','Temp','gui_status'};
    end
    %%
    
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
      save(['times_' data_handler.nick_name],vars{:});
    catch
      save(['times_' data_handler.nick_name],vars{:},'-v7.3');
    end


end

function counter = count_new_times(initial_date, filenames)
counter = 0;
for i = 1:length(filenames)
    fname = filenames{i};
    FileInfo = dir(['times_' fname(1:end-11) '.mat']);
    if length(FileInfo)==1 && (FileInfo.datenum > initial_date)
        counter = counter + 1;
    end
end
end

function plot_isi(index, class, line_freq)
    ISI_max = 100;
    xa=diff(index(class));
    [n,c]=hist(xa,0:1:ISI_max);
    bar(c(1:end-1),n(1:end-1))
    xlim([0 ISI_max])
    
    for i_freq=1:floor(ISI_max / (1000/line_freq))
        interval = i_freq*(1000/line_freq);
        line([interval,interval],ylim(),'LineWidth',0.8,'LineStyle',':','Color','r')
    end
    
    xlabel('ISI (ms)');
    title([num2str(nnz(xa<3)) ' in < 3ms']); 
end