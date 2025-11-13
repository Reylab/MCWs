classdef online_sorter < handle
    %online_sorter: class to concatenate spikies and sort when enought data
    %is available
    
    properties
        spikes2sort
        spikes
        nchannels
        %sorted
        classes
        %index %for now the sorter doesnt use the index information
        %sortres
        maxworkers
        channels
        maxchperworker
        cls_maxdist
        cls_centers
        nbrach
        minclus
        npca
        max_std
        autosort
        futures
        futures_chs
        sortings_state
        channels_order
    end
    
    methods
        function obj = online_sorter(channels_ID, priorities, spikes2sort, autosort, maxworkers)
            %online_sorter Construct an instance of this class
            priorities_ix = arrayfun(@(x) find(channels_ID==x),priorities);
            
            nchannels = numel(channels_ID);
            channels = 1:nchannels;
            obj.channels_order = [priorities_ix setdiff(channels,priorities_ix,'stable')];
            obj.channels = channels_ID;
            obj.nchannels = nchannels;
            obj.spikes2sort = spikes2sort;
            %obj.sorted = false(nchannels,1);
            obj.classes = cell(nchannels,1);
            obj.spikes = cell(nchannels,1);
            %obj.index = cell(nchannels,1);            
            obj.cls_maxdist =  cell(nchannels,1);
            obj.cls_centers =  cell(nchannels,1);
            obj.nbrach = 2;
            obj.minclus = 20;
            obj.npca = 20;
            obj.max_std = 3;
            if ~exist('maxworkers','var'), maxworkers=6; end %remove latter, for compatibility with older codes
            obj.maxworkers = maxworkers;
            %obj.maxworkers = 6;
            obj.maxchperworker = 5;
            obj.autosort = autosort;
            obj.futures = parallel.FevalFuture;
            obj.futures(1:obj.maxworkers) = parallel.FevalFuture;
            obj.futures_chs = cell(obj.maxworkers,1);
            obj.sortings_state = 0;
        end
        
        function add_spikes(obj,new_spikes)
            %add spikes to each channel,in autosort mode if enough spikes
            %are added, the sort process will be launch in parallel
            if obj.autosort
                %fetchOutputs for finished futures
                obj.retrieve_sorting_results();

                if obj.sortings_state == 1 && all(cellfun(@(x) ~isempty(x),obj.cls_centers))
                    disp('Last sorting Done.')
                    obj.sortings_state = 2;
                end
            end
            for i = 1:obj.nchannels
                obj.spikes{i} = [obj.spikes{i}; new_spikes{i}];
            end
            if obj.autosort
            if obj.sortings_state < 2
                obj.run_auto_sortings(false,obj.spikes2sort)
            end
            for i = 1:obj.nchannels %start forcing spikes on the finished sortings
                if ~isempty(obj.cls_centers{i}) %with sorting
                    obj.force_new_spikes(i);
                end
            end

            end
            
        end
        
        function run_auto_sortings(obj,do_all,min_spikes)
            availableworkers = find(arrayfun(@(x) x.ID==-1,obj.futures));
            naw = numel(availableworkers);
            if naw>0
                sortintchs = vertcat(obj.futures_chs{:}); %been sorted
                sortintchs  = [sortintchs; find(cellfun(@(x) ~isempty(x),obj.cls_centers))]; %adds sorted
                short_sp = find(cellfun(@(x) size(x,1) < min_spikes,obj.spikes)); %not enough spikes
                remaining_chs = setdiff(obj.channels_order,[sortintchs; short_sp],'stable');                
                ch2sort = numel(remaining_chs);
                if ch2sort>0
                    if ~do_all
                        ch2sort = min(naw*obj.maxchperworker,ch2sort);
                    end
                    chdist = ones(naw,1) * floor(ch2sort/naw);
                    extra_ch = rem(ch2sort,naw);
                    chdist(1:extra_ch) = chdist(1:extra_ch) + 1 ;
                    for i=1:naw
                        if chdist(i)==0
                            break
                        end
                        chs = remaining_chs(1:chdist(i));
                        remaining_chs  = setdiff(remaining_chs,chs);
                        obj.futures(availableworkers(i)) = parfeval(@sort_spikes_cell,3,...
                            cellfun(@(x) x(1:min(size(x,1),obj.spikes2sort),:),obj.spikes(chs),'UniformOutput',false), obj.nbrach,...
                            obj.npca,obj.minclus,obj.max_std);                            
                        obj.futures_chs{availableworkers(i)} = chs;
                        if obj.sortings_state == 0
                            obj.sortings_state = 1;
                            disp('Starting first sorting.')
                        end
                    end
                end
            end
        end
        function force = force_new_spikes(obj,i)
            %this disable force for invalid sortings for low amount of
            %spikes (fix in a better way)
            force = ~isempty(obj.cls_centers{i}) && ~isnan(obj.cls_centers{i}(1)) && numel(obj.classes{i}) < size(obj.spikes{i},1);
            if force
                new_labels = force_spikes_from_model(obj.cls_centers{i}, obj.cls_maxdist{i}, obj.spikes{i}(numel(obj.classes{i})+1:end,:));
                obj.classes{i} = [obj.classes{i} new_labels];
            end
        end
        function do_remaining_sorting(obj)
            obj.run_auto_sortings(true,0)
        end
        
        function retrieve_sorting_results(obj)
            for i = 1:obj.maxworkers
                if strcmp(obj.futures(i).State,'finished')
                    if isempty(obj.futures(i).Error)
                        [classes_cell,cls_centers_cell, cls_maxdist_cell] = fetchOutputs(obj.futures(i));
                        for j = 1:numel(obj.futures_chs{i})
                            ci = obj.futures_chs{i}(j);
                            obj.classes{ci} = classes_cell{j};
                            obj.cls_centers{ci}= cls_centers_cell{j};
                            obj.cls_maxdist{ci}= cls_maxdist_cell{j};
                        end
                        
                        %dur = obj.futures{i}.FinishDateTime - obj.futures{i}.StartDateTime;
                        %fprintf('Channel sorting finished in: %.2f seconds.\n',seconds(dur));
                    else
                        for j = 1:numel(obj.futures_chs{i})
                            ci = obj.futures_chs{i}(j);
                            warning('ERROR: in sorting :%s. All classes set to 1.',obj.futures(i).Error.message)
                            n = min(size(obj.spikes{ci},1),obj.spikes2sort);
                            obj.classes{ci} = ones(1,n);
                            [centers, sd, ~] = build_templates(obj.classes{ci}, obj.spikes{ci}(1:n,:));
                            obj.cls_centers{ci} = centers;
                            obj.cls_maxdist{ci} = (obj.max_std*sd).^2;
                        end
                    end

                    obj.futures(i) = parallel.FevalFuture;
                    obj.futures_chs{i} = [];
                end
            end
            
            if obj.sortings_state == 1 && all(cellfun(@(x) ~isempty(x),obj.cls_centers))
                disp('Last sorting Done.')
                obj.sortings_state = 2;
            end
            
        end
        function save_sorting_results(obj, filename)
            for i=1:obj.nchannels
                if isempty(obj.cls_centers{i})
                    warning('Channel %d unsorted.',obj.channels(i))
                end
            end
            cls_centers = obj.cls_centers;
            cls_maxdist = obj.cls_maxdist;
            channels = obj.channels;
            max_std = obj.max_std;
            save(filename, 'cls_centers', 'cls_maxdist', 'channels', 'max_std');
        end
        
        function load_sorting_results(obj, input)
            if isempty(input)
                return
            end

            data = load(input);
            for i=1:obj.nchannels
                ch_ix = find(data.channels == obj.channels(i));
                if isempty(ch_ix)
                    warning('File %s doesn''t have channel %d templates.',input, obj.channels(i))
                else
                    obj.cls_centers{i} = data.cls_centers{ch_ix};
                    obj.cls_maxdist{i} = data.cls_maxdist{ch_ix};
                    if obj.max_std ~= data.max_std
                        obj.cls_maxdist{i} = (sqrt(obj.cls_maxdist{i})./obj.max_std.*data.max_std).^2;
                    end                    

                end

            end
            disp('Spike Sorting loaded.')
        end
        
        
        function [classes_out, ch_ix_out] = get_done_sorts(obj,omit_chs_ix)
            %first try to force the ones with a retrieved sorting
            classes_out = {};
            ch_ix_out = [];
            obj.retrieve_sorting_results();
            
            for i = obj.channels_order
                if any(omit_chs_ix==i)
                    continue
                end
                force_new_spikes(obj,i);
                if ~isempty(obj.cls_centers{i})
                    ch_ix_out(end+1) = i;
                    classes_out{end+1} = obj.classes{i};
                end
            end
        end
        function classes = get_sort(obj)
            %start remaining channels
            obj.do_remaining_sorting()
            if obj.autosort
                for i = 1:obj.nchannels
                    force_new_spikes(obj,i);%will try to force the ones with a retrieved sorting
                end
            else
                wait(obj.futures(arrayfun(@(x) x.ID~=-1,obj.futures)));
            end
            %non empty loops are still running
            waiting = arrayfun(@(x) x.ID~=-1,obj.futures);

            while nnz(waiting)>0
                obj.retrieve_sorting_results();
                waiting = arrayfun(@(x) x.ID~=-1,obj.futures);
                if nnz(waiting)==1
                    wait(obj.futures(waiting));
                end
            end
            classes = obj.classes;
        end
    end
end

function [ch_classes,centers, maxdist] = sort_spikes(spikes, nbrach,npca,minclus,max_std)
        ch_classes = msort(spikes, nbrach,npca,minclus);
        [centers, sd, ~] = build_templates(ch_classes, spikes); % we are going to ignore pd
        maxdist = (max_std*sd).^2;
end

function [ch_classes_cell,centers_cell, maxdist_cell] = sort_spikes_cell(spikes_cell,nbrach,npca,minclus,max_std)
    n = numel(spikes_cell);
    ch_classes_cell = cell(n,1);
    centers_cell = cell(n,1);
    maxdist_cell = cell(n,1);

    for i=1:n
        %this disable force for invalid sortings for low amount of
            %spikes (fix in a better way)
        if size(spikes_cell{i},1) < 20
            centers_cell{i} = nan;
            maxdist_cell{i} = inf;
            ch_classes_cell{i} = [];
%         elseif size(spikes_cell{i},1) < 20 %minimum amoun of spiks to call msort
%             ch_classes_cell{i} = ones(1, size(spikes_cell{i},1));
%             [centers_cell{i}, ~, ~] = build_templates(ch_classes_cell{i}, spikes_cell{i}); % we are going to ignore pd
%             maxdist_cell{i} = inf;
        else
            ch_classes_cell{i} = msort(spikes_cell{i}, nbrach,npca,minclus);
            [centers_cell{i}, sd, ~] = build_templates(ch_classes_cell{i}, spikes_cell{i}); % we are going to ignore pd
            maxdist_cell{i} = (max_std*sd).^2;
        end
    end
end

function classes=force_spikes_from_model(centers, maxdist, spikes)
    classes = zeros(1,size(spikes,1));
    aux_ones = ones(size(centers,1),1);
    for i=1:size(spikes,1)
        distances = sum((aux_ones*spikes(i,:) - centers).^2,2)';
        conforming = find(distances < maxdist);
        if( ~isempty(conforming))
            [~, ii] = min(distances(conforming));
            classes(i) = conforming(ii);
        end
    end
end