classdef handle_plot_continuous<handle
    
    properties
        current_n
        samples2show
        par
        stage %0: nothing, 1:loading, 2:ready to plot, 3:plotting,
              %4:ended 
        buffer
        pool
        u_ch_ix
        u_chID
        labels
        conversion
        notches_folder
        remove_artifacts
        selected_path
        pfv
    end
    
    methods
        function obj = handle_plot_continuous(seconds2show, par,...
                u_ch_ix,u_chID, labels, conversion,notches_folder,remove_artifacts)
            if par.use_parallel == true
                pool = gcp('nocreate');
                if isempty(pool) % If already a pool, do not create new one.
                    pool = parpool();
                end
                obj.pool = pool;
                obj.u_chID = u_chID;
                obj.labels = labels;
                obj.conversion = conversion;
                obj.notches_folder = notches_folder;
                obj.remove_artifacts = remove_artifacts;
                obj.samples2show = ceil(seconds2show*30000);
                obj.current_n = zeros(numel(u_ch_ix),1);
                obj.u_ch_ix = u_ch_ix;
                obj.buffer = [];
            end
            obj.par = par;
            obj.stage = 0;
        end
             
        
        function start(obj,path)
            if (obj.par.use_parallel == false)
                return;
            end
            if isempty(obj.buffer)
               obj.buffer = zeros(length(obj.u_ch_ix),obj.samples2show,'int16');
            end
            obj.stage = 1;
            obj.selected_path = path;
        end
        function isl = isloading(obj)
            isl = obj.stage == 1;
        end
        
        function update(obj,stream)
            % this function suposses that the buffer of each channel haves
            %  the same number of samples
            if obj.stage ~= 1
                return
            end
            for ci = 1:length(obj.u_ch_ix)                
                n = numel(stream.data{obj.u_ch_ix(ci)});
                to_update = min(n, obj.samples2show-obj.current_n(ci));
                if to_update>0
                    obj.buffer(ci,obj.current_n(ci)+(1:to_update)) = stream.data{obj.u_ch_ix(ci)}(1:to_update);
                    obj.current_n(ci) = obj.current_n(ci) + to_update;
                end
            end
            
            
            if all(obj.current_n==obj.samples2show)
                obj.stage = 2;
            end
        end
        
        function ready = isready2plot(obj)
            ready = (obj.stage == 2);
        end

%         function start_plotting(obj,notches_folder)
        function start_plotting(obj)
            if obj.stage~=2
                return
            end
            obj.stage = 3;
            %UPDATE NOTCHES FOLDER!!!
%             obj.notches_folder = notches_folder;
%             plot_continuous_data(obj.buffer, obj.par, obj.u_chID, obj.labels, obj.conversion, obj.selected_path,obj.notches_folder, obj.remove_artifacts)
            obj.pfv=parfeval(@plot_continuous_data,0, obj.buffer, obj.par,...
                obj.u_chID, obj.labels,obj.conversion,obj.selected_path,obj.notches_folder, obj.remove_artifacts);
        end
        
        
        function done=isdone(obj)
            if obj.stage~=3 %if not waiting to confirm
                done = false;
                return
            end 
            done = strcmp(obj.pfv.State,'finished');
            %done = isempty([obj.pool.FevalQueue.RunningFutures obj.pool.FevalQueue.QueuedFutures]);
            obj.current_n(:) = 0;
            obj.buffer = [];
            if done
                obj.stage=4;
            end
        end
        function  reset(obj)
            obj.current_n = obj.current_n*0;
            if obj.stage==3
                cancel(obj.pfv)
            end
            obj.stage = 0;            
        end
        function  ge = got_error(obj)
            ge = ~isempty(obj.pfv.Error);           
        end
        function  error_str = get_error_str(obj)
            errors = obj.pfv.Error.remotecause{1}.stack;
            for i = 1:numel(errors)
                st = errors(i);
                fprintf(2,'%s\n%s l:%d\n\n', st.name,st.file,st.line);
            end
            error_str = ['ERROR:' obj.pfv.Error.message];           
        end
    end
end

