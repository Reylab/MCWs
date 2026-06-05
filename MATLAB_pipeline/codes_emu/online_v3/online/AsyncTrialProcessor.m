classdef AsyncTrialProcessor < handle
    % ASYNCTRIALPROCESSOR - Manages asynchronous neural data processing
    %   This class enables parallel processing of neural data while PTB
    %   continues stimulus presentation, simulating the behavior of having
    %   two separate MATLAB instances (one for screening, one for processing).
    %
    %   Each trial submits per-channel parfeval jobs for true parallelism
    %   (parfor inside a parfeval worker runs serially - this avoids that).
    %
    %   Usage:
    %       processor = AsyncTrialProcessor(channels, ch_filters, par, det_conf, ...
    %           'DO_SORTING', true, 'sorter', sorter_obj);
    %       processor.submit_trial(trial_data, datacounter, irep);
    %       [detections, spikes] = processor.get_trial_results(irep);
    %       processor.wait_all();

    properties
        % Configuration
        channels            % Channel numbers
        ch_filters          % Channel filters for spike detection
        par                 % Detection parameters
        det_conf            % Detection configuration (-1, 0, or 1 per channel)
        chan_label          % Channel labels

        % Sorting
        DO_SORTING = true   % Enable spike sorting
        sorter              % Online sorter object

        % Collision removal
        b_remove_collisions = true
        b_make_coll_plots = false

        % Job management
        pending_jobs        % Map of trial_id -> job_info struct
        completed_results   % Map of trial_id -> results struct

        % Pool management
        processing_pool     % Dedicated parallel pool for processing
        use_dedicated_pool = false  % Use separate pool from main

        % Statistics
        total_submitted = 0
        total_completed = 0
        processing_times    % Track processing times per trial
    end

    methods
        function obj = AsyncTrialProcessor(channels, ch_filters, par, det_conf, varargin)
            % Constructor
            %   processor = AsyncTrialProcessor(channels, ch_filters, par, det_conf, ...)
            %
            % Optional parameters:
            %   'DO_SORTING'        - Enable spike sorting (default: true)
            %   'sorter'            - Online sorter object
            %   'chan_label'        - Channel labels
            %   'b_remove_collisions' - Enable collision removal (default: true)
            %   'use_dedicated_pool'  - Use separate pool (default: false)
            %   'num_workers'       - Number of workers for dedicated pool

            p = inputParser;
            addParameter(p, 'DO_SORTING', true);
            addParameter(p, 'sorter', []);
            addParameter(p, 'chan_label', {});
            addParameter(p, 'b_remove_collisions', true);
            addParameter(p, 'b_make_coll_plots', false);
            addParameter(p, 'use_dedicated_pool', false);
            addParameter(p, 'num_workers', []);
            parse(p, varargin{:});

            obj.channels = channels;
            obj.ch_filters = ch_filters;
            obj.par = par;
            obj.det_conf = det_conf;
            obj.chan_label = p.Results.chan_label;
            obj.DO_SORTING = p.Results.DO_SORTING;
            obj.sorter = p.Results.sorter;
            obj.b_remove_collisions = p.Results.b_remove_collisions;
            obj.b_make_coll_plots = p.Results.b_make_coll_plots;
            obj.use_dedicated_pool = p.Results.use_dedicated_pool;

            % Initialize job tracking
            obj.pending_jobs = containers.Map('KeyType', 'double', 'ValueType', 'any');
            obj.completed_results = containers.Map('KeyType', 'double', 'ValueType', 'any');
            obj.processing_times = [];

            % Setup parallel pool
            if obj.use_dedicated_pool
                obj.setup_dedicated_pool(p.Results.num_workers);
            else
                % Use existing pool or create default
                obj.processing_pool = gcp('nocreate');
                if isempty(obj.processing_pool)
                    obj.processing_pool = parpool('local');
                end
            end

            fprintf('AsyncTrialProcessor initialized with %d channels, %d workers\n', ...
                numel(channels), obj.processing_pool.NumWorkers);
        end

        function setup_dedicated_pool(obj, num_workers)
            % Setup a dedicated parallel pool for processing
            % This helps isolate processing from any other parallel tasks

            if isempty(num_workers)
                num_cores = feature('numcores');
                num_workers = max(2, floor(num_cores * 0.7));
            end

            % Check if we can create a pool
            cluster = parcluster('local');
            max_workers = cluster.NumWorkers;
            num_workers = min(num_workers, max_workers);

            % Try to get existing pool or create new one
            obj.processing_pool = gcp('nocreate');
            if isempty(obj.processing_pool) || obj.processing_pool.NumWorkers < num_workers
                if ~isempty(obj.processing_pool)
                    delete(obj.processing_pool);
                end
                obj.processing_pool = parpool('local', num_workers);
            end
        end

        function submit_trial(obj, trial_data, datacounter, trial_id, varargin)
            % Submit a trial for background processing
            %   processor.submit_trial(trial_data, datacounter, trial_id)
            %
            % Submits per-channel parfeval jobs for true parallel execution.
            % (parfor inside a parfeval worker runs serially - this avoids that)
            %
            % Inputs:
            %   trial_data   - Cell array of channel data
            %   datacounter  - Vector of data counts per channel
            %   trial_id     - Unique trial identifier (e.g., irep or n_scr*100+irep)
            %
            % Optional:
            %   'priority'   - Job priority (not yet implemented)

            p = inputParser;
            addParameter(p, 'priority', 'normal');
            parse(p, varargin{:});

            n_channels = numel(obj.channels);

            % Create trimmed copies and submit per-channel jobs
            trimmed_data = cell(n_channels, 1);
            channel_futures(1:n_channels) = parallel.FevalFuture;
            for i = 1:n_channels
                trimmed_data{i} = trial_data{i}(1:datacounter(i));
                if obj.DO_SORTING
                    channel_futures(i) = parfeval(obj.processing_pool, ...
                        @get_spikes_online, 2, ...
                        trimmed_data{i}, obj.ch_filters{i}.det, obj.par, obj.det_conf(i));
                else
                    channel_futures(i) = parfeval(obj.processing_pool, ...
                        @detect_mu_online, 1, ...
                        trimmed_data{i}, obj.ch_filters{i}.det, obj.par, obj.det_conf(i));
                end
            end

            % Store job info (trimmed_data needed for collision removal)
            job_info = struct();
            job_info.channel_futures = channel_futures;
            job_info.trial_id = trial_id;
            job_info.trimmed_data = {trimmed_data};  % wrap in cell for Map storage
            job_info.submit_time = tic;

            obj.pending_jobs(trial_id) = job_info;
            obj.total_submitted = obj.total_submitted + 1;

            fprintf('Trial %d submitted for async processing (%d channels)\n', trial_id, n_channels);
        end

        function [detections, spikes, success] = get_trial_results(obj, trial_id, varargin)
            % Get results for a specific trial (waits if not complete)
            %   [detections, spikes, success] = processor.get_trial_results(trial_id)
            %
            % Optional:
            %   'timeout' - Max seconds to wait (default: 60)
            %   'nowait'  - Return immediately if not ready (default: false)

            p = inputParser;
            addParameter(p, 'timeout', 60);
            addParameter(p, 'nowait', false);
            parse(p, varargin{:});

            detections = {};
            spikes = {};
            success = false;

            % Check if already completed
            if obj.completed_results.isKey(trial_id)
                result = obj.completed_results(trial_id);
                detections = result.detections;
                spikes = result.spikes;
                success = true;
                return;
            end

            % Check if job exists
            if ~obj.pending_jobs.isKey(trial_id)
                warning('Trial %d not found in pending or completed jobs', trial_id);
                return;
            end

            job_info = obj.pending_jobs(trial_id);
            futures = job_info.channel_futures;

            % Wait for all channel futures to complete
            if p.Results.nowait
                all_done = all(arrayfun(@(f) strcmp(f.State, 'finished'), futures));
                if all_done
                    [detections, spikes] = obj.fetch_and_store(trial_id, job_info);
                    success = true;
                end
            else
                try
                    t_wait = tic;
                    for i = 1:numel(futures)
                        remaining = p.Results.timeout - toc(t_wait);
                        if remaining <= 0
                            error('Timeout waiting for channel %d', i);
                        end
                        wait(futures(i), 'finished', remaining);
                    end
                    [detections, spikes] = obj.fetch_and_store(trial_id, job_info);
                    success = true;
                catch ME
                    warning('Timeout or error waiting for trial %d: %s', trial_id, ME.message);
                end
            end
        end

        function [ready_ids, pending_ids] = check_status(obj)
            % Check which trials are ready and which are still pending
            %   [ready_ids, pending_ids] = processor.check_status()

            ready_ids = [];
            pending_ids = [];

            keys = obj.pending_jobs.keys;
            for i = 1:numel(keys)
                trial_id = keys{i};
                job_info = obj.pending_jobs(trial_id);
                all_done = all(arrayfun(@(f) strcmp(f.State, 'finished'), job_info.channel_futures));
                if all_done
                    ready_ids(end+1) = trial_id;
                else
                    pending_ids(end+1) = trial_id;
                end
            end

            % Also include already-fetched completed results
            completed_keys = obj.completed_results.keys;
            for i = 1:numel(completed_keys)
                ready_ids(end+1) = completed_keys{i};
            end
            ready_ids = unique(ready_ids);
        end

        function results = collect_ready_results(obj)
            % Collect all results that are ready without waiting
            %   results = processor.collect_ready_results()
            %
            % Returns struct array with fields: trial_id, detections, spikes

            results = struct('trial_id', {}, 'detections', {}, 'spikes', {});

            keys = obj.pending_jobs.keys;
            for i = 1:numel(keys)
                trial_id = keys{i};
                job_info = obj.pending_jobs(trial_id);
                all_done = all(arrayfun(@(f) strcmp(f.State, 'finished'), job_info.channel_futures));
                if all_done
                    [det, spk] = obj.fetch_and_store(trial_id, job_info);
                    results(end+1).trial_id = trial_id;
                    results(end).detections = det;
                    results(end).spikes = spk;
                end
            end
        end

        function wait_all(obj, timeout)
            % Wait for all pending jobs to complete
            %   processor.wait_all(timeout)

            if nargin < 2
                timeout = 300;  % 5 minutes default
            end

            keys = obj.pending_jobs.keys;
            if isempty(keys)
                return;
            end

            fprintf('Waiting for %d pending processing jobs...\n', numel(keys));
            start_wait = tic;

            for i = 1:numel(keys)
                trial_id = keys{i};
                remaining_time = timeout - toc(start_wait);
                if remaining_time <= 0
                    warning('Timeout waiting for all jobs');
                    break;
                end
                obj.get_trial_results(trial_id, 'timeout', remaining_time);
            end

        end

        function add_spikes_to_sorter(obj, trial_id)
            % Add spikes from a completed trial to the sorter
            if isempty(obj.sorter)
                return;
            end

            if obj.completed_results.isKey(trial_id)
                result = obj.completed_results(trial_id);
                if ~isempty(result.spikes)
                    obj.sorter.add_spikes(result.spikes);
                end
            end
        end

        function stats = get_statistics(obj)
            % Get processing statistics
            stats = struct();
            stats.total_submitted = obj.total_submitted;
            stats.total_completed = obj.total_completed;
            stats.pending_count = obj.pending_jobs.Count;
            stats.completed_count = obj.completed_results.Count;
            if ~isempty(obj.processing_times)
                stats.mean_processing_time = mean(obj.processing_times);
                stats.max_processing_time = max(obj.processing_times);
                stats.min_processing_time = min(obj.processing_times);
            else
                stats.mean_processing_time = 0;
                stats.max_processing_time = 0;
                stats.min_processing_time = 0;
            end
            stats.pool_workers = obj.processing_pool.NumWorkers;
        end

        function cleanup(obj)
            % Cleanup resources
            % Cancel any pending channel jobs
            keys = obj.pending_jobs.keys;
            for i = 1:numel(keys)
                job_info = obj.pending_jobs(keys{i});
                for j = 1:numel(job_info.channel_futures)
                    if strcmp(job_info.channel_futures(j).State, 'running')
                        cancel(job_info.channel_futures(j));
                    end
                end
            end

            % Don't delete the pool as it might be shared
            fprintf('AsyncTrialProcessor cleanup complete\n');
        end
    end

    methods (Access = private)
        function [detections, spikes] = fetch_and_store(obj, trial_id, job_info)
            % Fetch per-channel results, run collision removal, store
            t_collect = tic;
            futures = job_info.channel_futures;
            n_channels = numel(futures);
            detections = cell(1, n_channels);
            spikes = cell(n_channels, 1);

            % Collect results from all channel futures
            for i = 1:n_channels
                if obj.DO_SORTING
                    [spikes{i}, detections{i}] = fetchOutputs(futures(i));
                    if numel(spikes{i}) < 1
                        fprintf('No spikes in channel %d. See if micros are connected. \n', i)
                    end
                else
                    detections{i} = fetchOutputs(futures(i));
                end
            end

            % Collision removal (needs all channels together)
            if obj.DO_SORTING && obj.b_remove_collisions && ~isempty(obj.chan_label)
                trimmed_data = job_info.trimmed_data{1};  % unwrap from cell
                try
                    [spikes, detections] = remove_collisions(trimmed_data, detections, ...
                        obj.chan_label, spikes, obj.b_make_coll_plots);
                catch ME
                    warning('%s', sprintf('Collision removal failed: %s', ME.message));
                end
            end

            % Processing time = wall time from submit to collection complete
            % (channels ran in parallel, so this reflects actual parallel time)
            proc_time = toc(job_info.submit_time);
            obj.processing_times(end+1) = proc_time;

            % Store result
            result = struct();
            result.detections = detections;
            result.spikes = spikes;
            result.processing_time = proc_time;
            obj.completed_results(trial_id) = result;

            % Remove from pending (free trimmed_data memory)
            obj.pending_jobs.remove(trial_id);
            obj.total_completed = obj.total_completed + 1;

            fprintf('Trial %d complete.  ', ...
                trial_id);
        end
    end
end
