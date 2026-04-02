classdef BackgroundDataCollectorWorker < handle
    % BACKGROUNDDATACOLLECTORWORKER - Data collection in dedicated parfeval worker
    %   Runs device_com in a separate worker process, completely independent
    %   from the main thread. Main thread handles PTB only.
    %
    %   This is similar to the two-MATLAB architecture but within a single
    %   MATLAB process using parfeval workers.
    %
    %   Usage:
    %       collector = BackgroundDataCollectorWorker(params);
    %       collector.wait_ready();  % Wait for worker to initialize
    %       info = collector.get_channel_info();  % Get channel info from worker
    %       
    %       collector.start_trial();
    %       % ... PTB presentation ...
    %       [data, counts, events] = collector.get_trial_data();
    %       
    %       collector.shutdown();
    
    properties (SetAccess = private)
        % Configuration
        params              % Task parameters
        channels            % Channel numbers to collect (set by worker)
        ev_channels         % Event channels (photodiode etc)
        num_channels        % Number of channels
        poll_interval       % Collection interval in worker (seconds)
        
        % Worker management
        worker_future       % parfeval Future object
        cmd_queue           % PollableDataQueue for commands (main -> worker)
        data_queue          % DataQueue for data (worker -> main)
        
        % State
        is_ready            % Worker initialized and ready
        is_collecting       % Currently collecting data
        last_error          % Last error from worker
        
        % Channel info (received from worker)
        channel_info        % Full channel info struct from device
        
        % Received data (from worker)
        received_data       % Latest trial data
        received_counts     % Latest data counts
        received_events     % Latest events
        data_received       % Flag indicating data was received
    end
    
    properties (Constant)
        CMD_START = 'start'
        CMD_STOP = 'stop'
        CMD_GET_DATA = 'get_data'
        CMD_SHUTDOWN = 'shutdown'
        CMD_STATUS = 'status'
        
        RESP_READY = 'ready'
        RESP_DATA = 'data'
        RESP_ERROR = 'error'
        RESP_SHUTDOWN = 'shutdown_complete'
        RESP_CHANNEL_INFO = 'channel_info'
    end
    
    methods
        function obj = BackgroundDataCollectorWorker(params, varargin)
            % Constructor - starts the worker
            %
            % Inputs:
            %   params - Task parameters (must include system, use_photodiodo, etc.)
            %   
            % Optional:
            %   'poll_interval' - Collection interval (default: 0.010 = 10ms)
            %   'remove_channels' - Channel numbers to exclude
            %   'remove_channels_by_label' - Cell array of label patterns to exclude
            
            p = inputParser;
            addRequired(p, 'params');
            addParameter(p, 'poll_interval', 0.010);  % 10ms default
            addParameter(p, 'remove_channels', []);
            addParameter(p, 'remove_channels_by_label', {});
            parse(p, params, varargin{:});
            
            obj.params = p.Results.params;
            obj.poll_interval = p.Results.poll_interval;
            
            % Store channel exclusion settings in params for worker
            obj.params.remove_channels = p.Results.remove_channels;
            obj.params.remove_channels_by_label = p.Results.remove_channels_by_label;
            
            obj.is_ready = false;
            obj.is_collecting = false;
            obj.data_received = false;
            obj.last_error = '';
            obj.channel_info = [];
            obj.channels = [];
            obj.num_channels = 0;
            
            % Create communication queues
            % PollableDataQueue for commands (worker polls for commands)
            % DataQueue for data (main thread receives via callback)
            obj.cmd_queue = parallel.pool.PollableDataQueue;
            obj.data_queue = parallel.pool.DataQueue;
            
            % Set up listener for data from worker
            afterEach(obj.data_queue, @(msg) obj.handle_worker_message(msg));
            
            % Start the worker
            obj.start_worker();
        end
        
        function wait_ready(obj, timeout)
            % Wait for worker to be ready
            if nargin < 2
                timeout = 30;  % 30 second default timeout
            end
            
            start_time = tic;
            while ~obj.is_ready && toc(start_time) < timeout
                pause(0.1);
                
                % Check for worker errors
                if ~isempty(obj.worker_future) && strcmp(obj.worker_future.State, 'finished')
                    if ~isempty(obj.worker_future.Error)
                        error('Worker failed to start: %s', obj.worker_future.Error.message);
                    end
                end
            end
            
            if ~obj.is_ready
                error('Worker did not become ready within %d seconds', timeout);
            end
            
            fprintf('BackgroundDataCollectorWorker ready with %d channels.\n', obj.num_channels);
        end
        
        function info = get_channel_info(obj)
            % Get channel information from worker
            % Returns struct with: channels, chan_label, conversion, ev_channels
            if ~obj.is_ready
                error('Worker not ready. Call wait_ready() first.');
            end
            
            info = obj.channel_info;
        end
        
        function start_trial(obj)
            % Start collecting data for a new trial
            if ~obj.is_ready
                error('Worker not ready. Call wait_ready() first.');
            end
            
            obj.data_received = false;
            obj.received_data = {};
            obj.received_counts = [];
            obj.received_events = struct();
            
            send(obj.cmd_queue, struct('cmd', obj.CMD_START));
            obj.is_collecting = true;
        end
        
        function [data, counts, events] = get_trial_data(obj, timeout)
            % Stop collection and get trial data
            if nargin < 2
                timeout = 30;  % 30 second timeout
            end
            
            % Send stop command (which also triggers data return)
            send(obj.cmd_queue, struct('cmd', obj.CMD_STOP));
            obj.is_collecting = false;
            
            % Wait for data
            start_time = tic;
            while ~obj.data_received && toc(start_time) < timeout
                pause(0.01);
            end
            
            if ~obj.data_received
                warning('Timeout waiting for trial data');
                data = {};
                counts = [];
                events = struct('time', [], 'value', [], 'pdiode', [], ...
                    'timestamps', {{}}, 'data_loss', true);
                return;
            end
            
            data = obj.received_data;
            counts = obj.received_counts;
            events = obj.received_events;
        end
        
        function stats = get_statistics(obj)
            % Request statistics from worker
            send(obj.cmd_queue, struct('cmd', obj.CMD_STATUS));
            
            % For now, return basic stats
            stats.is_ready = obj.is_ready;
            stats.is_collecting = obj.is_collecting;
            stats.last_error = obj.last_error;
        end
        
        function stop(obj)
            % Stop collection without getting data
            if obj.is_collecting
                send(obj.cmd_queue, struct('cmd', obj.CMD_STOP));
                obj.is_collecting = false;
            end
        end
        
        function shutdown(obj)
            % Shutdown the worker completely
            if ~isempty(obj.cmd_queue)
                try
                    send(obj.cmd_queue, struct('cmd', obj.CMD_SHUTDOWN));
                catch
                end
            end
            
            % Wait for worker to finish
            if ~isempty(obj.worker_future) && isvalid(obj.worker_future)
                try
                    wait(obj.worker_future, 10);  % 10 second timeout
                catch
                end
                
                if strcmp(obj.worker_future.State, 'running')
                    cancel(obj.worker_future);
                end
            end
            
            obj.is_ready = false;
            obj.is_collecting = false;
            fprintf('BackgroundDataCollectorWorker shutdown complete.\n');
        end
        
        function delete(obj)
            % Destructor
            obj.shutdown();
        end
    end
    
    methods (Access = private)
        function start_worker(obj)
            % Start the worker process
            
            % Ensure parallel pool exists
            pool = gcp('nocreate');
            if isempty(pool)
                error('BackgroundDataCollectorWorker requires an active parallel pool. Start one with parpool() first.');
            end
            
            % Package parameters for worker
            worker_params = struct();
            worker_params.system = obj.params.system;
            worker_params.use_photodiodo = obj.params.use_photodiodo;
            worker_params.poll_interval = obj.poll_interval;
            
            % Channel exclusion settings
            worker_params.remove_channels = obj.params.remove_channels;
            worker_params.remove_channels_by_label = obj.params.remove_channels_by_label;
            
            % Get device_com parameters from params
            if isfield(obj.params, 'which_nsp_micro')
                worker_params.which_nsp_micro = obj.params.which_nsp_micro;
            else
                worker_params.which_nsp_micro = 1;
            end
            
            if isfield(obj.params, 'nsp_address')
                worker_params.nsp_address = obj.params.nsp_address;
            else
                worker_params.nsp_address = {};
            end
            
            if isfield(obj.params, 'mapfile')
                worker_params.mapfile = obj.params.mapfile;
            else
                worker_params.mapfile = 'ripple.map';
            end
            
            if isfield(obj.params, 'nsp_type')
                worker_params.nsp_type = obj.params.nsp_type;
            else
                worker_params.nsp_type = 256;
            end
            
            if isfield(obj.params, 'photodiode_channel')
                worker_params.photodiode_channel = obj.params.photodiode_channel;
            else
                worker_params.photodiode_channel = [];  % Will use default in worker
            end
            
            % Start worker with parfeval
            obj.worker_future = parfeval(@BackgroundDataCollectorWorker.worker_main, 0, ...
                worker_params, obj.cmd_queue, obj.data_queue);
            
            fprintf('BackgroundDataCollectorWorker starting...\n');
        end
        
        function handle_worker_message(obj, msg)
            % Handle messages from worker
            
            if ~isstruct(msg)
                return;
            end
            
            switch msg.type
                case obj.RESP_CHANNEL_INFO
                    % Channel info received from worker
                    obj.channel_info = msg.info;
                    obj.channels = msg.info.channels;
                    obj.ev_channels = msg.info.ev_channels;
                    obj.num_channels = numel(obj.channels);
                    fprintf('Worker received channel info: %d channels.\n', obj.num_channels);
                    
                case obj.RESP_READY
                    obj.is_ready = true;
                    fprintf('Worker reports ready.\n');
                    
                case obj.RESP_DATA
                    obj.received_data = msg.data;
                    obj.received_counts = msg.counts;
                    obj.received_events = msg.events;
                    obj.data_received = true;
                    
                case obj.RESP_ERROR
                    obj.last_error = msg.message;
                    warning('Worker error: %s', msg.message);
                    
                case obj.RESP_SHUTDOWN
                    obj.is_ready = false;
                    fprintf('Worker shutdown confirmed.\n');
            end
        end
    end
    
    methods (Static)
        function success = test_communication(verbose)
            % TEST_COMMUNICATION - Test worker communication without hardware
            %   Verifies that parfeval and DataQueue communication works.
            %
            %   Usage:
            %       BackgroundDataCollectorWorker.test_communication()
            %
            %   This test does NOT require the actual Ripple/Blackrock hardware.
            %   Run this first to verify MATLAB parallel computing is working.
            
            if nargin < 1
                verbose = true;
            end
            
            success = false;
            
            if verbose
                fprintf('\n');
                fprintf('===========================================================\n');
                fprintf('  Worker Communication Test (no hardware required)\n');
                fprintf('===========================================================\n\n');
            end
            
            % Check for parallel pool
            if verbose
                fprintf('[1/3] Checking parallel pool...\n');
            end
            pool = gcp('nocreate');
            if isempty(pool)
                if verbose
                    fprintf('      No pool found. Creating one...\n');
                end
                try
                    pool = parpool('local', 2);
                    if verbose
                        fprintf('      Created pool with %d workers\n', pool.NumWorkers);
                    end
                catch ME
                    if verbose
                        fprintf('      FAILED: %s\n', ME.message);
                    end
                    return;
                end
            else
                if verbose
                    fprintf('      Pool exists with %d workers\n', pool.NumWorkers);
                end
            end
            
            % Test DataQueue communication
            if verbose
                fprintf('\n[2/3] Testing DataQueue communication...\n');
            end
            try
                data_queue = parallel.pool.DataQueue;
                received_msg = [];
                afterEach(data_queue, @(msg) assignin('caller', 'received_msg', msg));
                
                % Send message from worker
                f = parfeval(@(dq) send(dq, struct('test', 'hello', 'value', 42)), 0, data_queue);
                wait(f, 5);
                pause(0.5);  % Wait for callback
                
                if ~isempty(received_msg) && strcmp(received_msg.test, 'hello')
                    if verbose
                        fprintf('      DataQueue: message received correctly\n');
                    end
                else
                    if verbose
                        fprintf('      FAILED: message not received\n');
                    end
                    return;
                end
            catch ME
                if verbose
                    fprintf('      FAILED: %s\n', ME.message);
                end
                return;
            end
            
            % Test PollableDataQueue
            if verbose
                fprintf('\n[3/3] Testing PollableDataQueue...\n');
            end
            try
                cmd_queue = parallel.pool.PollableDataQueue;
                
                % Send command to worker and get response
                response_queue = parallel.pool.DataQueue;
                received_response = [];
                afterEach(response_queue, @(msg) assignin('caller', 'received_response', msg));
                
                f = parfeval(@test_pollable_worker, 0, cmd_queue, response_queue);
                
                % Send a command
                send(cmd_queue, struct('cmd', 'ping'));
                
                % Wait for response
                pause(1);
                
                if ~isempty(received_response) && strcmp(received_response, 'pong')
                    if verbose
                        fprintf('      PollableDataQueue: command/response working\n');
                    end
                else
                    if verbose
                        fprintf('      WARNING: response not received (may be timing issue)\n');
                    end
                end
                
                % Send shutdown
                send(cmd_queue, struct('cmd', 'shutdown'));
                wait(f, 5);
                
            catch ME
                if verbose
                    fprintf('      FAILED: %s\n', ME.message);
                end
                return;
            end
            
            success = true;
            
            if verbose
                fprintf('\n===========================================================\n');
                fprintf('  COMMUNICATION TEST PASSED\n');
                fprintf('===========================================================\n\n');
            end
            
            function test_pollable_worker(cmd_q, resp_q)
                running = true;
                while running
                    [cmd, has_cmd] = poll(cmd_q, 0.1);
                    if has_cmd
                        if strcmp(cmd.cmd, 'ping')
                            send(resp_q, 'pong');
                        elseif strcmp(cmd.cmd, 'shutdown')
                            running = false;
                        end
                    end
                end
            end
        end
        
        function [success, results] = test(system, varargin)
            % TEST - Test device connectivity and data collection
            %   Run this before the main experiment to verify device works.
            %
            %   Usage:
            %       BackgroundDataCollectorWorker.test('RIP')
            %       BackgroundDataCollectorWorker.test('RIP', 'mapfile', 'path/to/map')
            %       BackgroundDataCollectorWorker.test('BRK', 'address', '192.168.137.3')
            %       [success, results] = BackgroundDataCollectorWorker.test('RIP', 'duration', 5)
            %
            %   Inputs:
            %       system - 'RIP' for Ripple or 'BRK' for Blackrock
            %
            %   Optional parameters:
            %       'mapfile' - Path to Ripple map file (required for RIP)
            %       'address' - NSP address (for BRK)
            %       'instance' - NSP instance (default: 0)
            %       'nsp_type' - NSP type (default: 256)
            %       'duration' - Test duration in seconds (default: 3)
            %       'poll_interval' - Collection interval (default: 0.010)
            %       'verbose' - Print detailed output (default: true)
            %
            %   Outputs:
            %       success - true if test passed
            %       results - struct with test details
            
            p = inputParser;
            addRequired(p, 'system', @(x) ismember(x, {'RIP', 'BRK'}));
            addParameter(p, 'mapfile', []);
            addParameter(p, 'address', '192.168.137.3');
            addParameter(p, 'instance', 0);
            addParameter(p, 'nsp_type', 256);
            addParameter(p, 'duration', 3);
            addParameter(p, 'poll_interval', 0.010);
            addParameter(p, 'verbose', true);
            parse(p, system, varargin{:});
            
            opts = p.Results;
            success = false;
            results = struct();
            results.system = system;
            results.errors = {};
            
            if opts.verbose
                fprintf('\n');
                fprintf('===========================================================\n');
                fprintf('  BackgroundDataCollectorWorker Device Test\n');
                fprintf('===========================================================\n');
                fprintf('  System: %s\n', system);
                fprintf('  Test duration: %.1f seconds\n', opts.duration);
                fprintf('  Poll interval: %.3f seconds\n', opts.poll_interval);
                fprintf('-----------------------------------------------------------\n\n');
            end
            
            % Step 1: Test device_com open
            if opts.verbose
                fprintf('[1/5] Opening device connection...\n');
            end
            
            try
                if strcmp(system, 'BRK')
                    device_com('open', system, 'address', opts.address, ...
                        'instance', opts.instance, 'nsp_type', opts.nsp_type);
                elseif strcmp(system, 'RIP')
                    if isempty(opts.mapfile)
                        error('mapfile is required for Ripple system');
                    end
                    device_com('open', system, 'mapfile', opts.mapfile);
                end
                results.device_opened = true;
                if opts.verbose
                    fprintf('      SUCCESS: Device connection opened\n\n');
                end
            catch ME
                results.device_opened = false;
                results.errors{end+1} = sprintf('Device open failed: %s', ME.message);
                if opts.verbose
                    fprintf('      FAILED: %s\n\n', ME.message);
                end
                return;
            end
            
            % Step 2: Get channel info
            if opts.verbose
                fprintf('[2/5] Getting channel information...\n');
            end
            
            try
                info = device_com('get_chs_info');
                results.total_channels = numel(info.ch);
                results.micro_channels = sum(info.ismicro);
                results.channel_labels = info.label(info.ismicro);
                
                if opts.verbose
                    fprintf('      Total channels: %d\n', results.total_channels);
                    fprintf('      Micro channels: %d\n', results.micro_channels);
                    if results.micro_channels > 0 && results.micro_channels <= 10
                        fprintf('      Labels: %s\n', strjoin(results.channel_labels, ', '));
                    elseif results.micro_channels > 10
                        fprintf('      First 10 labels: %s, ...\n', strjoin(results.channel_labels(1:10), ', '));
                    end
                    fprintf('      SUCCESS: Channel info retrieved\n\n');
                end
            catch ME
                results.errors{end+1} = sprintf('Get channel info failed: %s', ME.message);
                if opts.verbose
                    fprintf('      FAILED: %s\n\n', ME.message);
                end
                device_com('close');
                return;
            end
            
            % Step 3: Enable channels
            if opts.verbose
                fprintf('[3/5] Enabling channels...\n');
            end
            
            try
                channels = info.ch(info.ismicro);
                if strcmp(system, 'BRK')
                    ev_channels = 257;  % Default photodiode
                else
                    ev_channels = 1;
                end
                
                device_com('enable_chs', channels, true, ev_channels);
                device_com('clear_buffer');
                pause(0.2);
                
                results.enabled_channels = numel(channels);
                if opts.verbose
                    fprintf('      Enabled %d channels\n', numel(channels));
                    fprintf('      SUCCESS: Channels enabled\n\n');
                end
            catch ME
                results.errors{end+1} = sprintf('Enable channels failed: %s', ME.message);
                if opts.verbose
                    fprintf('      FAILED: %s\n\n', ME.message);
                end
                device_com('close');
                return;
            end
            
            % Step 4: Collect data
            if opts.verbose
                fprintf('[4/5] Collecting data for %.1f seconds...\n', opts.duration);
            end
            
            try
                num_channels = numel(channels);
                max_samples = 30000 * (opts.duration + 2);
                data_buffers = cell(num_channels, 1);
                for i = 1:num_channels
                    data_buffers{i} = zeros(max_samples, 1);
                end
                data_counts = zeros(num_channels, 1);
                
                collection_count = 0;
                data_loss = false;
                start_time = tic;
                
                while toc(start_time) < opts.duration
                    streams = device_com('get_stream');
                    
                    if ~isempty(streams) && isfield(streams, 'data')
                        collection_count = collection_count + 1;
                        
                        for i = 1:min(num_channels, size(streams.data, 1))
                            if streams.lost_prev(i) > 0 || streams.lost_post(i) > 0
                                data_loss = true;
                            end
                            
                            ch_data = streams.data{i};
                            if ~isempty(ch_data)
                                n = length(ch_data);
                                start_idx = data_counts(i) + 1;
                                end_idx = start_idx + n - 1;
                                if end_idx <= max_samples
                                    data_buffers{i}(start_idx:end_idx) = ch_data;
                                    data_counts(i) = end_idx;
                                end
                            end
                        end
                    end
                    
                    pause(opts.poll_interval);
                end
                
                elapsed = toc(start_time);
                results.collection_count = collection_count;
                results.elapsed_time = elapsed;
                results.data_loss = data_loss;
                results.samples_per_channel = data_counts;
                results.mean_samples = mean(data_counts);
                results.expected_samples = elapsed * 30000;
                results.collection_rate = results.mean_samples / results.expected_samples * 100;
                
                if opts.verbose
                    fprintf('      Collections: %d\n', collection_count);
                    fprintf('      Elapsed time: %.2f sec\n', elapsed);
                    fprintf('      Samples collected: %.0f (expected: %.0f)\n', ...
                        results.mean_samples, results.expected_samples);
                    fprintf('      Collection rate: %.1f%%\n', results.collection_rate);
                    fprintf('      Data loss detected: %s\n', string(data_loss));
                    fprintf('      SUCCESS: Data collection working\n\n');
                end
                
            catch ME
                results.errors{end+1} = sprintf('Data collection failed: %s', ME.message);
                if opts.verbose
                    fprintf('      FAILED: %s\n\n', ME.message);
                end
                device_com('close');
                return;
            end
            
            % Step 5: Verify data quality
            if opts.verbose
                fprintf('[5/5] Verifying data quality...\n');
            end
            
            try
                % Check for non-zero data
                has_data = false;
                for i = 1:num_channels
                    if data_counts(i) > 0 && any(data_buffers{i}(1:data_counts(i)) ~= 0)
                        has_data = true;
                        break;
                    end
                end
                
                results.has_nonzero_data = has_data;
                results.collection_rate_ok = results.collection_rate > 90;
                
                if opts.verbose
                    fprintf('      Non-zero data received: %s\n', string(has_data));
                    fprintf('      Collection rate >= 90%%: %s\n', string(results.collection_rate_ok));
                    
                    if has_data && results.collection_rate_ok && ~data_loss
                        fprintf('      SUCCESS: Data quality OK\n\n');
                    else
                        fprintf('      WARNING: Data quality issues detected\n\n');
                    end
                end
                
            catch ME
                results.errors{end+1} = sprintf('Data verification failed: %s', ME.message);
                if opts.verbose
                    fprintf('      FAILED: %s\n\n', ME.message);
                end
            end
            
            % Close device
            try
                device_com('close');
                if opts.verbose
                    fprintf('Device connection closed.\n\n');
                end
            catch
            end
            
            % Final result
            success = results.device_opened && ...
                      results.micro_channels > 0 && ...
                      results.collection_count > 0 && ...
                      ~data_loss && ...
                      results.collection_rate > 90;
            
            results.success = success;
            
            if opts.verbose
                fprintf('===========================================================\n');
                if success
                    fprintf('  TEST PASSED - Device is working correctly!\n');
                else
                    fprintf('  TEST FAILED - Issues detected:\n');
                    if ~results.device_opened
                        fprintf('    - Could not open device connection\n');
                    end
                    if results.micro_channels == 0
                        fprintf('    - No micro channels found\n');
                    end
                    if results.collection_count == 0
                        fprintf('    - No data collected\n');
                    end
                    if data_loss
                        fprintf('    - Data loss detected during collection\n');
                    end
                    if results.collection_rate <= 90
                        fprintf('    - Low collection rate (%.1f%%)\n', results.collection_rate);
                    end
                    for i = 1:length(results.errors)
                        fprintf('    - %s\n', results.errors{i});
                    end
                end
                fprintf('===========================================================\n\n');
            end
        end
        
        function worker_main(params, cmd_queue, data_queue)
            % Main worker function - runs in parfeval worker
            % This function opens device_com and runs the collection loop
            
            fprintf('Worker starting - opening device_com...\n');
            
            % Initialize
            poll_interval = params.poll_interval;
            use_photodiodo = params.use_photodiodo;
            max_buffer_samples = 30000 * 120;  % 120 seconds at 30kHz
            
            channels = [];
            ev_channels = [];
            chan_label = {};
            conversion = [];
            
            try
                % Open device_com in this worker
                if strcmp(params.system, 'BRK')
                    inst_num = params.which_nsp_micro - 1;
                    address = params.nsp_address;
                    if iscell(address) && ~isempty(address)
                        address = address{params.which_nsp_micro};
                    end
                    device_com('open', params.system, 'address', address, ...
                        'instance', inst_num, 'nsp_type', params.nsp_type);
                    % Use photodiode channel from params if available
                    if ~isempty(params.photodiode_channel)
                        ev_channels = params.photodiode_channel;
                    else
                        ev_channels = 257;  % Default for BRK
                    end
                elseif strcmp(params.system, 'RIP')
                    device_com('open', params.system, 'mapfile', params.mapfile);
                    ev_channels = 1;  % Default for Ripple
                else
                    error('Unknown system: %s', params.system);
                end
                
                % Get channel info and filter
                info = device_com('get_chs_info');
                
                remove_channels = params.remove_channels;
                remove_channels_by_label = params.remove_channels_by_label;
                
                for ci = 1:numel(info.ch)
                    if any(info.ch(ci) == remove_channels)
                        continue
                    end
                    rem_ch = false;
                    for i = 1:numel(remove_channels_by_label)
                        if ~isempty(regexp(info.label{ci}, remove_channels_by_label{i}, 'match'))
                            rem_ch = true;
                            break;
                        end
                    end
                    if rem_ch
                        continue
                    end
                    if info.ismicro(ci)
                        channels(end+1) = info.ch(ci); %#ok<AGROW>
                        chan_label{end+1} = info.label{ci}; %#ok<AGROW>
                        conversion(end+1) = info.conversion(ci); %#ok<AGROW>
                    end
                end
                
                num_channels = numel(channels);
                fprintf('Worker: Found %d micro channels.\n', num_channels);
                
                % Send channel info back to main thread
                channel_info = struct();
                channel_info.channels = channels;
                channel_info.chan_label = chan_label;
                channel_info.conversion = conversion;
                channel_info.ev_channels = ev_channels;
                channel_info.full_info = info;
                send(data_queue, struct('type', 'channel_info', 'info', channel_info));
                
                % Enable channels
                device_com('enable_chs', channels, true, ev_channels);
                device_com('clear_buffer');
                
                fprintf('Worker: device_com opened and channels enabled.\n');
                
            catch ME
                % Send error to main thread
                send(data_queue, struct('type', 'error', 'message', ME.message));
                return;
            end
            
            % Signal ready
            send(data_queue, struct('type', 'ready'));
            
            % Initialize buffers
            data_buffers = cell(num_channels, 1);
            for i = 1:num_channels
                data_buffers{i} = zeros(max_buffer_samples, 1);
            end
            data_counts = zeros(num_channels, 1);
            timestamps = {};
            event_times = [];
            event_values = [];
            pdiode_times = [];
            data_loss_detected = false;
            collection_count = 0;
            
            collecting = false;
            running = true;
            
            % Main collection loop
            while running
                % Check for commands (non-blocking poll)
                [cmd, has_cmd] = poll(cmd_queue, 0);  % 0 timeout = non-blocking
                
                if has_cmd
                    switch cmd.cmd
                        case 'start'
                            % Reset buffers for new trial
                            for i = 1:num_channels
                                data_buffers{i}(:) = 0;
                            end
                            data_counts(:) = 0;
                            timestamps = {};
                            event_times = [];
                            event_values = [];
                            pdiode_times = [];
                            data_loss_detected = false;
                            collection_count = 0;
                            
                            device_com('clear_buffer');
                            collecting = true;
                            
                        case 'stop'
                            collecting = false;
                            
                            % Do final collection
                            try
                                [data_buffers, data_counts, timestamps, event_times, event_values, pdiode_times, data_loss_detected, collection_count] = ...
                                    BackgroundDataCollectorWorker.collect_data(...
                                        data_buffers, data_counts, timestamps, event_times, event_values, pdiode_times, ...
                                        data_loss_detected, collection_count, num_channels, max_buffer_samples, use_photodiodo);
                            catch
                            end
                            
                            % Package and send data
                            data = cell(num_channels, 1);
                            for i = 1:num_channels
                                data{i} = data_buffers{i}(1:data_counts(i));
                            end
                            
                            events = struct();
                            events.time = event_times;
                            events.value = event_values;
                            events.pdiode = pdiode_times;
                            events.timestamps = timestamps;
                            events.data_loss = data_loss_detected;
                            events.collection_count = collection_count;
                            
                            send(data_queue, struct('type', 'data', ...
                                'data', {data}, 'counts', data_counts, 'events', events));
                            
                        case 'shutdown'
                            running = false;
                            
                        case 'status'
                            % Could send status info here
                    end
                end
                
                % Collect data if active
                if collecting
                    try
                        [data_buffers, data_counts, timestamps, event_times, event_values, pdiode_times, data_loss_detected, collection_count] = ...
                            BackgroundDataCollectorWorker.collect_data(...
                                data_buffers, data_counts, timestamps, event_times, event_values, pdiode_times, ...
                                data_loss_detected, collection_count, num_channels, max_buffer_samples, use_photodiodo);
                    catch ME
                        send(data_queue, struct('type', 'error', 'message', ME.message));
                    end
                end
                
                % Small pause to prevent CPU spinning
                pause(poll_interval);
            end
            
            % Cleanup
            try
                device_com('close');
                fprintf('Worker: device_com closed.\n');
            catch
            end
            
            send(data_queue, struct('type', 'shutdown_complete'));
        end
        
        function [data_buffers, data_counts, timestamps, event_times, event_values, pdiode_times, data_loss_detected, collection_count] = ...
                collect_data(data_buffers, data_counts, timestamps, event_times, event_values, pdiode_times, ...
                             data_loss_detected, collection_count, num_channels, max_buffer_samples, use_photodiodo)
            % Collect data from device buffer
            
            streams = device_com('get_stream');
            
            if isempty(streams) || ~isfield(streams, 'data')
                return;
            end
            
            collection_count = collection_count + 1;
            
            % Store timestamp
            if isfield(streams, 'timestamp')
                timestamps{end+1} = streams.timestamp;
            end
            
            % Accumulate data for each channel
            num_ch = min(num_channels, size(streams.data, 1));
            for i = 1:num_ch
                % Handle lost data before current segment
                if streams.lost_prev(i) > 0
                    lost = streams.lost_prev(i);
                    start_idx = data_counts(i) + 1;
                    end_idx = start_idx + lost - 1;
                    if end_idx <= max_buffer_samples
                        data_buffers{i}(start_idx:end_idx) = 0;
                        data_counts(i) = end_idx;
                    end
                    data_loss_detected = true;
                end
                
                % Store current segment
                ch_data = streams.data{i};
                if ~isempty(ch_data)
                    n_samples = length(ch_data);
                    start_idx = data_counts(i) + 1;
                    end_idx = start_idx + n_samples - 1;
                    
                    if end_idx > max_buffer_samples
                        end_idx = max_buffer_samples;
                        n_samples = end_idx - start_idx + 1;
                        data_loss_detected = true;
                    end
                    
                    if n_samples > 0
                        data_buffers{i}(start_idx:end_idx) = ch_data(1:n_samples);
                        data_counts(i) = end_idx;
                    end
                end
                
                % Handle lost data after current segment
                if streams.lost_post(i) > 0
                    lost = streams.lost_post(i);
                    start_idx = data_counts(i) + 1;
                    end_idx = start_idx + lost - 1;
                    if end_idx <= max_buffer_samples
                        data_buffers{i}(start_idx:end_idx) = 0;
                        data_counts(i) = end_idx;
                    end
                    data_loss_detected = true;
                end
            end
            
            % Accumulate parallel port events
            if isfield(streams, 'parallel') && ~isempty(streams.parallel.values)
                event_times = [event_times; double(streams.parallel.times(:)) / 30];
                event_values = [event_values; streams.parallel.values(:)];
            end
            
            % Accumulate photodiode events
            if use_photodiodo && isfield(streams, 'analog_ev_t') && ~isempty(streams.analog_ev_t)
                if ~isempty(streams.analog_ev_t{1})
                    pdiode_times = [pdiode_times; double(streams.analog_ev_t{1}(:)) / 30];
                end
            end
        end
    end
end
