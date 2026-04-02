classdef ContinuousDataCollector < handle
    % CONTINUOUSDATACOLLECTOR - Background data collection from neural device
    %   This class runs continuous data collection in a background worker,
    %   preventing any interruption to PsychToolbox timing. Data is collected
    %   via a DataQueue and accumulated in the main thread.
    %
    %   Usage:
    %       collector = ContinuousDataCollector(n_channels, sample_rate);
    %       collector.start();
    %       ... PTB stimulus presentation ...
    %       [data, events] = collector.get_accumulated_data();
    %       collector.stop();
    %
    %   The collector runs in a background worker, continuously polling
    %   device_com('get_stream') and sending data back via DataQueue.
    
    properties
        % Configuration
        n_channels          % Number of channels
        sample_rate = 30000 % Sample rate in Hz
        use_photodiode = false
        poll_interval = 0.005  % Polling interval in seconds (5ms default)
        
        % Data storage (accumulated in main thread)
        data                % Cell array of channel data
        datacounter         % Current position in each channel buffer
        timestamps          % Stream timestamps
        
        % Event storage
        Event_Time          % DAQ event times
        Event_Value         % DAQ event values
        Event_Time_pdiode   % Photodiode event times
        
        % State tracking
        is_running = false
        data_loss_detected = false
        total_samples_collected = 0
        
        % Background worker
        collector_future    % parfeval Future object
        data_queue          % DataQueue for receiving data
        stop_flag           % Shared flag to signal stop
        
        % Buffer settings
        max_buffer_size = 30000 * 600  % 10 minutes of data per channel
        
        % Statistics
        collection_count = 0
        last_collection_time
    end
    
    properties (Access = private)
        queue_listener      % Listener for data queue
    end
    
    methods
        function obj = ContinuousDataCollector(n_channels, varargin)
            % Constructor
            %   collector = ContinuousDataCollector(n_channels, ...)
            %
            % Optional parameters:
            %   'sample_rate'      - Sample rate in Hz (default: 30000)
            %   'use_photodiode'   - Enable photodiode events (default: false)
            %   'poll_interval'    - Polling interval in seconds (default: 0.005)
            %   'max_duration'     - Max recording duration in seconds (default: 600)
            
            p = inputParser;
            addParameter(p, 'sample_rate', 30000);
            addParameter(p, 'use_photodiode', false);
            addParameter(p, 'poll_interval', 0.005);
            addParameter(p, 'max_duration', 600);
            parse(p, varargin{:});
            
            obj.n_channels = n_channels;
            obj.sample_rate = p.Results.sample_rate;
            obj.use_photodiode = p.Results.use_photodiode;
            obj.poll_interval = p.Results.poll_interval;
            obj.max_buffer_size = obj.sample_rate * p.Results.max_duration;
            
            % Initialize data buffers
            obj.reset_buffers();
            
            fprintf('ContinuousDataCollector initialized for %d channels\n', n_channels);
        end
        
        function reset_buffers(obj)
            % Reset all data buffers to initial state
            obj.data = cell(obj.n_channels, 1);
            obj.datacounter = zeros(obj.n_channels, 1);
            for i = 1:obj.n_channels
                obj.data{i} = zeros(1, obj.max_buffer_size, 'int16');
            end
            
            obj.timestamps = {};
            obj.Event_Time = [];
            obj.Event_Value = [];
            obj.Event_Time_pdiode = [];
            obj.data_loss_detected = false;
            obj.total_samples_collected = 0;
            obj.collection_count = 0;
        end
        
        function start(obj)
            % Start continuous background data collection
            if obj.is_running
                warning('Collector is already running');
                return;
            end
            
            % Create data queue for receiving data from worker
            obj.data_queue = parallel.pool.DataQueue;
            
            % Setup listener to process incoming data
            obj.queue_listener = afterEach(obj.data_queue, @(data) obj.process_queue_data(data));
            
            % Create stop flag using a pollable data queue
            obj.stop_flag = parallel.pool.PollableDataQueue;
            
            obj.is_running = true;
            obj.last_collection_time = tic;
            
            % Start background collection worker
            obj.collector_future = parfeval(@obj.collection_worker, 0, ...
                obj.data_queue, obj.stop_flag, obj.poll_interval, obj.use_photodiode);
            
            fprintf('Background data collection started\n');
        end
        
        function stop(obj)
            % Stop background data collection
            if ~obj.is_running
                return;
            end
            
            % Signal worker to stop
            send(obj.stop_flag, true);
            
            % Wait for worker to finish (with timeout)
            try
                wait(obj.collector_future, 5);
            catch
                cancel(obj.collector_future);
            end
            
            obj.is_running = false;
            
            % Delete listener
            if ~isempty(obj.queue_listener)
                delete(obj.queue_listener);
                obj.queue_listener = [];
            end
            
            fprintf('Background data collection stopped. Total collections: %d\n', obj.collection_count);
        end
        
        function [data_out, datacounter_out, events] = get_accumulated_data(obj)
            % Get all accumulated data (non-blocking)
            %   [data, datacounter, events] = collector.get_accumulated_data()
            %
            % Returns current state of accumulated data without blocking.
            % Call this between trials or at end of subscreening.
            
            % Process any pending queue items first
            drawnow('limitrate');
            
            data_out = obj.data;
            datacounter_out = obj.datacounter;
            
            events = struct();
            events.time = obj.Event_Time;
            events.value = obj.Event_Value;
            events.pdiode = obj.Event_Time_pdiode;
            events.data_loss = obj.data_loss_detected;
        end
        
        function [data_out, datacounter_out, events] = get_trial_data(obj, reset_events)
            % Get accumulated data and optionally reset event buffers
            %   [data, datacounter, events] = collector.get_trial_data(reset_events)
            %
            % If reset_events is true, clears event buffers after returning.
            
            if nargin < 2
                reset_events = false;
            end
            
            % Process pending queue data
            drawnow('limitrate');
            
            data_out = obj.data;
            datacounter_out = obj.datacounter;
            
            events = struct();
            events.time = obj.Event_Time;
            events.value = obj.Event_Value;
            events.pdiode = obj.Event_Time_pdiode;
            events.data_loss = obj.data_loss_detected;
            events.timestamps = obj.timestamps;
            
            if reset_events
                obj.Event_Time = [];
                obj.Event_Value = [];
                obj.Event_Time_pdiode = [];
                obj.timestamps = {};
                obj.data_loss_detected = false;
            end
        end
        
        function mark_trial_boundary(obj)
            % Mark current position as trial boundary (for reference)
            % Can be used to segment data later
            obj.timestamps{end+1} = struct('type', 'trial_boundary', ...
                'position', obj.datacounter, 'time', GetSecs);
        end
        
        function stats = get_statistics(obj)
            % Get collection statistics
            stats = struct();
            stats.is_running = obj.is_running;
            stats.collection_count = obj.collection_count;
            stats.total_samples = obj.total_samples_collected;
            stats.data_loss = obj.data_loss_detected;
            stats.buffer_usage = max(obj.datacounter) / obj.max_buffer_size * 100;
            if obj.collection_count > 0 && ~isempty(obj.last_collection_time)
                stats.elapsed_time = toc(obj.last_collection_time);
                stats.avg_samples_per_sec = obj.total_samples_collected / stats.elapsed_time;
            else
                stats.elapsed_time = 0;
                stats.avg_samples_per_sec = 0;
            end
        end
        
        function delete(obj)
            % Destructor - ensure cleanup
            obj.stop();
        end
    end
    
    methods (Access = private)
        function process_queue_data(obj, queue_data)
            % Process data received from background worker
            % This runs in the main thread when data arrives
            
            if isempty(queue_data) || ~isstruct(queue_data)
                return;
            end
            
            % Accumulate neural data
            if isfield(queue_data, 'streams') && ~isempty(queue_data.streams)
                streams = queue_data.streams;
                
                for jj = 1:min(size(streams.data, 1), obj.n_channels)
                    % Handle lost data before segment
                    if streams.lost_prev(jj) > 0
                        lost_count = min(streams.lost_prev(jj), ...
                            obj.max_buffer_size - obj.datacounter(jj));
                        if lost_count > 0
                            obj.data{jj}(obj.datacounter(jj) + (1:lost_count)) = 0;
                            obj.datacounter(jj) = obj.datacounter(jj) + lost_count;
                        end
                        obj.data_loss_detected = true;
                    end
                    
                    % Store current segment
                    lseg = length(streams.data{jj});
                    if lseg > 0
                        space_left = obj.max_buffer_size - obj.datacounter(jj);
                        samples_to_store = min(lseg, space_left);
                        if samples_to_store > 0
                            obj.data{jj}(obj.datacounter(jj) + (1:samples_to_store)) = streams.data{jj}(1:samples_to_store);
                            obj.datacounter(jj) = obj.datacounter(jj) + samples_to_store;
                            obj.total_samples_collected = obj.total_samples_collected + samples_to_store;
                        end
                    end
                    
                    % Handle lost data after segment
                    if streams.lost_post(jj) > 0
                        lost_count = min(streams.lost_post(jj), ...
                            obj.max_buffer_size - obj.datacounter(jj));
                        if lost_count > 0
                            obj.data{jj}(obj.datacounter(jj) + (1:lost_count)) = 0;
                            obj.datacounter(jj) = obj.datacounter(jj) + lost_count;
                        end
                        obj.data_loss_detected = true;
                    end
                end
                
                % Store timestamp
                if isfield(streams, 'timestamp')
                    obj.timestamps{end+1} = streams.timestamp;
                end
            end
            
            % Accumulate DAQ events
            if isfield(queue_data, 'parallel') && ~isempty(queue_data.parallel)
                if ~isempty(queue_data.parallel.times)
                    obj.Event_Time = [obj.Event_Time; double(queue_data.parallel.times) / 30];
                    obj.Event_Value = [obj.Event_Value; queue_data.parallel.values];
                end
            end
            
            % Accumulate photodiode events
            if isfield(queue_data, 'pdiode') && ~isempty(queue_data.pdiode)
                obj.Event_Time_pdiode = [obj.Event_Time_pdiode; queue_data.pdiode];
            end
            
            obj.collection_count = obj.collection_count + 1;
        end
    end
    
    methods (Static)
        function collection_worker(data_queue, stop_flag, poll_interval, use_photodiode)
            % Background worker function that continuously collects data
            % This runs in a parallel worker, isolated from PTB
            
            fprintf('Collection worker started (poll interval: %.3f sec)\n', poll_interval);
            
            while true
                % Check for stop signal
                [has_data, ~] = poll(stop_flag, 0);
                if has_data
                    break;
                end
                
                % Collect data from device
                try
                    streams = device_com('get_stream');
                    
                    % Package data to send back
                    queue_data = struct();
                    queue_data.streams = streams;
                    
                    % Extract parallel port events
                    if isfield(streams, 'parallel') && ~isempty(streams.parallel.values)
                        queue_data.parallel = streams.parallel;
                    else
                        queue_data.parallel = [];
                    end
                    
                    % Extract photodiode events
                    if use_photodiode && isfield(streams, 'analog_ev_t') && ~isempty(streams.analog_ev_t)
                        if ~isempty(streams.analog_ev_t{1})
                            queue_data.pdiode = double(streams.analog_ev_t{1}(:)) / 30;
                        else
                            queue_data.pdiode = [];
                        end
                    else
                        queue_data.pdiode = [];
                    end
                    
                    % Send data to main thread
                    send(data_queue, queue_data);
                    
                catch ME
                    % Log error but continue
                    warning('Collection error: %s', ME.message);
                end
                
                % Wait before next poll
                pause(poll_interval);
            end
            
            fprintf('Collection worker stopped\n');
        end
    end
end
