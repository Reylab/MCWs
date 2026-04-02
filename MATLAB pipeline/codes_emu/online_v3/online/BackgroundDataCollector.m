classdef BackgroundDataCollector < handle
    % BACKGROUNDDATACOLLECTOR - Timer-based data collection on main thread
    %   Uses MATLAB timer to collect data automatically at regular intervals.
    %   This runs on the main thread but is non-blocking - PTB continues
    %   while the timer fires between frames.
    %
    %   Unlike parfeval-based approach, this works because:
    %   1. Timer callbacks run on main thread (can access device_com)
    %   2. Timer fires asynchronously between Screen flips
    %   3. Collection is fast enough to fit in slack time
    %
    %   Usage:
    %       collector = BackgroundDataCollector(num_channels);
    %       collector.start();
    %       % ... PTB presentation ...
    %       [data, counts, events] = collector.get_trial_data();
    %       collector.stop();
    
    properties (SetAccess = private)
        num_channels        % Number of channels
        sample_rate         % Sampling rate (Hz)
        use_photodiode      % Whether to collect photodiode events
        poll_interval       % Timer interval (seconds)
        max_buffer_samples  % Maximum samples per channel
        
        % Timer
        collection_timer    % MATLAB timer object
        is_running          % Collection state
        
        % Data buffers
        data_buffers        % Cell array of data for each channel
        data_counts         % Sample counts per channel
        timestamps          % Stream timestamps
        event_times         % Event times
        event_values        % Event values
        pdiode_times        % Photodiode times
        
        % Statistics
        collection_count    % Number of collections performed
        total_samples       % Total samples collected
        data_loss_detected  % Whether data loss was detected
        last_collection_time % Time of last collection
    end
    
    methods
        function obj = BackgroundDataCollector(num_channels, varargin)
            % Constructor
            p = inputParser;
            addRequired(p, 'num_channels');
            addParameter(p, 'sample_rate', 30000);
            addParameter(p, 'use_photodiode', true);
            addParameter(p, 'poll_interval', 0.050);  % 50ms default - less aggressive
            addParameter(p, 'max_duration', 60);  % Max trial duration in seconds
            parse(p, num_channels, varargin{:});
            
            obj.num_channels = p.Results.num_channels;
            obj.sample_rate = p.Results.sample_rate;
            obj.use_photodiode = p.Results.use_photodiode;
            obj.poll_interval = p.Results.poll_interval;
            obj.max_buffer_samples = ceil(p.Results.max_duration * obj.sample_rate);
            
            obj.is_running = false;
            obj.collection_timer = [];
            
            % Initialize buffers
            obj.reset_buffers();
        end
        
        function reset_buffers(obj)
            % Reset all data buffers for new trial
            obj.data_buffers = cell(obj.num_channels, 1);
            for i = 1:obj.num_channels
                obj.data_buffers{i} = zeros(obj.max_buffer_samples, 1);
            end
            obj.data_counts = zeros(obj.num_channels, 1);
            obj.timestamps = {};
            obj.event_times = [];
            obj.event_values = [];
            obj.pdiode_times = [];
            obj.collection_count = 0;
            obj.total_samples = 0;
            obj.data_loss_detected = false;
            obj.last_collection_time = tic;
        end
        
        function start(obj)
            % Start timer-based collection
            if obj.is_running
                return;
            end
            
            obj.is_running = true;
            obj.last_collection_time = tic;
            
            % Create and start timer
            obj.collection_timer = timer(...
                'ExecutionMode', 'fixedSpacing', ...
                'Period', obj.poll_interval, ...
                'TimerFcn', @(~,~) obj.collect_callback(), ...
                'ErrorFcn', @(~,e) obj.error_callback(e), ...
                'BusyMode', 'drop');  % Drop if previous callback still running
            
            start(obj.collection_timer);
        end
        
        function stop(obj)
            % Stop timer and do final collection
            if ~obj.is_running
                return;
            end
            
            % Stop and delete timer FIRST to prevent new callbacks
            if ~isempty(obj.collection_timer) && isvalid(obj.collection_timer)
                stop(obj.collection_timer);
                delete(obj.collection_timer);
            end
            obj.collection_timer = [];
            
            % Do one final collection to get any remaining data
            % (is_running is still true so collect_data_now will execute)
            obj.collect_data_now();
            
            % Mark as stopped AFTER final collection
            obj.is_running = false;
        end
        
        function [data, counts, events] = get_trial_data(obj, stop_first)
            % Get accumulated trial data
            if nargin < 2
                stop_first = true;
            end
            
            if stop_first && obj.is_running
                obj.stop();
            end
            
            % Package data (trim to actual counts)
            data = cell(obj.num_channels, 1);
            for i = 1:obj.num_channels
                data{i} = obj.data_buffers{i}(1:obj.data_counts(i));
            end
            counts = obj.data_counts;
            
            events.time = obj.event_times;
            events.value = obj.event_values;
            events.pdiode = obj.pdiode_times;
            events.timestamps = obj.timestamps;
            events.data_loss = obj.data_loss_detected;
        end
        
        function stats = get_statistics(obj)
            % Get collection statistics
            stats.collection_count = obj.collection_count;
            stats.total_samples = obj.total_samples;
            stats.data_loss = obj.data_loss_detected;
            stats.samples_per_channel = obj.data_counts;
            stats.buffer_usage = max(obj.data_counts) / obj.max_buffer_samples * 100;
        end
        
        function collect_now(obj)
            % Manual collection trigger (for explicit collection points)
            obj.collect_data_now();
        end
        
        function delete(obj)
            % Destructor - cleanup timer
            obj.stop();
        end
    end
    
    methods (Access = private)
        function collect_callback(obj)
            % Timer callback - collect data
            if ~obj.is_running
                return;
            end
            obj.collect_data_now();
        end
        
        function collect_data_now(obj)
            % Actual data collection - matches device_com('get_stream') structure
            if ~obj.is_running
                return;
            end
            try
                streams = device_com('get_stream');
                
                if isempty(streams) || ~isfield(streams, 'data')
                    return;
                end
                
                obj.collection_count = obj.collection_count + 1;
                
                % Store timestamp (singular)
                if isfield(streams, 'timestamp')
                    obj.timestamps{end+1} = streams.timestamp;
                end
                
                % Accumulate data for each channel
                % streams.data is a cell array, streams.lost_prev/lost_post are arrays
                num_ch = size(streams.data, 1);
                for i = 1:min(obj.num_channels, num_ch)
                    % Handle lost data before current segment
                    if streams.lost_prev(i) > 0
                        lost = streams.lost_prev(i);
                        start_idx = obj.data_counts(i) + 1;
                        end_idx = start_idx + lost - 1;
                        if end_idx <= obj.max_buffer_samples
                            obj.data_buffers{i}(start_idx:end_idx) = 0;
                            obj.data_counts(i) = end_idx;
                        end
                        obj.data_loss_detected = true;
                    end
                    
                    % Store current segment
                    ch_data = streams.data{i};
                    if isempty(ch_data)
                        continue;
                    end
                    
                    n_samples = length(ch_data);
                    start_idx = obj.data_counts(i) + 1;
                    end_idx = start_idx + n_samples - 1;
                    
                    % Check for buffer overflow
                    if end_idx > obj.max_buffer_samples
                        end_idx = obj.max_buffer_samples;
                        n_samples = end_idx - start_idx + 1;
                        obj.data_loss_detected = true;
                    end
                    
                    if n_samples > 0
                        obj.data_buffers{i}(start_idx:end_idx) = ch_data(1:n_samples);
                        obj.data_counts(i) = end_idx;
                        obj.total_samples = obj.total_samples + n_samples;
                    end
                    
                    % Handle lost data after current segment
                    if streams.lost_post(i) > 0
                        lost = streams.lost_post(i);
                        start_idx = obj.data_counts(i) + 1;
                        end_idx = start_idx + lost - 1;
                        if end_idx <= obj.max_buffer_samples
                            obj.data_buffers{i}(start_idx:end_idx) = 0;
                            obj.data_counts(i) = end_idx;
                        end
                        obj.data_loss_detected = true;
                    end
                end
                
                % Accumulate parallel port events (DAQ)
                if isfield(streams, 'parallel') && ~isempty(streams.parallel.values)
                    obj.event_times = [obj.event_times; double(streams.parallel.times(:)) / 30];  % Convert to ms
                    obj.event_values = [obj.event_values; streams.parallel.values(:)];
                end
                
                % Accumulate photodiode events
                if obj.use_photodiode && isfield(streams, 'analog_ev_t') && ~isempty(streams.analog_ev_t)
                    if ~isempty(streams.analog_ev_t{1})
                        obj.pdiode_times = [obj.pdiode_times; double(streams.analog_ev_t{1}(:)) / 30];  % Convert to ms
                    end
                end
                
                obj.last_collection_time = tic;
                
            catch ME
                % Log error but don't stop collection
                warning('BackgroundDataCollector:CollectionError', ...
                    'Collection error: %s', ME.message);
            end
        end
        
        function error_callback(obj, e)
            % Timer error callback
            warning('BackgroundDataCollector:TimerError', ...
                'Timer error: %s', e.Data.messageID);
        end
    end
end