classdef IFRCalculator
    %UNTITLED11 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ejex
        t_selected
        int_window
        pos_time
        neg_time
        sample_period
        resolution
        half_ancho_gauss
    end
    
    methods
        function obj = IFRCalculator(alpha_gauss,sigma_gauss,resolution_ms,sr,time_pre_ms,time_pos_ms)
            gauss_sample_period = 1000/sr;
            half_ancho_gauss = alpha_gauss * sigma_gauss;
            N_gauss = 2*round(half_ancho_gauss/gauss_sample_period)+1; % Number of points of the gaussian window
            ejex = -time_pre_ms-half_ancho_gauss:resolution_ms:time_pos_ms+half_ancho_gauss;  %should be the same length as spike_timeline
            obj.resolution = resolution_ms;
            obj.int_window = gausswin(N_gauss, alpha_gauss);
            obj.int_window = obj.int_window(1:ceil(resolution_ms/gauss_sample_period):end);
            %obj.int_window = 1000*obj.int_window/sum(obj.int_window)/sample_period;
            obj.int_window = obj.int_window/sum(obj.int_window)*1000/resolution_ms;
            obj.t_selected = find(ejex>-time_pre_ms,1)-1:1:find(ejex>time_pos_ms,1);
            
            obj.half_ancho_gauss = half_ancho_gauss;
            obj.pos_time = time_pos_ms+half_ancho_gauss;
            obj.neg_time = -time_pre_ms-half_ancho_gauss;
            obj.sample_period = resolution_ms;
            %number of elements from number of elemens in histogram 
            obj.ejex = ejex(obj.t_selected);
        end
        
        function IFR = get_ifr(obj,spikes1)
            if iscell(spikes1)
                lst = numel(spikes1);
                spikes1 = sort(horzcat(spikes1{:}));
            else
                lst = size(spikes1,1);
                spikes1=sort(spikes1(:));
                spikes1=spikes1(spikes1~=10000);
            end
            
            spikes_tot=spikes1(spikes1 < obj.pos_time & spikes1 > obj.neg_time);
            spike_timeline = histcounts(spikes_tot,(obj.neg_time:obj.resolution:obj.pos_time))/lst;
            
            IFR = conv(spike_timeline, obj.int_window, 'same');
            IFR =  single(IFR(obj.t_selected));
            
        end

        function IFR = get_ifr_pre_pos(obj, spikes, pre, pos)
            pos_t = pos + obj.half_ancho_gauss;
            pre_t = pre - obj.half_ancho_gauss;
            hist_bins = pre_t:obj.resolution:pos_t;
            if iscell(spikes)
                lst = numel(spikes);
                spikes = sort(horzcat(spikes{:}));
            else
                lst    = size(spikes,1);
                spikes = sort(spikes(:));
                spikes = spikes(spikes~=10000);
            end

            obj.t_selected = find(hist_bins>pre,1)-1:1:find(hist_bins>pos,1);
            obj.ejex = hist_bins(obj.t_selected);
            
            spikes_tot = spikes(spikes < pos_t & spikes > pre_t);
            spike_timeline = histcounts(spikes_tot,hist_bins)/lst;

            IFR = conv(spike_timeline, obj.int_window, 'same');
            IFR = single(IFR(obj.t_selected));            
        end
        function IFR = kde_ifr(obj, spikes, pre, pos)
            % spikes: vector of spike times
            % time_window: [start, end] time window for IFR calculation

            if iscell(spikes)
                spikes = sort(horzcat(spikes{:}));
            else
                spikes = sort(spikes(:));
                spikes = spikes(spikes~=10000);
            end

            time_window = [pre, pos];
            % bandwidth: KDE bandwidth (controls smoothing)
            bandwidth = 10; % KDE bandwidth in ms

            % Create a grid of time points
            time_grid = linspace(time_window(1), time_window(2), ...
                                 time_window(2) - time_window(1));
        
            % Calculate the KDE
            kde = ksdensity(spikes, time_grid, 'Bandwidth', bandwidth);
        
            % Normalize the KDE to get the IFR
            IFR = kde / (sum(kde) * (time_grid(2) - time_grid(1)));
        
            % Optional: smooth the IFR with a Gaussian window
            % IFR = conv(IFR, gausswin(10), 'same');

            % Convert to Hz
            IFR = IFR * 1000;
        end
    end
end

