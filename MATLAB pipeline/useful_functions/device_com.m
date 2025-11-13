function varargout = device_com(varargin)
    % inputs: command, command arguments
    % commands:
    %   'open': arguments: 'BRK' or 'RIP' for BLK it requires
    %       central ip and instance as next inptus. When RIP, optional
    %       input path of the map file. Default: 'ripple.map', if empty
    %       won't use custom labels.
    %   'get_chs_info': 
    %   'enable_chs':
    %   'get_stream':
    %   'close': close current device
    
    persistent curr_device
    if strcmp(varargin{1},'open')
        p = inputParser; 
        addOptional(p,'address',[]); addOptional(p,'instance',0);
        addOptional(p,'use_tcp',1);  addOptional(p,'mapfile','ripple.map');
        addOptional(p,'nsp_type',256);  
        parse(p,varargin{3:end});
        if ~isempty(curr_device)
            device_com('close');
            pause(0.02);
        end
        
        if strcmp(varargin{2},'BRK') || strcmp(varargin{2},'BLK')
            add_device_path_blk()
            cbmex('open', 'central-addr',p.Results.address,'instance',p.Results.instance);
            if p.Results.nsp_type==256
                dig_ch = 279;
            else
                dig_ch = 151;
            end
            curr_device = struct('type','BRK','instance',p.Results.instance,...
                'central_addr',p.Results.address,'dig_ch',dig_ch);
        elseif strcmp(varargin{2},'RIP') || strcmp(varargin{2},'RIPL')
            add_device_path_ripple()
            open_ripple(p.Results.use_tcp)
            curr_device = struct('type','RIP','mapfile',p.Results.mapfile);
        elseif strcmp(varargin{2},'NCx')
            if ~exist([p.Results.address filesep 'NSx.mat'],'file')
                error('Folder without NSx.mat file.')
            end
            curr_device = struct('type','NCx','address',p.Results.address,...
                'ev_chs',[],'chs',[],'parallel',false,'start_time',-1);            
        else
            error('Invalid device: %s', varargin{2});
        end
        return
    end
    if isempty(curr_device)
        error('ERROR: run open first.')
    end
    if strcmp(curr_device.type,'BRK')
       switch varargin{1}
       case 'enable_chs'
            curr_device = enable_chs_blk(varargin{2},varargin{3},varargin{4}, curr_device); %enable_chs_blk(chs,ev_chs)
        case 'get_stream'
           [varargout{1},curr_device] = get_stream_blk(curr_device); 
        case 'get_chs_info'
            varargout{1} = get_chs_info_blk(curr_device.instance);
        case 'clear_buffer'
            %clear_buffer_blk(curr_device.instance);
             get_stream_blk(curr_device); 
        case 'close'
%           cbmex('close')
          curr_device = [];
        otherwise
          error('Command not implemented: %s', varargin{1});
        end
    elseif strcmp(curr_device.type,'RIP')
        switch varargin{1}
        case 'get_chs_info'
            if length(varargin) == 2
                mac_stream_type = varargin{2};
            else
                mac_stream_type = 'hifreq';
            end
          [varargout{1}, curr_device] = get_chs_info_ripple(curr_device, mac_stream_type);
        case 'enable_chs'
           curr_device = enable_chs_ripple(varargin{2},varargin{3},varargin{4},curr_device);
        case 'get_stream'
           [varargout{1},curr_device] = get_stream_ripple(curr_device); 
        case 'clear_buffer'
            curr_device = clear_buffer_ripple(curr_device);
        case 'close'
          xippmex('close')
          curr_device = [];
        otherwise
            error('Command not implemented: %s', varargin{1});
        end
    elseif strcmp(curr_device.type,'NCx')
        switch varargin{1}
        case 'get_chs_info'
          [varargout{1}, curr_device] = get_chs_info_ncx(curr_device);
        case 'enable_chs'
           curr_device = enable_chs_ncx(varargin{2},varargin{3},varargin{4},curr_device);
        case 'get_stream'
           [varargout{1},curr_device] = get_stream_ncx(curr_device);
        case 'clear_buffer'
            disp(''); %nothing
        case 'close'
          close_ncx(curr_device); 
          curr_device = [];
        otherwise
            error('Command not implemented: %s', varargin{1});
        end
    else
        error('Invalid device: %s', device_type);
    end
end

function add_device_path_blk()

    paths = {'C:\Program Files\Blackrock Microsystems\NeuroPort Windows Suite',...
    'C:\Program Files (x86)\Blackrock Microsystems\Cerebus Windows Suite',...
    'C:\Program Files (x86)\Blackrock Microsystems\NeuroPort Windows Suite'};
    for path = paths
        if exist(path{1}, 'dir')
            addpath(genpath(path{1}));
            break;
        end
    end
end

function add_device_path_ripple()
    if contains(path,'xippmex')
        return
    end
    if isunix
        addpath('/opt/Trellis/Tools/xippmex/')
    else %windows
        valnames = winqueryreg('name','HKEY_LOCAL_MACHINE','SOFTWARE\Microsoft\Windows\CurrentVersion\Installer\Folders');
        xippmexix = find(cellfun(@(x) ~isempty(regexp(x,'(Ripple\\Trellis\\Tools\\xippmex\\)$','match')),valnames));

        if isempty(xippmexix)
            error('Trellis not found, add path using pathtool.')
        else
            addpath(valnames{xippmexix})
        end
    end
end

function [streams, blk_info] = get_stream_blk(blk_info)
    %streams struct: 
    %  data cells of data
    %  lost_prev 
    %  lost_post
    %  sr
    instance = blk_info.instance;
    [ts_array, t_ini_buffer_prev, blk_buffer] = cbmex('trialdata',1,'instance',instance);
    pause(0.02)
    [~, t_ini_buffer_curr, ~] = cbmex('trialdata',0,'instance',instance);
    pause(0.02)
%     while (size(blk_buffer,1) == 0)
%         pause(0.02)
%         [ts_array, t_ini_buffer, blk_buffer] = cbmex('trialdata',1,'instance',instance);
%     end
    available_chs = cell2mat(blk_buffer(:,1));
    chs = blk_info.enabled_chs;
    streams = struct;
    
    if ~isempty(chs)
        streams.timestamp = zeros(length(chs),1);
        streams.data = cell(length(chs),1);
        streams.lost_prev = zeros(length(chs),1);
        streams.sr = zeros(length(chs),1);
        streams.lost_post = zeros(length(chs),1);

        for ci = 1:length(chs)
            aci = find(chs(ci)==available_chs, 1);
            if isempty(aci)
                continue
            else
                streams.data{ci} = blk_buffer{aci,3}; %output dimension must be Nx1
                streams.sr(ci) = blk_buffer{aci,2};
                news = size(blk_buffer{aci,3},1);
                if news >= 102399 %by default max datapoints per channel 102400
                    streams.lost_post(ci) = round((t_ini_buffer_curr-t_ini_buffer_prev)*streams.sr(ci))-news;
                end
                streams.timestamp(ci) = t_ini_buffer_prev*30000;
            end
        end
    end
    
    if blk_info.parallel
        streams.parallel = struct; 
        dig_in_ch = ts_array{blk_info.dig_ch,3};
        if ~isempty(dig_in_ch)
            c = typecast(dig_in_ch,'uint8');
            v = c(1:2:end);
            streams.parallel.times = ts_array{blk_info.dig_ch,2} + t_ini_buffer_prev*30000;
            streams.parallel.values = v;
        else
            streams.parallel.times =[];
            streams.parallel.values = [];
        end
    end
    
    if ~isempty(blk_info.ev_chs)

        streams.analog_ev_t = cell(length(blk_info.ev_chs),1);
        ev_chs = blk_info.ev_chs;
        for i=1:length(ev_chs)
            streams.analog_ev_t{i} = ts_array{ev_chs(i),2} + t_ini_buffer_prev*30000;
        end
    end
end


function [streams, ripple_info] = get_stream_ripple(ripple_info)
    chs = ripple_info.chs;
    nch = length(chs);
    streams = struct;
    streams.data = cell(nch,1);
    streams.timestamp = zeros(nch,1);
    streams.lost_prev = zeros(nch,1);
    streams.lost_post = zeros(nch,1);
    streams.sr = ripple_info.sr;
%% start of multichannel reading


%     if ~isempty(ripple_info.enabled_mic_ch)
% 		[data, time]  = xippmex('cont',ripple_info.enabled_mic_ch,5000,'raw',ripple_info.prev_time(1));
% 		lost = 0;
% 		if isempty(data)
% 			[data, time]  = xippmex('cont',ripple_info.enabled_mic_ch,5000,'raw');
% 			lost = time - ripple_info.mic_prev_time; %sr=30k
% 			if lost<0
% 				data = data(:, ceil(-lost):end);
%                 time = time - lost;
% 				lost = 0;
% 			end
% 		end
% 		ripple_info.prev_time = double(time + size(data,2));
% 		
% 		for i =1:length(ripple_info.enabled_mic_ch)
% 			chi = find(ripple_info.enabled_mic_ch(i)==chs);
% 			streams.lost_prev(chi) = lost;
% 			streams.data{chi} = data(i,:)';
%             streams.timestamp(chi) = time;
% 		end
% 		
%     end
% 	if ~isempty(ripple_info.enabled_analog_ch)
% 		[data, time]  = xippmex('cont',ripple_info.enabled_analog_ch,5000,'30ksps',ripple_info.analog_prev_time);
% 		
% 		if isempty(data)
% 			[data, time]  = xippmex('cont',ripple_info.enabled_analog_ch,5000,'30ksps');
% 			lost = time - ripple_info.analog_prev_time; %sr=30k
% 			if lost<0
% 				data = data(:, ceil(-lost):end);
%               time = time -lost;
% 				lost = 0;
% 			end
% 		end
% 		ripple_info.analog_prev_time = double(time + size(data,2));
% 		
% 		for i =1:length(ripple_info.enabled_analog_ch)
% 			chi = find(ripple_info.enabled_analog_ch(i)==chs);
% 			streams.lost_prev(chi) = lost;
% 			streams.data{chi} = data(i,:)';
%           streams.timestamp(chi) = time;
% 		end
% 	end
%% end of multichannel reading

    %% start of single channel lines
    
    if nch>0
        for ci = 1:nch
            c = chs(ci);
            is_micro = any(c==ripple_info.enabled_mic_ch);
            is_mic_hires = any(c==ripple_info.enabled_mic_ch_hires);
            is_mac_hires = any(c==ripple_info.enabled_mac_ch_hires);
            is_mac_hifreq = any(c==ripple_info.enabled_mac_ch_hifreq);
            if is_micro
                [data, time]  = xippmex('cont',c,5000,'raw',ripple_info.prev_time(ci));
                if isempty(data)
                     [data, time]  = xippmex('cont',c,5000,'raw');
                     lost = time - ripple_info.prev_time(ci); %sr=30k
                     if lost>0
                        streams.lost_prev(ci) = lost;
                     elseif  lost<0
                         data = data(ceil(-lost):end);
                         time = time -lost;
                     end
                end
            elseif is_mic_hires || is_mac_hires
                [data, time]  = xippmex('cont',c,5000,'hi-res',ripple_info.prev_time(ci));
                if isempty(data)
                     [data, time]  = xippmex('cont',c,5000,'hi-res');
                     lost = (time - ripple_info.prev_time(ci))*streams.sr(ci)/30000;
                     if lost>0
                        streams.lost_prev(ci) = lost;
                     elseif  lost<0
                         data = data(ceil(-lost):end);
                         time = time -lost;
                     end
                end
            elseif is_mac_hifreq
                [data, time]  = xippmex('cont',c,5000,'hifreq',ripple_info.prev_time(ci));
                if isempty(data)
                     [data, time]  = xippmex('cont',c,5000,'hifreq');
                     lost = (time - ripple_info.prev_time(ci))*streams.sr(ci)/30000;
                     if lost>0
                        streams.lost_prev(ci) = lost;
                     elseif  lost<0
                         data = data(ceil(-lost):end);
                         time = time -lost;
                     end
                end
            else
                [data, time]  = xippmex('cont',c,5000,'30ksps',ripple_info.prev_time(ci));
                 if isempty(data)
                     [data, time]  = xippmex('cont',c,5000,'30ksps');
                     lost = time - ripple_info.prev_time(ci); %sr=30k
                     if lost>0
                        streams.lost_prev(ci) = lost;
                     elseif  lost<0
                         data = data(ceil(-lost):end);
                         time = time - lost;
                     end
                 end
            end
            streams.data{ci} = data';
            streams.timestamp(ci) = time;
            ripple_info.prev_time(ci) = double(time + round(size(streams.data{ci},1)*30000/streams.sr(ci)));
       end
    end


    %% end of single channel lines
    
    %% events data
    ev_chs = ripple_info.ev_chs;
    if ripple_info.parallel || ~isempty(ev_chs)
        [~, timestamps, events] = xippmex('digin');
        reason = cell2mat({events.reason});
    end 
    if ripple_info.parallel
        streams.analog_ev_t = cell(length(ripple_info.ev_chs),1);
        streams.parallel = struct;
        parallel_ev = reason==1;
        if any(parallel_ev)
            streams.parallel.values = cell2mat({events(parallel_ev).parallel});
            streams.parallel.values = streams.parallel.values(:);
            streams.parallel.times = timestamps(parallel_ev)'; %output should be column
        else
            streams.parallel.times = [];
            streams.parallel.values = [];
        end
    end
    
    
    if ~isempty(ev_chs)
        for i=1:length(ev_chs)
            ev_ix = reason == (2^ev_chs(1));
            streams.analog_ev_t{i} = timestamps(ev_ix);
        end
    end
end

function ripple_info = clear_buffer_ripple(ripple_info)
    chs = ripple_info.chs;
    nch = length(chs);
    time = xippmex('time');
    if nch>0
        ripple_info.prev_time(1:nch) = double(time);
    end
end


function clear_buffer_blk(instance)
    cbmex('trialdata',1,'instance', instance);
end

function info = get_chs_info_blk(instance)
    pause(0.01)
    cbmex('mask',0,1,'instance',instance)
    pause(0.01)
    cbmex('trialconfig',1,'noevent','instance',instance);
    pause(0.02)
    [~, x] = cbmex('trialdata',1,'instance',instance);
    chs = cell2mat(x(:,1));
    pause(0.01)
    labels = cbmex('chanlabel',chs,'instance',instance);
    pause(0.01)
    cbmex('trialconfig',0,'instance',instance);
    labels = labels(:,1);
    sr = cell2mat(x(:,2));
    
    aux_info = {};
    for c = chs(:)'
        pause(0.01)
        if isempty(aux_info)  
            aux_info = cbmex('config',c,'instance',instance);
        else
            tmp = cbmex('config',c,'instance',instance);
            aux_info = [aux_info, tmp(:,2)];
        end
    end
    
    info = struct;
    if isempty(x)  
        info.conversion = [];
        info.unit = {};
        info.smpfilter = {};
        info.parsr = [];
    else
        max_digital = cellfun(@(x) strcmp(x,'max_digital'),aux_info(:,1));
        max_analog = cellfun(@(x) strcmp(x,'max_analog'),aux_info(:,1));
        analog_unit = cellfun(@(x) strcmp(x,'analog_unit'),aux_info(:,1));
        smpfilter_n = cellfun(@(x) strcmp(x,'smpfilter'),aux_info(:,1));
        info.conversion = arrayfun(@(x) aux_info{max_analog,x+1}/aux_info{max_digital,x+1}, 1:length(chs));
        info.unit = aux_info(analog_unit,2:end);
        info.smpfilter = cellfun(@(x) smpfilter2str_blk(x), aux_info(smpfilter_n,2:end), 'UniformOutput', false);
        info.parsr = arrayfun(@(x) ['f' num2str(x)], sr, 'UniformOutput', false);
    end
    
    info.ch = chs;
    info.label = labels;
    info.sr = sr;
    info.ismicro = arrayfun(@(x) strcmp('uV',info.unit{x})&& info.sr(x)==30000,1:length(info.sr));
end

function [info,ripple_info] = get_chs_info_ripple(ripple_info, mac_stream_type)    
    micchans   = xippmex('elec','micro');
    macchans   = xippmex('elec','macro');
    analchans   = xippmex('elec','analog');
    if  isempty(micchans)
        raw_mic = [];
        hires_mic = [];
    else
        raw_mic = xippmex('signal',micchans,'raw');
        hires_mic   = xippmex('signal',micchans,'hi-res');
    end

    if  isempty(macchans)
        hires_mac = [];
        hifreq_mac = [];
    else
        hires_mac  = xippmex('signal',macchans,'hi-res');
        hifreq_mac = xippmex('signal', macchans, 'hifreq');
    end
    
    if isempty(analchans)
        enabled_analog = [];
    else
        enabled_analog = xippmex('signal',analchans,'30ksps');
    end

    %prioritize high resolution channels over raw for micro FE
    raw_mic = raw_mic&~hires_mic;
    
    if strcmp(mac_stream_type, 'hifreq') || isempty(mac_stream_type)
        %prioritize high freq channels over high res for macro FE
        hires_mac = hires_mac&~hifreq_mac;
    elseif strcmp(mac_stream_type, 'hires')
        %prioritize high res channels over high freq for macro FE
        hifreq_mac = hifreq_mac&~hires_mac;
    end
    
    micchans_raw = micchans(logical(raw_mic));
    micchans_hires  = micchans(logical(hires_mic));
    macchans_hires  = macchans(logical(hires_mac));
    macchans_hifreq = macchans(logical(hifreq_mac));
    analchans = analchans(logical(enabled_analog));
    ripple_info.mic_ch = micchans_raw;
    ripple_info.analog_chs = analchans;
    ripple_info.mic_ch_hires = micchans_hires;
    ripple_info.mac_ch_hires = macchans_hires;
    ripple_info.mac_ch_hifreq = macchans_hifreq;
    chs = [micchans_raw, micchans_hires, macchans_hires, macchans_hifreq, analchans]';
    sr = [30000 * ones(length(micchans_raw),1);...
        2000 * ones(length(micchans_hires),1);...
        2000 * ones(length(macchans_hires),1);...
        7500 * ones(length(macchans_hifreq),1);...
        30000 * ones(length(analchans),1) ];
    
    mic_lb = arrayfun(@(x) sprintf('micro %d',x),micchans_raw,'UniformOutput',false);
    mic_hires_lb = arrayfun(@(x) sprintf('micro_hr %d',x),micchans_hires,'UniformOutput',false);
    mac_hires_lb = arrayfun(@(x) sprintf('macro_hr %d',x),macchans_hires,'UniformOutput',false);
    mac_hifreq_lb = arrayfun(@(x) sprintf('macro_hf %d',x),macchans_hifreq,'UniformOutput',false);
    anal_lb = arrayfun(@(x) sprintf('analog %d',x),analchans,'UniformOutput',false);
    labels = [mic_lb, mic_hires_lb, mac_hires_lb, mac_hifreq_lb, anal_lb]'; %default labels
    [mapch_id,maplabels] = read_ripple_map(ripple_info.mapfile);
    if ~isempty(mapch_id)
        for i = 1:numel(mapch_id)
            ch2edit = find(chs==mapch_id(i));
            if ~isempty(ch2edit)
                labels{ch2edit}=maplabels{i};
            end
        end
    end
    
    info = struct;
    info.ch = chs;
    info.label = labels;
    info.sr = sr;
    
    %conversion fix por ripple
    info.conversion = [repmat(0.25,1,length(micchans_raw)),repmat(1,1,length(micchans_hires)),repmat(1,1,length(macchans_hires)),repmat(1,1,length(macchans_hifreq)),repmat(0.152562,1,length(anal_lb))];
    info.unit = [repmat({'uV'},1,length(micchans_raw)),repmat({'uV'},1,length(micchans_hires)),repmat({'uV'},1,length(macchans_hires)),repmat({'uV'},1,length(macchans_hifreq)),repmat({'mV'},1,length(anal_lb))];
    info.smpfilter = repmat({'None'},1,length(chs));
    info.parsr = arrayfun(@(x) ['f' num2str(x)], sr, 'UniformOutput', false);
    info.ismicro = [true(1,numel(micchans_raw)), false(1,numel(micchans_hires)), false(1,numel(macchans_hires)), false(1,numel(macchans_hifreq)) ,false(1,numel(analchans))];
end

function blk_info = enable_chs_blk(chs,parallel,ev_chs, blk_info)
    pause(0.01)
    instance = blk_info.instance;
    
    cbmex('mask',0,0,'instance',instance)
    for j = chs
        pause(0.02)
        cbmex('mask',j,1,'instance',instance)
    end

    pause(0.01)
    blk_info.enabled_chs = chs;
    blk_info.ev_chs = ev_chs;
    
    for i=1:length(ev_chs)
        cbmex('mask',ev_chs(i),1,'instance',instance); %digital input
        pause(0.05)
    end
    blk_info.parallel = parallel;
    if parallel
        cbmex('mask',blk_info.dig_ch,1,'instance',instance); %digital input
    end
    
    if ~isempty(ev_chs) || parallel
        cbmex('trialconfig',1,'instance',instance);
    else
        cbmex('trialconfig',1,'instance',instance,'noevent');
    end
end

function ripple_info = enable_chs_ripple(chs, parallel ,ev_chs, ripple_info)
    ripple_info.ev_chs = ev_chs;
    ripple_info.parallel = parallel;
    if parallel
        xippmex('digin', 'bit-change', 1);
    end
    available_chs = [ripple_info.mic_ch, ripple_info.mic_ch_hires, ripple_info.mac_ch_hires, ripple_info.mac_ch_hifreq, ripple_info.analog_chs];
    if ~all(ismember(chs,available_chs))
        error('No stream available for some channel.')
    end
    aux = ismember(chs,ripple_info.mic_ch);
    ripple_info.enabled_mic_ch = reshape(chs(aux),[1,sum(aux)]); %should be 1xN

    aux = ismember(chs,ripple_info.mic_ch_hires);
    ripple_info.enabled_mic_ch_hires = reshape(chs(aux),[1,sum(aux)]); %should be 1xN

    aux = ismember(chs,ripple_info.mac_ch_hires);
    ripple_info.enabled_mac_ch_hires = reshape(chs(aux),[1,sum(aux)]); %should be 1xN

    aux = ismember(chs,ripple_info.mac_ch_hifreq);
    ripple_info.enabled_mac_ch_hifreq = reshape(chs(aux),[1,sum(aux)]); %should be 1xN
    
    aux = ismember(chs,ripple_info.analog_chs);
    ripple_info.enabled_analog_ch = reshape(chs(aux),[1,sum(aux)]); %should be 1xN 
    ripple_info.chs = chs;
    ripple_info.sr = ones(numel(chs),1)*30000;
    ripple_info.sr(ismember(chs,ripple_info.enabled_mic_ch_hires))=2000;
    ripple_info.sr(ismember(chs,ripple_info.enabled_mac_ch_hires))=2000;
    ripple_info.sr(ismember(chs,ripple_info.enabled_mac_ch_hifreq))=7500;

    timezero = xippmex('time');
    ripple_info.prev_time = double(timezero)*ones(length(chs),1);
    ripple_info.mic_prev_time  = double(timezero);
    ripple_info.analog_prev_time = double(timezero);
end

function open_ripple(use_tcp)
    if use_tcp
        status = xippmex('tcp');%
    else
        status = xippmex();
    end
    if status ~= 1; error('Xippmex Did not Initialize.');  end
end

function string = smpfilter2str_blk(n)
switch n
    case 0
        string = 'None';
    case 1
        string = '750Hz High pass';
    case 2
        string = '250Hz High pass';
    case 3
        string = '100Hz High pass';
    case 4
        string = '50Hz Low pass';
    case 5
        string = '125Hz Low pass';
    case 6
        string = '250Hz Low pass';
    case 7
        string = '500Hz Low pass';
    case 8
        string = '150Hz Low pass';
    case 9
        string = '10Hz-250Hz Band pass';
    case 10
        string = '2.5kHz Low pass';
    case 11
        string = '2kHz Low pass';
    case 12
        string = '250Hz-5kHz Band pass';
    otherwise
        string='unknown';
end
end

function [ch_id,labels] = read_ripple_map(file)
    labels = {};
    ch_id = [];    
    if isempty(file)
        return
    end
    fid = fopen(file);

    if fid==-1
        return
    end
    line_ex = fgetl(fid);
    while ~(isnumeric(line_ex))
        if ~isempty(line_ex) && line_ex(1)~='#'
            labeline = regexp(line_ex,'^\d\.\S*\.\S*\.\d*;\s*\S*(?=\s*;)','match','once');
            if ~isempty(labeline)
                aux = regexp(labeline,'\s*;\s*','split');
                new_label = aux{2};
                aux = regexp(aux{1},'\.','split'); %unused.port.FEslot.FEch
                if length(aux{2})==1 % port=letter is a micro
                    ch = (double(aux{2})-double('A'))*128+(str2double(aux{3})-1)*32+str2double(aux{4});
                    labels{end+1} = sprintf('%s-%03d',new_label,ch);
                    ch_id(end+1) = ch;
                elseif strcmp(aux{2},'AIO') %only AIO matter this omits digital I/O
                        if strcmp(aux{3},'BNC')
                            labels{end+1} = sprintf('%s',new_label);
                            ch_id(end+1) = 10240+str2double(aux{4});
                        end     
                        if strcmp(aux{3},'AUD')
                            labels{end+1} = sprintf('%s',new_label);
                            ch_id(end+1) = 10268+str2double(aux{4});
                        end
    
                end
    
            end
        end
        line_ex = fgetl(fid);
    end
    fclose(fid);
end



function [info, curr_device] = get_chs_info_ncx(curr_device)
    load([curr_device.address filesep 'NSx.mat'], 'NSx','files');
    curr_device.NSx = NSx;
    curr_device.files_info = files;
    info = struct;
    info.ch = cell2mat({NSx.chan_ID}');
    info.label = {NSx.label};
    info.sr = cell2mat({NSx.sr}');
    info.conversion = cell2mat({NSx.conversion}');
    info.unit = {NSx.unit};
    info.smpfilter = repmat({'None'},1,length(info.ch));
    info.parsr = cellfun(@(x) ['f' num2str(x)], {NSx.sr}, 'UniformOutput', false);
    info.ismicro = cellfun(@(x) strcmp(x,'uV'), {NSx.unit}, 'UniformOutput', true) & cell2mat({NSx.sr})==30000;
end


function curr_device = enable_chs_ncx(chs, parallel ,ev_chs, curr_device)
    %checks channels
    available_channels = cell2mat({curr_device.NSx.chan_ID});
    chs_in_files = arrayfun(@(x) any(x==available_channels),chs);
    curr_device.chs = chs(chs_in_files);
    if any(~chs_in_files)
        warning('Channels not in files and have been removed');
    end
    %not implemented
    %checks that NEV exist... but it depends on the format
    if ~isempty(ev_chs) || parallel
        error('events and parallel not implemented for ncx');
    end
    curr_device.ev_chs = ev_chs;
    curr_device.parallel = parallel;
    curr_device.last_time = tic();
    curr_device.timestamp = 0;
    curr_device.files = zeros(1,numel(curr_device.chs));
    curr_device.enable_sr = zeros(1,numel(curr_device.chs));
    for i = 1:numel(curr_device.chs)
        chi = available_channels==curr_device.chs(i);
        label = [curr_device.address curr_device.NSx(chi).output_name curr_device.NSx(chi).ext];
        if ~exist(label,'file')
            error(['File: ' label 'not found']);
        end
        curr_device.files(i) = fopen(label,'r','l');
        curr_device.enable_sr(i) = curr_device.NSx(chi).sr;
    end
end


function [streams, curr_device] = get_stream_ncx(curr_device)
    readsec = toc(curr_device.last_time);
    curr_device.last_time = tic();
    curr_device.timestamp = curr_device.timestamp + readsec;
    
    timestamp = curr_device.last_time*30000;
    chs = curr_device.chs;
    if ~isempty(chs)
        
        streams.timestamp = zeros(length(chs),1);
        streams.data = cell(length(chs),1);
        streams.lost_prev = zeros(length(chs),1);
        streams.sr = zeros(length(chs),1);
        streams.lost_post = zeros(length(chs),1);

        for ci = 1:length(chs)
            samples = round(readsec* curr_device.enable_sr(ci));
            streams.data{ci} = fread(curr_device.files(ci),samples,'int16'); %output dimension must be Nx1
            streams.sr(ci) = curr_device.enable_sr(ci);
            streams.timestamps(ci) = timestamp;
        end
    end
    
    if curr_device.parallel
         error('not implemented');
    end
    
    if ~isempty(curr_device.ev_chs)
        error('not implemented');
    end
end

function close_ncx(curr_device)
    for ci = 1:length(curr_device.chs)
        if curr_device.files(ci)~=-1
            fclose(curr_device.files(ci)); %output dimension must be Nx1
        end
    end
end