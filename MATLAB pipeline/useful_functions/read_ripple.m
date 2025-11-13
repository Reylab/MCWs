function read_ripple(filename,hFile,macro,channels, with_dig)
if ~exist('with_dig','var')|| isempty(with_dig), with_dig=false; end

ElectrodeIDs = double(cell2mat({hFile.Entity.ElectrodeID}));
Events_ii = find(ElectrodeIDs==0);

%% Digital Input
if with_dig
    N = hFile.Entity(Events_ii).Count; % CHECK WHAT HAPPENS WHEN length(Events_ii)>1 
    Event_Time = zeros(1,N); Event_Value = Event_Time;
    for jj = 1:N
        [ns_RESULT, TimeStamp, Data, DataSize] = ...
            ns_GetEventData(hFile, Events_ii, jj);
        Event_Time(jj) = TimeStamp; % in seconds
        Event_Value(jj) = Data; % data value (0 to 255)
    end
    save Events Event_Time Event_Value
end
%% Electrodes (Raw Data)
All_labels = {hFile.Entity.Label};

% raw_iis = find(startsWith(All_labels(setdiff(1:length(All_labels),Events_ii)),'raw'));
raw_iis = find(startsWith(All_labels,'raw'));

% if ~isempty(Events_ii)
%     raw_iis = raw_iis(raw_iis>Events_ii)+1; %check if Events_ii is not empty
% end
elecs = raw_iis(ismember(cell2mat({hFile.Entity(raw_iis).ElectrodeID}),channels));
%         outfile_handles = cell(1,nchan); %some will be empty
elecs = [elecs find(ElectrodeIDs>10240)];

freq_priority = [30000,2000,1000,500];

Data = struct;
NSx = struct;
outfile_handles = cell(1,numel(elecs));
try
    for ii=1:numel(elecs)
        [~, Data(ii).nsAnalogInfo] = ns_GetAnalogInfo(hFile, elecs(ii));
        [~, Data(ii).DataPoints, Data(ii).Data] = ns_GetAnalogData(hFile, elecs(ii), 1, hFile.Entity(elecs(ii)).Count,'unscale');
        
        NSx(ii).chan_ID = hFile.Entity(elecs(ii)).ElectrodeID;
        NSx(ii).conversion = hFile.Entity(elecs(ii)).Scale;
        NSx(ii).label = hFile.Entity(elecs(ii)).Label;
        NSx(ii).unit = hFile.Entity(elecs(ii)).Units(1:2);
        NSx(ii).electrode_ID = hFile.Entity(elecs(ii)).ElectrodeID;
        NSx(ii).nsp = [];
%         NSx(ii).which_system = 'RIP';
        NSx(ii).ext = '.NC5';
        NSx(ii).lts = hFile.Entity(elecs(ii)).Count;
        NSx(ii).filename = filename;
        NSx(ii).sr = 30000;
        if strcmp(NSx(ii).unit,'uV')
%             NSx(ii).macro = macro{ceil(ii/9)};
            NSx(ii).macro = macro{ceil(ii/8)};
        else
            NSx(ii).macro = hFile.Entity(elecs(ii)).Label;
        end
%         NSx(ii).output_name = sprintf('%s_%d',hFile.Entity(elecs(ii)).Label,hFile.Entity(elecs(ii)).ElectrodeID);
        NSx(ii).output_name = sprintf('%s_%d',NSx(ii).macro,hFile.Entity(elecs(ii)).ElectrodeID);
        pak_lost = Data(ii).Data<Data(ii).nsAnalogInfo.MinVal/Data(ii).nsAnalogInfo.Resolution;
        Data(ii).Data(pak_lost)=0;
        NSx(ii).num_pack_lost = sum(pak_lost);
        NSx(ii).samples_pack_lost = find(pak_lost);
        
        %     num_inds=regexp(Data(ii).nsAnalogInfo.ProbeInfo, '\d');
        %     outfile_handles{ii} = fopen([Data(ii).nsAnalogInfo.ProbeInfo(1:num_inds(end)) '.NC5'],'w');
        %     outfile_handles{ii} = fopen(sprintf('%s_%d.NC5',Data(ii).nsAnalogInfo.ProbeInfo(1:num_inds(end)),Data(ii).nsAnalogInfo.ProbeInfo),'w');
        outfile_handles{ii} = fopen(sprintf('%s.NC5',NSx(ii).output_name),'w');
        fwrite(outfile_handles{ii},Data(ii).Data,'int16');
        fclose(outfile_handles{ii});
    end
catch ME
    ii
    rethrow(ME)
end
Entity = hFile.Entity;
files = [];
save Entity Entity
save NSx NSx freq_priority files

% % least significant bit and Parallel Digital Input
% figure
% plot((0:(AnalogInput(3).DataPoints-1))/AnalogInput(3).nsAnalogInfo.SampleRate,AnalogInput(3).Data*AnalogInput(3).nsAnalogInfo.Resolution)
% hold on
% plot(Event_Time,mod(Event_Value,2)*2500,'xr')
% ylabel(AnalogInput(3).nsAnalogInfo.Units  ); xlabel('Time (s)');
% title('least significant bit as an analog input and Parallel Digital Input events (rx)')


% figure
% ii = 1;
% plot((0:(Data(ii).DataPoints-1))/Data(ii).nsAnalogInfo.SampleRate,Data(ii).Data*Data(ii).nsAnalogInfo.Resolution  )
% ylabel(Data(ii).nsAnalogInfo.Units  ); xlabel('Time (s)');
% title(sprintf('Channel %s',Data(ii).nsAnalogInfo.ProbeInfo  ))

% f1 = fopen(sprintf('%s%s',NSx(posch).output_name,NSx(posch).ext),'r','l');
% fseek(f1,(min_record-1)*2,'bof');
% Samples = fread(f1,(max_record-min_record+1),'int16=>double')*NSx(posch).conversion;
% fclose(f1);