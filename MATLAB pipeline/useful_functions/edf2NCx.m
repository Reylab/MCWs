function edf2NCx(filename, channels)
%edf2NCx parse edf files to NCx binart channels with a NSx emtadata file
%   For empty arguments edf2NCx() will find the edf file in the current 
%   folder and parse all its channels
%
%   edf2NC4(filename, channels) parse only the channels channels of the
%   given edf file of name filename.

    if ~exist('filename','var') || isempty(filename)
        filename = ls('*edf');
    end    
    
    [hdr, recorddata] = open_edf(filename,'asint16',1);

    %%
    %Date and time 
startdate = hdr.startdate;
starttime = hdr.starttime;

% month = str2double(startdate(4:5));
% day = str2double(startdate(1:2));
% year = str2double(startdate(7:10));


% Start_Time = sprintf('%d/%d/%d %s', month,day,year,starttime);
Start_Date = datetime(startdate, 'InputFormat', 'dd.MM.yy');
Start_Time = datetime(starttime, 'InputFormat', 'HH.mm.ss');

DateTime = datetime(Start_Date + timeofday(Start_Time));

Duration = hdr.duration * hdr.records;
EndTime = DateTime + seconds(Duration);

Date_Time = [DateTime;EndTime];

%% 

    annotation_ix = find(cellfun(@(x) ~isempty(x), regexpi(hdr.label,'((comment)|(annotation))','once')));
    
    
    if numel(annotation_ix)>1
        error('Conflics with labels and annotations, check the regular expression above this line.');
    end
    
    %parse commments
    if ~isempty(annotation_ix)
        ann = recorddata(annotation_ix,:);
        
        words = {};
        reading = 0;
        s_word = 0;
        for i = 1: length(ann)
            if ann(i)>0
                if ~reading
                    s_word = i;
                    reading = 1;
                end
            else
                if reading
                    s=typecast(ann(s_word:i),'int8');
                    comment=strtrim(char(s(s>0)));
                    %this line removes all the comments with only one set 
                    %of numbers in NK
                    if length(comment)>14
                        words{end+1} = comment;
                    end
                    reading = 0;
                end
            end
        end
        T = cell2table(words','VariableNames',{'strings'});
        writetable(T,'annotations.csv')
    end
    
    if ~exist('channels','var') || isempty(channels)
        channels = setdiff(1:size(recorddata,1), annotation_ix);
    end
   
    %% 
    
    nchan = numel(channels);
    % min_ch = min(recorddata(channels,:),[],2); %-12.8mV??
    % max_ch = max(recorddata(channels,:),[],2); %+12.8mV??
    % figure(1)
    % plot(min_ch/1000)
    % hold on
    % plot(max_ch/1000)
    % line(xlim,[12.8 12.8],'linestyle','--','color','k')
    % line(xlim,[-12.8 -12.8],'linestyle','--','color','k')
    % xlabel('Channel index')
    % ylabel('Voltage (mV)')
    % legend('minimum voltage','maximum voltage')
    % find(max_ch>8000) {'POLLT1aT12','POLLT2bHb2','POLLT310','POLLP1bC1','POLLOa2','POLLOb2','POLLOb3','EEGPZRef'}
    % IDs=[1:124 129:192];



    try
        for i=1:nchan
            NSx(i).chan_ID = i;
            scalefac = (hdr.physicalMax(i) - hdr.physicalMin(i))./(hdr.digitalMax(i) - hdr.digitalMin(i));
            dc = hdr.physicalMax(i) - scalefac .* hdr.digitalMax(i);
            NSx(i).conversion = scalefac;
            NSx(i).dc = dc;
            NSx(i).label = hdr.label{channels(i)};
            NSx(i).unit = hdr.units{channels(i)};
            NSx(i).electrode_ID = i;
            NSx(i).nsp = [];            
            NSx(i).lts = hdr.samples(channels(i))*hdr.records  ;
            NSx(i).filename = filename;
            NSx(i).sr = round(hdr.frequency(channels(i)));
            switch NSx(i).sr
                case 10000
                    NSx(i).ext = '.NC4';
                case 2000
                    NSx(i).ext = '.NC3';
                case 1000
                    NSx(i).ext = '.NC2';
                case 500
                    NSx(i).ext = '.NC1';
            end
            NSx(i).output_name = sprintf('%s_%d',hdr.label{channels(i)},i);

            tt = regexp(NSx(i).label,'\d+','once');
            if ~isempty(tt)
                NSx(i).bundle = NSx(i).label(1:tt-1);
            else
                NSx(i).bundle = NSx(i).label;
            end
%             tt = regexp(hdr.label{channels(i)},'\D');   
%             NSx(i).macro = hdr.label{channels(i)}(1:tt(end));
            NSx(i).is_micro = false;
            outfile_handles = fopen([NSx(i).output_name NSx(i).ext],'w');
            fwrite(outfile_handles,recorddata(channels(i),:),'int16');
            fclose(outfile_handles);
        end
    catch ME
        i
        rethrow(ME)
    end
    freq_priority = [30000,2000,10000,1000,500];
    files = [];
    save NSx NSx files freq_priority Date_Time