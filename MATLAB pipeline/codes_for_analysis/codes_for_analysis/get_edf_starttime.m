function [Date_Time,End_Time] = get_edf_starttime(filename)

[hdr] = open_edf(filename,'asint16',1);

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

Date_Time = datetime(Start_Date + timeofday(Start_Time));

Duration = hdr.duration * hdr.records;
End_Time = Date_Time + seconds(Duration);



