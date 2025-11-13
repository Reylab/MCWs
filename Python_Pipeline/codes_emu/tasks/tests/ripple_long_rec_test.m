
addpath('/opt/Trellis/Tools/xippmex/')
xippmex('tcp');

%select one micro channel
micchans   = xippmex('elec','micro');
enabled_mic = xippmex('signal',micchans,'raw');
micchans = micchans(logical(enabled_mic));



time1  = xippmex('time');
pause(0.5)
[data, time2]  = xippmex('cont',micchans(1),5000,'raw',double(time1));


fprintf("%.2f hours\n",time1/30000/60/60)


[data, timew]  = xippmex('cont',micchans(1),5000,'raw');
pause(0.5)
[data, timew2]  = xippmex('cont',micchans(1),5000,'raw');
timew2-timew

xippmex('close');