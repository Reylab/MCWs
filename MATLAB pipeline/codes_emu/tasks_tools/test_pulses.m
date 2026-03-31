
clear all
clear all
clear all

% device_name = 'MC';
device_name = 'LJ';


device = TTL_device(device_name);
values = [1 2 5 8 17 32 65 128 85 84 4 16 0];
%%
for i=1:length(values)
    device.send(values(i));
    pause(0.4)
end
device.close()