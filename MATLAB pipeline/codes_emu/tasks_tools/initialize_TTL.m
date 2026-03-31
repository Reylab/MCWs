function [DAQ_param_1,DAQ_param_2] = initialize_TTL(which_system)
if strcmp(which_system,'RIP')
    DAQ_param_1=DaqDeviceIndex; % get a handle for the USB-1208FS
    DaqDConfigPort(DAQ_param_1,0,0); % configure digital port A for output
%     DaqDConfigPort(out1,1,0); % configure digital port B for output
    DAQ_param_2=[];
elseif strcmp(which_system,'BRK')
    NET.addAssembly('LJUDDotNet');
    DAQ_param_1 = LabJack.LabJackUD.LJUD;
    [~, DAQ_param_2] = DAQ_param_1.OpenLabJackS('LJ_dtU3', 'LJ_ctUSB', '0', true, 0);
    DAQ_param_1.ePutS(DAQ_param_2, 'LJ_ioPIN_CONFIGURATION_RESET', 0, 0, 0);
    DAQ_param_1.ePutS(DAQ_param_2, 'LJ_ioPUT_ANALOG_ENABLE_PORT', 0, 0, 8);
end
send_TTL(which_system,DAQ_param_1,DAQ_param_2, 0); 