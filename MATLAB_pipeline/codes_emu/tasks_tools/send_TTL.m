function trigger_time = send_TTL(which_system,DAQ_param_1,DAQ_param_2, trigger_value) 
if strcmp(which_system,'RIP')
    DaqDOut(DAQ_param_1,0,trigger_value);
%     DaqDOut(DAQ_param_1,1,trigger_value);   
elseif strcmp(which_system,'BRK')
    DAQ_param_1.AddRequestS(DAQ_param_2, 'LJ_ioPUT_DIGITAL_PORT', 0, trigger_value, 8, 0);
    DAQ_param_1.GoOne(DAQ_param_2);
end
trigger_time = GetSecs;
