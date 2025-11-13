%this function handle U3 (labjack.m and exodrive libraty) and USB-1208FS (DAQ in pychtoolbox)
%device types: LJ for U3 and MC for USB-1208FS

%device_names: MC and LJ
classdef TTL_device < handle
	
	properties
        device_name = ''
        info = struct
        initialized = false
    end
    methods
        function obj=TTL_device(device_name)
            %set all 16 pins as outputs and value 0
            if obj.initialized
                if strcmp(device_name, obj.device_name)
                    error('Trying to initialize other device with the same object.')
                end
                warning('device already initialized')
            end
            if strcmp(device_name, 'MC')
                obj.info.DAQ_index = DaqDeviceIndex; % get a handle for the USB-1208FS
                DaqDConfigPort(obj.info.DAQ_index,0,0); % configure digital port A for output
                DaqDConfigPort(obj.info.DAQ_index,1,0); % configure digital port B for output
                DaqDOut(obj.info.DAQ_index,0,0);
                DaqDOut(obj.info.DAQ_index,1,0);  
            
            elseif strcmp(device_name, 'LJ')
                if IsLinux
                    warning off
                    obj.info = labJack('verbose',false);
                    warning on
                    if ~obj.info.validHandle()
                        error('Unable to connect to labjack device')
                    end
                    obj.info.setDIODirection([255,255,0],[255,255,0])
                    obj.info.setDIOValue([0,0,0],[255,255,0]);
                elseif IsWindows
                    NET.addAssembly('LJUDDotNet');
                    ljudObj = LabJack.LabJackUD.LJUD;
                    [~, ljhandle] = ljudObj.OpenLabJackS('LJ_dtU3', 'LJ_ctUSB', '0', true, 0);
                    ljudObj.ePutS(ljhandle, 'LJ_ioPIN_CONFIGURATION_RESET', 0, 0, 0);
                    ljudObj.ePutS(ljhandle, 'LJ_ioPUT_ANALOG_ENABLE_PORT', 0, 0, 8);
                    ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, 0, 8, 0);
                    ljudObj.GoOne(ljhandle);
                    obj.info.ljudObj = ljudObj;
                    obj.info.ljhandle = ljhandle;
                else
                    error('Device not supported on mac')
                end
            else
                error('Device name not supported try: MC or LJ')
            end
            obj.initialized = true;
            obj.device_name = device_name;
        end

        function trigger_time = send(obj, value)
            if ~obj.initialized
                error('Initialize device first.')
            end
            if strcmp(obj.device_name, 'MC')
                DaqDOut(obj.info.DAQ_index,0,value);
                DaqDOut(obj.info.DAQ_index,1,value);  
            
            else %strcmp(obj.device_name, 'LJ')
                if IsLinux
                    obj.info.setDIOValue([value,value,0],[255,255,0]);
                elseif IsWindows
                    obj.info.ljudObj.AddRequestS(obj.info.ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, value, 8, 0);
                    obj.info.ljudObj.GoOne(obj.info.ljhandle);
                end
            end
            trigger_time = GetSecs;
        end
        function close(obj)
            if obj.initialized
                if strcmp(obj.device_name, 'LJ') && IsLinux
                    obj.info.close()
                    obj.info = [];
                end
            end
            obj.initialized = false;
        end
    end
end