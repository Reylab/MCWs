%this class handle recording start, stop and comments for ripple (not for comments)
% and blackrock. It requires an structure parameters as the input
%device_names: RIP BRK
classdef recording_handler < handle
	
	properties
        params
        status
        rec_name
        onlineNSP
    end
    methods
        function obj=recording_handler(params, fname) %prepare recording
            obj.status = 'stopped';
            obj.params = params;
            if ~exist('fname','var')
                fname = '';
            end
            obj.rec_name = [fname '_' params.system];

            if strcmp(params.system,'RIP')
                try
                    xippmex('tcp');
                catch
                    xippmex('close');
                    xippmex('tcp');
                end
                xippmex('addoper', 129);
                try 
                    trellis = xippmex('trial','stopped',fullfile(obj.params.trellis_data_path, obj.rec_name));
                catch %catch the error about the existing filename, but that means, it's nor recording
                    trellis = xippmex('trial','stopped',fullfile(obj.params.trellis_data_path,  obj.rec_name));
                end     
                if strcmp(trellis.status, 'recording')
                   error('Unable to control Trellis because it is recording already, Check the "Enable Remote Control" checkbox.')
                end
            else
               %StopBlackrockAquisition(obj.rec_name,obj.onlineNSP);
            end
        end

        function check_status(obj)
            warning off
            trellis = xippmex('trial');
            warning on
            obj.status = trellis.status;
        end

        function start(obj)
            if ~strcmp(obj.status,'stopped')
                error('recording already started');
            end
            
            if strcmp(obj.params.system, 'RIP')
                pause(2)
                trellis = xippmex('trial','recording',fullfile(obj.params.trellis_data_path, obj.rec_name),0); %the last zero is to disable the auto stop
                if strcmp(trellis.status, 'stopped')
                    error('Unable to control Trellis, Check the "Enable Remote Control" checkbox.')
                end
            else
                [obj.onlineNSP, ~, ~] = StartBlackrockAquisition_noAutoInc(obj.rec_name,0);
            end

            obj.status = 'running';
            
        end
        function stop_and_close(obj)
            if strcmp(obj.status,'stopped')
                return
            end
            if strcmp(obj.params.system,'RIP')
                try
                   xippmex('trial','stopped');
                end
                xippmex('close');
            else
                StopBlackrockAquisition(obj.rec_name,obj.onlineNSP);
            end
            obj.status = 'stopped';
        end
        
        function close(obj)
            if ~strcmp(obj.status,'stopped')
                obj.stop_with_connection_check()
            end
            if strcmp(obj.params.system,'RIP')
                xippmex('close');
%             else
%
            end
        end
        function stop_with_connection_check(obj)
            if strcmp(obj.status,'stopped')
                return
            end
            if strcmp(obj.params.system,'RIP')
                try
                   xippmex('trial','stopped');
                catch ME
                    if strcmp(ME.message, 'no response')
                        xippmex('close');
                        xippmex('tcp');
                        xippmex('addoper', 129);
                        try
                            xippmex('trial','stopped');
                        end
                    end
                end
            else
                StopBlackrockAquisition_with_connection_check(obj.rec_name,obj.onlineNSP,0,[]);
            end
            obj.status = 'stopped';
        end
    end
end