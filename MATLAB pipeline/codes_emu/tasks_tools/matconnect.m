classdef matconnect < handle
   properties
      udp_connect
      fail = false;
%       synchronized = false;
%       sinc_msg_size
      in_port
      ip
      out_port
      use_udpport
   end
   methods
      function obj = matconnect(other_pc) %one matlab shoud use number 1 and other number 2
         if verLessThan('matlab','9.13')
            instrreset %reset all connections
         else
            clear obj.udp_connect
         end
         
         if isnumeric(other_pc)
             obj.ip ='127.0.0.1';
             switch other_pc
                 case 1
                     obj.in_port = 2011;
                     obj.out_port = 2012;
                 case 2
                     obj.in_port = 2012;
                     obj.out_port = 2011;
                 otherwise
                     error('invalid number')
             end
         else
             obj.in_port = 2011;
             obj.out_port = 2011;
             isip = all(arrayfun(@(x) x=='.' || isstrprop(x,'digit'), other_pc));
             if isip
                 obj.ip = other_pc;
             elseif isunix
                 [~, ipstr]=system(['nmblookup ' other_pc]);
                 obj.ip = ipstr(1: regexp(ipstr,' ', 'once')-1);
             else
                 error('invalid ip or hostname not implemented for this system')
             end
         end
         if verLessThan('matlab','9.9')
            obj.udp_connect = udp(obj.ip, obj.out_port, 'LocalPort', obj.in_port);
             fopen(obj.udp_connect);
             obj.use_udpport = false;
             try
                 if strcmp(obj.udp_connect.Status, 'open')
                    fprintf('Connected to: %s via port: %d', obj.ip, obj.in_port);
                 end
             catch ME
                 errMsg = getReport(ME);
                 disp(errMsg)
                 fprintf('\nFailed to connect to: %s via port: %d.\n', obj.ip, obj.in_port);
             end
         else
             obj.use_udpport = true;
             obj.udp_connect = udpport('byte', 'IPV4','LocalPort', obj.in_port);
             try
                 fprintf('\nConnected to: %s via port: %d.\n', obj.udp_connect.LocalHost, obj.udp_connect.LocalPort);
             catch ME
                 errMsg = getReport(ME);
                 disp(errMsg)
                 fprintf('\nFailed to connect to: %s via port: %d.\n', obj.ip, obj.in_port);
             end
         end

         
      end
      function flag=msg_available(obj)
          if obj.use_udpport
              flag = obj.udp_connect.NumBytesAvailable>0;
          else
              flag = obj.udp_connect.BytesAvailable>0;
          end
      end
      
      function close(obj)
          if ~obj.use_udpport
              fclose(obj.udp_connect);
          end
          clear obj.udp_connect;
      end
      
      function send(obj,message)
          if  obj.use_udpport
            writeline(obj.udp_connect,message,obj.ip,obj.out_port);
          else
            fprintf(obj.udp_connect, '%s\n', message);
          end
      end
      
      function msg=read(obj)
          if  obj.use_udpport
            msg = char(readline(obj.udp_connect));
          else
             msg = fgetl(obj.udp_connect);
          end
      end 
      
      
      function [message, time_wait] = waitmessage(obj,timeout)
          if nargin == 1
            timeout = Inf;
          end
          if obj.fail
              message=[];
              return
          end
          start_time = tic();
          while(obj.udp_connect.BytesAvailable==0 && toc(start_time)<timeout)
              pause(0.01)
          end
          time_wait = toc(start_time);
          if obj.msg_available()
              message=obj.read();
          else
              obj.fail = true;
              warning('timeout on matlab communication, it will be disabled.')
              message=[];
          end
      end
      
      function [message, time_wait] = waitmessage_nofail(obj,timeout)
          if nargin == 1
            timeout = Inf;
          end
          if obj.fail
              message=[];
              return
          end
          start_time = tic();
          while(obj.udp_connect.BytesAvailable==0 && toc(start_time)<timeout)
              pause(0.01)
          end
          time_wait = toc(start_time);
          if obj.msg_available()
              message=obj.read();
          else
              message=[];
          end
      end
      
      function [message, twait] = waitmessage_or_key(obj,key,dev_used)
            if nargin==1
                key = 'F2';
            end
            twait = tic;
            firstPress=zeros(1,KbName(key));
            message = [];
            pressed = false;
            while ~(obj.msg_available()) && ~(pressed && firstPress(KbName(key)))
                [pressed,firstPress,~,~] = multiKbQueueCheck(dev_used);
                pause(0.01)
            end
            for d= dev_used; KbQueueFlush(d);   end
            
            
            
            if obj.msg_available()
                message=obj.read();
            end
            twait =toc(twait);
        end
   end
end