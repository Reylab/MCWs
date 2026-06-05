classdef file_lock < handle
    % FILE_LOCK Simple file-based locking mechanism for MATLAB
    %
    % This class provides a simple file-based locking mechanism to prevent
    % race conditions when multiple MATLAB instances access the same file.
    %
    % Usage:
    %   lock = file_lock('myfile.mat');
    %   if lock.acquire()
    %       % Do file operations
    %       lock.release();
    %   end
    %
    % The lock is automatically released when the object is destroyed.
    
    properties
        lockfile
        timeout
        is_locked = false
    end
    
    methods
        function obj = file_lock(target_file, timeout_sec)
            % FILE_LOCK Constructor
            %   lock = file_lock(target_file) creates a lock for target_file
            %   lock = file_lock(target_file, timeout_sec) specifies timeout
            
            if nargin < 2
                timeout_sec = 30;
            end
            obj.lockfile = [target_file '.lock'];
            obj.timeout = timeout_sec;
        end
        
        function success = acquire(obj)
            % ACQUIRE Attempt to acquire the lock
            %   Returns true if lock was acquired, false if timeout
            
            tStart = tic;
            
            % Wait for any existing lock to be released
            while exist(obj.lockfile, 'file')
                % Check if lock is stale (older than 5 minutes)
                try
                    lockInfo = dir(obj.lockfile);
                    if ~isempty(lockInfo)
                        lockAge = (now - lockInfo.datenum) * 24 * 60; % age in minutes
                        if lockAge > 5
                            warning('Removing stale lock file: %s (age: %.1f min)', obj.lockfile, lockAge);
                            delete(obj.lockfile);
                            break;
                        end
                    end
                catch
                    % Ignore errors checking lock age
                end
                
                if toc(tStart) > obj.timeout
                    warning('Lock timeout exceeded for %s', obj.lockfile);
                    success = false;
                    return;
                end
                pause(0.1);
            end
            
            % Create lock file with timestamp and process info
            try
                fid = fopen(obj.lockfile, 'w');
                if fid == -1
                    warning('Could not create lock file: %s', obj.lockfile);
                    success = false;
                    return;
                end
                fprintf(fid, 'locked_by: MATLAB\n');
                fprintf(fid, 'timestamp: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
                fprintf(fid, 'datenum: %f\n', now);
                fclose(fid);
                obj.is_locked = true;
                success = true;
            catch ME
                warning('Error creating lock file: %s', ME.message);
                success = false;
            end
        end
        
        function release(obj)
            % RELEASE Release the lock
            
            if obj.is_locked && exist(obj.lockfile, 'file')
                try
                    delete(obj.lockfile);
                catch
                    warning('Could not delete lock file: %s', obj.lockfile);
                end
                obj.is_locked = false;
            end
        end
        
        function delete(obj)
            % DELETE Destructor - automatically releases lock
            obj.release();
        end
    end
end
