function signal_file_ready(data_file)
    % SIGNAL_FILE_READY Creates a .ready file to signal that data_file has been written
    %
    % This function creates a companion .ready file that indicates the data file
    % has been completely written and is safe to read. Use in combination with
    % wait_for_file_ready() for explicit synchronization between MATLAB instances.
    %
    % Usage:
    %   signal_file_ready(data_file)
    %
    % Input:
    %   data_file - Path to the data file that has been written
    %
    % The ready file will contain:
    %   - Timestamp of when the signal was created
    %   - Original file name for reference
    %
    % Example (Writer side):
    %   safe_save_mat('experiment.mat', '-struct', data);
    %   signal_file_ready('experiment.mat');
    %
    % Example (Reader side):
    %   if wait_for_file_ready('experiment.mat', 60)
    %       [data, ok] = safe_load_mat('experiment.mat');
    %   end
    %
    % See also: wait_for_file_ready, safe_save_mat
    
    if nargin < 1
        error('signal_file_ready: data_file argument is required');
    end
    
    ready_file = [data_file '.ready'];
    
    try
        fid = fopen(ready_file, 'w');
        if fid == -1
            error('Could not create ready file: %s', ready_file);
        end
        
        fprintf(fid, 'file: %s\n', data_file);
        fprintf(fid, 'timestamp: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        fprintf(fid, 'datenum: %f\n', now);
        fclose(fid);
        
    catch ME
        warning('signal_file_ready: Failed to create ready file: %s', ME.message);
        if exist('fid', 'var') && fid ~= -1
            try
                fclose(fid);
            catch
            end
        end
        rethrow(ME);
    end
end
