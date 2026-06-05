function new_emu_num = increment_emu_num(patient_id, rec_metadata_path, varargin)
% INCREMENT_EMU_NUM Increment the EMU number for a new recording session
%
% This function is used when starting a new recording session/day
% to ensure the EMU number is incremented from the last session.
%
% Usage:
%   new_emu_num = increment_emu_num(patient_id, rec_metadata_path)
%   new_emu_num = increment_emu_num(patient_id, rec_metadata_path, 'set_emu', 5)
%
% Inputs:
%   patient_id        - Subject ID string (e.g., 'MCW-FH_001')
%   rec_metadata_path - Path to rec_metadata folder where CSV is stored
%
% Optional Parameters:
%   'set_emu'         - Explicitly set the EMU number instead of incrementing
%
% Outputs:
%   new_emu_num       - The new EMU number
%
% Author: ReyLab

    % Parse inputs
    p = inputParser;
    addRequired(p, 'patient_id', @ischar);
    addRequired(p, 'rec_metadata_path', @ischar);
    addParameter(p, 'set_emu', [], @isnumeric);
    parse(p, patient_id, rec_metadata_path, varargin{:});
    opts = p.Results;
    
    % Define CSV path
    csv_filename = sprintf('%s_Task_History.csv', patient_id);
    csv_path = fullfile(rec_metadata_path, csv_filename);
    
    if ~isfile(csv_path)
        % No history yet, start at 1
        if ~isempty(opts.set_emu)
            new_emu_num = opts.set_emu;
        else
            new_emu_num = 1;
        end
        fprintf('No task history found. Starting at EMU-%d\n', new_emu_num);
        return
    end
    
    % Read current CSV data
    try
        data = readtable(csv_path, 'TextType', 'string', 'Delimiter', ',');
    catch
        new_emu_num = 1;
        fprintf('Could not read task history. Starting at EMU-%d\n', new_emu_num);
        return
    end
    
    if height(data) == 0 || ~ismember('emu_num', data.Properties.VariableNames)
        if ~isempty(opts.set_emu)
            new_emu_num = opts.set_emu;
        else
            new_emu_num = 1;
        end
        fprintf('Empty task history. Starting at EMU-%d\n', new_emu_num);
        return
    end
    
    % Get the highest EMU number
    max_emu = max(data.emu_num(~isnan(data.emu_num)));
    if isempty(max_emu)
        max_emu = 0;
    end
    
    if ~isempty(opts.set_emu)
        new_emu_num = opts.set_emu;
    else
        new_emu_num = max_emu + 1;
    end
    
    fprintf('EMU number updated: %d -> %d\n', max_emu, new_emu_num);
end
