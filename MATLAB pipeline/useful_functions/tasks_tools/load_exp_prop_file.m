% function [exp_prop_file_data] = load_exp_prop_file(exp_prop_file, varNames, allVars)
% 
%     try
%         % Try loading the requested variables from the MAT file
%         exp_prop_file_data = load(exp_prop_file, varNames{:});
%         
%         % If successful, return the loaded data
%         disp('File loaded successfully');
%         
%     catch ME
%         % If loading fails, just recreate the MAT file from scratch
%         fprintf(2, 'Error loading MAT file (%s). Recreating file.\n', ME.message);
% 
%         % Recreate fresh MAT file with the current structure
%         save(exp_prop_file, '-struct', 'allVars', varNames{:});
%         disp('Recreated fresh MAT file successfully.');
% 
%         % Return the updated structure (i.e., the allVars) after recreation
%         exp_prop_file_data = allVars;
%     end
% end

function load_exp_prop_file(exp_prop_file, varNames, allVarsFull)
    try
        % Try loading the requested variables from the MAT file
        data = load(exp_prop_file, varNames{:});
        
        % Assign each loaded variable to the caller workspace
        for k = 1:numel(varNames)
            if isfield(data, varNames{k})
                assignin('caller', varNames{k}, data.(varNames{k}));
            else
                % If variable not present in file, assign empty
                assignin('caller', varNames{k}, []);
            end
        end
        
        %disp('File loaded successfully');
        
    catch ME
        fprintf("\n");
        fprintf(2, 'Error loading MAT file (%s). Recreating file.\n', ME.message);

        % Recreate fresh MAT file with the current structure
        save(exp_prop_file, '-struct', 'allVarsFull', varNames{:});
        fprintf("\n");
        disp('Recreated fresh MAT file successfully.');

        % Assign all variables from allVarsFull to the caller workspace
        for k = 1:numel(varNames)
            if isfield(allVarsFull, varNames{k})
                assignin('caller', varNames{k}, allVarsFull.(varNames{k}));
            else
                assignin('caller', varNames{k}, []);
            end
        end
    end
end

