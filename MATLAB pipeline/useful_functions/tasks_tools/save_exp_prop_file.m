% function save_exp_prop_file(exp_prop_file, subscr_exp_prop_file, varNames, allVars)
% 
%     try
%         % Save to exp_prop_file
%         save(exp_prop_file, '-struct', 'allVars', varNames{:}, '-append');
% 
%         % Save to subscr_exp_prop_file if provided
%         if ~isempty(subscr_exp_prop_file)
%             save(subscr_exp_prop_file, '-struct', 'allVars', varNames{:}, '-append');
%         end
% 
%     catch ME
%         %fprintf(2, 'Warning: Save failed (%s). Recreating fresh file.\n', ME.message);
% 
%         % Recreate fresh MAT file
%         save(exp_prop_file, '-struct', 'allVars', varNames{:});
%         
%         if ~isempty(subscr_exp_prop_file)
%             save(subscr_exp_prop_file, '-struct', 'allVars', varNames{:});
%         end
%     end
% end

function save_exp_prop_file(exp_prop_file, subscr_exp_prop_file, varNames, allVars)

    maxRetries = 5;
    retryCount = 0;
    success = false;

    while ~success && retryCount < maxRetries
        retryCount = retryCount + 1;
        try
            % Save to exp_prop_file
            save(exp_prop_file, '-struct', 'allVars', varNames{:}, '-append');

            % Save to subscr_exp_prop_file if provided
            if ~isempty(subscr_exp_prop_file)
                save(subscr_exp_prop_file, '-struct', 'allVars', varNames{:}, '-append');
            end

            % Check for corruption
            corrupted = false;
            try
                %tmp = load(exp_prop_file, varNames{:});
                tmp = load(exp_prop_file);
                clear tmp;
            catch
                corrupted = true;
                fprintf("\n");
                fprintf(2, 'Warning: %s is corrupted. Attempting to restore.\n', exp_prop_file);
                % Recreate fresh MAT file
                save(exp_prop_file, '-struct', 'allVars', varNames{:});
            end

            if ~isempty(subscr_exp_prop_file)
                try
                    %tmp = load(subscr_exp_prop_file, varNames{:});
                    tmp = load(subscr_exp_prop_file);
                    clear tmp;
                catch
                    corrupted = true;
                    fprintf("\n");
                    fprintf(2, 'Warning: %s is corrupted. Attempting to restore.\n', subscr_exp_prop_file);
                    save(subscr_exp_prop_file, '-struct', 'allVars', varNames{:});
                end
            end

            if ~corrupted
                success = true; % Save succeeded without corruption
            end

        catch ME
            fprintf("\n");
            %fprintf(2, 'Warning: Save attempt %d failed (%s). Retrying...\n', retryCount, ME.message);
            % Try recreating files in case of general save failure
            try
                save(exp_prop_file, '-struct', 'allVars', varNames{:});
                if ~isempty(subscr_exp_prop_file)
                    save(subscr_exp_prop_file, '-struct', 'allVars', varNames{:});
                end
            catch innerME
                fprintf("\n");
                fprintf(2, 'Warning: Recreate attempt failed (%s).\n', innerME.message);
            end

            pause(0.1); % small pause before retry
        end
    end

    if ~success
        fprintf("\n");
        error('Failed to save %s and/or %s after %d attempts.', exp_prop_file, subscr_exp_prop_file, maxRetries);
    end
end
