function collect_class_comparisons(channel, classes)
% collect_class_comparisons(channel, classes)
% 
% Inputs:
%   channel : numeric or string, the channel number (used for naming the output folder)
%   classes : numeric array or cell array of strings/numbers of the classes to search for (e.g., [1, 2] or {'1', '2'})

    % Convert channel to string
    if isnumeric(channel)
        chan_str = num2str(channel(1));
    else
        chan_str = channel;
    end
    
    % Convert classes to a cell array of strings for easier processing
    if isnumeric(classes)
        class_strs = arrayfun(@num2str, classes, 'UniformOutput', false);
    elseif ischar(classes)
        class_strs = {classes};
    else
        class_strs = classes;
    end
    
    % Define the target folders to search (matching batch_clust_resp.m)
    orig_clusters = {'sdnum_1', 'sdnum_3', 'sdnum_1_t_3'};
    algo_bases = {'algo1', 'algo2', 'algo3', 'algo4', 'algo5'};
    sd_bases = {'sd1', 'sd3', 'sd3_t_1'};
    
    algo_folders = orig_clusters;
    for i = 1:length(algo_bases)
        for j = 1:length(sd_bases)
            algo_folders{end+1} = sprintf('%s_strt_%s', algo_bases{i}, sd_bases{j});
        end
    end
    
    % Create the output directory based on channel and classes
    classes_joined = strjoin(class_strs, '_');
    comp_folder = fullfile(pwd, sprintf('Comparison_Chan_%s_Classes_%s', chan_str, classes_joined));
    
    if ~exist(comp_folder, 'dir')
        mkdir(comp_folder);
    end
    
    fprintf('\n=== Collecting comparison images for Channel %s, Classes: %s ===\n', chan_str, strjoin(class_strs, ', '));
    
    base_dir = pwd;
    valid_exts = {'.png', '.fig', '.jpg', '.jpeg', '.pdf', '.tif', '.bmp', '.emf'};
    
    % Regex pattern to strictly match the standalone channel number in the filename
    % (^|\D) means the start of the string OR a non-digit character
    % (\D|$) means a non-digit character OR the end of the string
    chan_pattern = sprintf('(^|\\D)%s(\\D|$)', chan_str);
    
    % Iterate over all defined algorithm folders
    for i = 1:length(algo_folders)
        algo_name = algo_folders{i};
        
        if ~exist(fullfile(base_dir, algo_name), 'dir')
            continue; % Skip if the algorithm folder doesn't exist in the current directory
        end
        
        % Search for each requested class
        for c = 1:length(class_strs)
            cls = class_strs{c};
            % Find everything with the class number
            search_pattern = fullfile(base_dir, algo_name, '**', sprintf('*class%s*', cls));
            found_images = dir(search_pattern);
            
            for f = 1:length(found_images)
                % Filter out folders
                if found_images(f).isdir
                    continue;
                end
                
                [~, name, ext] = fileparts(found_images(f).name);
                
                % Must be a valid image extension and ideally from a grapes folder
                if ~any(strcmpi(ext, valid_exts)) || ~contains(lower(found_images(f).folder), 'grapes')
                    continue;
                end
                
                original_filename = found_images(f).name;
                
                % =========================================================
                % EXACT CHANNEL FILTRATION CHECK
                % =========================================================
                % Ensure the specific channel number is actually in the filename.
                if isempty(regexp(original_filename, chan_pattern, 'once'))
                    continue; % Skip, it belongs to another channel (e.g. 322, 368, etc.)
                end
                
                src_file = fullfile(found_images(f).folder, found_images(f).name);
                
                % Logic to handle prefix
                % Check if the file starts with ANY of the known algorithm names
                has_any_algo_prefix = false;
                prefix_to_remove = '';
                for k = 1:length(algo_folders)
                    if startsWith(original_filename, [algo_folders{k} '_'])
                        has_any_algo_prefix = true;
                        prefix_to_remove = [algo_folders{k} '_'];
                        break;
                    end
                end
                
                if has_any_algo_prefix
                    if startsWith(original_filename, [algo_name '_'])
                        % Prefix is already correct
                        dest_name = original_filename;
                    else
                        % Prefix exists but is inaccurate (from a different algo), replace it
                        base_name_no_prefix = extractAfter(original_filename, prefix_to_remove);
                        dest_name = sprintf('%s_%s', algo_name, base_name_no_prefix);
                    end
                else
                    % No algorithm prefix exists, add it
                    dest_name = sprintf('%s_%s', algo_name, original_filename);
                end
                
                % Copy to the consolidated comparison folder
                copyfile(src_file, fullfile(comp_folder, dest_name));
            end
        end
    end
    
    fprintf('Images successfully copied to: %s\n', comp_folder);
end