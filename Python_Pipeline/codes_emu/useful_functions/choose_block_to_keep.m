function [pics_to_keep, pics_removed] = choose_block_to_keep(Npics, Nrep, ISI, nxt_scr, pics_not_to_be_shown, selected2explore_cell, same_units_cell, same_categories_cell,  others)
    
    % options
    opt_idx = 1;
    dict_pics = {};
    pics_to_keep = [];
    pics_removed = [];
    all_selected_to_exp = cell2mat(selected2explore_cell');
    for blk=1:length(selected2explore_cell)
        pics = pics_not_to_be_shown(ismember(pics_not_to_be_shown, selected2explore_cell{blk}));
        num_pics = numel(pics);
        extra_time = num_pics * Nrep * ISI;
        options{opt_idx} = sprintf("Keep %d selected2explore from block %d (will add %2.2f seconds)", ...
                                            num_pics, blk, extra_time);
        dict_pics{opt_idx} = pics;
        opt_idx = opt_idx + 1;
    end
    for blk=1:length(same_units_cell)
        same_units_blk = setdiff(same_units_cell{blk}, all_selected_to_exp, 'stable');
        pics = pics_not_to_be_shown(ismember(pics_not_to_be_shown, same_units_blk));
        num_pics = numel(pics);
        extra_time = num_pics * Nrep * ISI;
        options{opt_idx} = sprintf("Keep %d same_units from block %d (will add %2.2f seconds)", ...
                                            num_pics, blk, extra_time);
        dict_pics{opt_idx} = pics;
        opt_idx = opt_idx + 1;
    end
    for blk=1:length(same_categories_cell)
        same_categories_blk = setdiff(same_categories_cell{blk}, all_selected_to_exp, 'stable');
        pics = pics_not_to_be_shown(ismember(pics_not_to_be_shown, same_categories_blk));
        num_pics = numel(pics);
        extra_time = num_pics * Nrep * ISI;
        options{opt_idx} = sprintf("Keep %d same_categories from block %d (will add %2.2f seconds)", ...
                                            num_pics, blk, extra_time);
        dict_pics{opt_idx} = pics;
        opt_idx = opt_idx + 1;
    end
    % Others
    num_pics = numel(others);
    extra_time = num_pics * Nrep * ISI;
    options{opt_idx} = sprintf("Keep %d others (will add %2.2f seconds)", ...
                                                                    num_pics, extra_time);
    dict_pics{opt_idx} = others;

    % Get center of the screen
    screenSize = get(0, 'ScreenSize');
    centerX = screenSize(3) / 2;
    centerY = screenSize(4) / 2;
    fig_width = 500;
    fig_height = 300 + (opt_idx * 50);

    nrep_isi_msg_height = 50;

    % Create a figure window at the center of the screen
    fig = figure('Name', 'Extend Screening', 'NumberTitle', 'off', 'MenuBar', ...
                 'none', 'ToolBar', 'none', 'Resize','off','Position', ...
                 [centerX - 150, centerY - 150, fig_width, fig_height]);

    % Nrep, ISI
    nrep_isi = uicontrol("Style", "text", 'Position', [10, fig_height-50, fig_width, 30]);
    nrep_isi.String = sprintf("Config:- Npics: %d, Nrep: %d, ISI: %2.2f seconds", Npics, Nrep, ISI);
    
    % Create checkboxes for each choice
    numChoices = numel(options);
    checkboxes = gobjects(numChoices, 1);
    for i = 1:numChoices
        % Add checkboxes at the center of the figure
        checkboxes(i) = uicontrol('Style', 'checkbox', 'String', options{i}, ...
                                  'Position', [25, fig_height - nrep_isi_msg_height - 50 * i, fig_width, 30], ...
                                  'Callback',{@updateScreeningTime, i});
    end

    msg = uicontrol("Style", "text", 'Position', [10, fig_height - nrep_isi_msg_height - 55 * (numChoices+1), fig_width, 50]);

    % OK and Cancel buttons
    okButton = uicontrol('Style', 'pushbutton', 'String', 'OK', 'Callback', @okButtonCallback, 'Position', [120, 20, 60, 30]);
    cancelButton = uicontrol('Style', 'pushbutton', 'String', 'Cancel', 'Callback', @cancelButtonCallback, 'Position', [220, 20, 60, 30]);

    % Wait for the user to close the figure
    uiwait(fig);

    % Callback based on checkbox selection
    function updateScreeningTime(hObject, eventData, checkBoxId)
        extra_time = 0;
        checked = false(numChoices, 1);
        npics_scr = Npics;
        pics_removed = [];
        pics_to_keep = [];
        for i = 1:numChoices
            checked(i) = get(checkboxes(i), 'Value');
            if checked(i) == 1
                pics_to_keep = [pics_to_keep; dict_pics{i}(:)];
                num_pics = numel(dict_pics{i});
                npics_scr = npics_scr + num_pics;
                extra_time = extra_time + (num_pics * Nrep * ISI);
            else
                pics_removed = [pics_removed; dict_pics{i}(:)];
            end
        end

        min_seq_length = ceil(30/ISI);
        Nseqxrep = floor(npics_scr/min_seq_length);
        if Nseqxrep <=0
            Nseqxrep = 1;
        end
        seq_length = floor(npics_scr/Nseqxrep);

        if seq_length*Nseqxrep<npics_scr % Condtion to check if new num of pics are divisible by new seq_length

            Npics_new = (seq_length + 1) * Nseqxrep; % Increase seq_length by one to make sure none of the selected2notremove are lost.
            extra_others = Npics_new - npics_scr;
            if extra_others > 0
                if checked(4) ~= 1
                    seq_length = seq_length + 1;
                else
                    npics_scr = seq_length * Nseqxrep;
                    fprintf('No extra pics to round up. Reducing Npics to: %d \n', npics_scr);
                end
            end
        end
        seq_time = seq_length * ISI;
        tot_scr_time = npics_scr * Nrep * ISI;

        msg.String = sprintf("SCR %d will have %d sequences.\n Each sequence will have %d pics(%2.2f seconds).\n Total time: %2.2f seconds. Added: %2.2f seconds.",nxt_scr, Nseqxrep*Nrep, seq_length, seq_time, tot_scr_time, extra_time);
    end

    % Callback function for OK button
    function okButtonCallback(~, ~)
        % Get the state of each checkbox
        checked = false(numChoices, 1);
        for i = 1:numChoices
            checked(i) = get(checkboxes(i), 'Value');
        end

        options(checked)

%         choices = checked;

        % Close the figure
        delete(fig);
    end

    % Callback function for Cancel button
    function cancelButtonCallback(~, ~)
        pics_to_keep = [];

        % Close the figure
        delete(fig);
    end
end

