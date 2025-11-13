function choices = extend_seq_menu(Npics, Nrep, ISI, nxt_scr, num_selected2explore, num_same_units, num_same_categories, num_others)
    extra_time_selected2explore = num_selected2explore * Nrep * ISI;
    extra_time_same_units = num_same_units * Nrep * ISI;
    extra_time_same_categories = num_same_categories * Nrep * ISI;
    extra_time_others = num_same_categories * Nrep * ISI;

    opt_selected2explore = sprintf("Keep %d selected2explore (will add %2.2f seconds)", num_selected2explore, extra_time_selected2explore);
    opt_same_units = sprintf("Keep %d same_units (will add %2.2f seconds)", num_same_units, extra_time_same_units);
    opt_same_cat = sprintf("Keep %d same_categories (will add %2.2f seconds)", num_same_categories, extra_time_same_categories);
    opt_others = sprintf("Keep %d others (will add %2.2f seconds)", num_others, extra_time_others);
    
    % Get center of the screen
    screenSize = get(0, 'ScreenSize');
    centerX = screenSize(3) / 2;
    centerY = screenSize(4) / 2;
    fig_width = 500;
    fig_height = 400;

    nrep_isi_msg_height = 50;

    % Create a figure window at the center of the screen
    fig = figure('Name', 'Extend Screening', 'NumberTitle', 'off', 'MenuBar', ...
                 'none', 'ToolBar', 'none', 'Resize','off','Position', ...
                 [centerX - 150, centerY - 150, fig_width, fig_height]);

    % Nrep, ISI
    nrep_isi = uicontrol("Style", "text", 'Position', [10, fig_height-50, fig_width, 30]);
    nrep_isi.String = sprintf("Config:- Npics: %d, Nrep: %d, ISI: %2.2f seconds", Npics, Nrep, ISI);

    % options
    options = {opt_selected2explore, opt_same_units, opt_same_cat, opt_others};

    % Create checkboxes for each choice
    numChoices = numel(options);
    checkboxes = gobjects(numChoices, 1);
    for i = 1:numChoices
        % Add checkboxes at the center of the figure
        checkboxes(i) = uicontrol('Style', 'checkbox', 'String', options{i}, 'Position', [25, fig_height - nrep_isi_msg_height - 50 * i, fig_width, 30], 'Callback',{@updateScreeningTime, i});
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
        for i = 1:numChoices
            checked(i) = get(checkboxes(i), 'Value');
        end
        if checked(1) == 1
            npics_scr = npics_scr + num_selected2explore;
            extra_time = extra_time + extra_time_selected2explore;
        end
        if checked(2) == 1
            npics_scr = npics_scr + num_same_units;
            extra_time = extra_time + extra_time_same_units;
        end
        if checked(3) == 1
            npics_scr = npics_scr + num_same_categories;
            extra_time = extra_time + extra_time_same_categories;
        end
        if checked(4) == 1
            npics_scr = npics_scr + num_others;
            extra_time = extra_time + extra_time_others;
        end

        min_seq_length = ceil(30/ISI);
        Nseqxrep = floor(npics_scr/min_seq_length);
        seq_length = floor(npics_scr/Nseqxrep);

        if seq_length*Nseqxrep<npics_scr % Condtion to check if new num of pics are divisible by new seq_length

            Npics = (seq_length + 1) * Nseqxrep; % Increase seq_length by one to make sure none of the selected2notremove are lost.
            extra_others = Npics - npics_scr;
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

        msg.String = sprintf("SCR:%d will have %d sequences.\n Each sequence will have %d pics(%2.2f seconds).\n Total time: %2.2f seconds. Added: %2.2f seconds.",nxt_scr, Nseqxrep*Nrep, seq_length, seq_time, tot_scr_time, extra_time);
    end

    % Callback function for OK button
    function okButtonCallback(~, ~)
        % Get the state of each checkbox
        checked = false(numChoices, 1);
        for i = 1:numChoices
            checked(i) = get(checkboxes(i), 'Value');
        end

        options(checked)

        choices = checked;

        % Close the figure
        delete(fig);
    end

    % Callback function for Cancel button
    function cancelButtonCallback(~, ~)
        % Return an empty array to indicate cancellation
        options = [];

        % Close the figure
        delete(fig);
    end
end

