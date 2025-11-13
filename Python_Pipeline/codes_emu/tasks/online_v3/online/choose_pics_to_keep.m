function [pics_to_keep, pics_removed, categ_localiz_history_updated] = choose_pics_to_keep(experiment, next_pics, ISI, nxt_scr, ...
                                                            pics_not_to_be_shown, unused_available_pics, selected2explore_cell, ...
                                                            fetched_pics_cell,  others, ...
                                                            b_last_block, exp_time_taken, categ_localiz_history, num_datat_best)
    % UI Constants
    UI = struct();
    UI.WINDOW_WIDTH = 600;
    UI.WINDOW_MIN_HEIGHT = 300;
    UI.WINDOW_MAX_HEIGHT_RATIO = 0.8;  % Maximum height as ratio of screen height
    UI.HEADER_HEIGHT = 100;
    UI.FOOTER_HEIGHT = 60;
    UI.SCROLLBAR_WIDTH = 20;
    UI.CHECKBOX_HEIGHT = 30;
    UI.CHECKBOX_SPACING = 20;
    UI.CHECKBOX_ROW_HEIGHT = UI.CHECKBOX_HEIGHT + UI.CHECKBOX_SPACING;
    UI.CHECKBOX_LEFT_MARGIN = 25;
    UI.MESSAGE_HEIGHT = 50;
    UI.BUTTON_WIDTH = 60;
    UI.BUTTON_HEIGHT = 30;
    UI.BUTTON_SPACING = 40;
    UI.CONTENT_PADDING = 10;

    Npics = experiment.NPICS(nxt_scr);
    Nrep = experiment.NREP(nxt_scr);
    time_taken_mins = floor(exp_time_taken / 60);
    time_taken_secs = ceil(mod(exp_time_taken, 60));
    config_scr_time = Npics * Nrep * ISI;
    config_scr_time_estimate_mins = floor(config_scr_time / 60);
    config_scr_time_estimate_secs = ceil(mod(config_scr_time, 60));
    pics_to_keep = [];
    pics_removed = [];
    extra_rep_pics = [];
    categ_localiz_history_updated = {};

    % Generate options and initialize variables
    [options, dict_pics, extra_time_total] = generate_options(experiment, ISI, nxt_scr, ...
        pics_not_to_be_shown, selected2explore_cell, fetched_pics_cell, others, b_last_block);
    checked = false(numel(options), 1);
    
    % Calculate dimensions
    UI = calculate_window_dimensions(UI, options, extra_time_total);
    
    % Create and position UI elements
    [fig, UI_elements] = create_ui_elements(UI, options, experiment, ISI, nxt_scr, ...
        b_last_block, exp_time_taken);

    updateScreeningTime();

    % Scroll to top after all elements are created
    drawnow;  % Ensure UI is fully rendered
    set(UI_elements.scrollbar, 'Value', 0);
    
    uiwait(fig);
    
    function [options, dict_pics, extra_time_total] = generate_options(experiment, ISI, nxt_scr, ...
            pics_not_to_be_shown, selected2explore_cell, fetched_pics_cell, others, b_last_block)
        opt_idx = 1;
        dict_pics = {};
        extra_time_total = 0;
        options = {};
        all_selected_to_exp = cell2mat(selected2explore_cell');
        npics_scr = 0;
        
        for blk=1:length(selected2explore_cell)
            pics = pics_not_to_be_shown(ismember(pics_not_to_be_shown, selected2explore_cell{blk}));
            num_pics = numel(pics);
            if num_pics == 0
                continue;
            end
            npics_scr = npics_scr + num_pics;
            [extra_time, rep_count] = calculate_extra_time(pics);
            extra_time_total = extra_time_total + extra_time;
            if b_last_block
                options{opt_idx} = sprintf("Keep %d selected2explore from block %d", ...
                    num_pics, blk);
            else
                options{opt_idx} = sprintf("Keep %d selected2explore from block %d (will add %2.2f seconds)", ...
                    num_pics, blk, extra_time);
            end
            dict_pics{opt_idx} = pics;
            opt_idx = opt_idx + 1;
        end
        num_cats = numel(fetched_pics_cell{1}{1}.categ_info);
        categ_info_arr = cell(1, num_cats);
        for cat_idx = 1:num_cats
            categ_info_arr{cat_idx} = struct;
            categ_info_arr{cat_idx}.options = {};
            categ_info_arr{cat_idx}.dict_pics = {};
        end
        all_fetched_pics = [];
        for blk = 1:numel(fetched_pics_cell)
            for j = 1:numel(fetched_pics_cell{blk})
                categ_info_list = fetched_pics_cell{blk}{j}.categ_info;
                for k = 1:numel(categ_info_list)
                    categ_info = categ_info_list{k};
                    fetched_item_blk = setdiff(categ_info.pic_ids, all_selected_to_exp, 'stable');
                    pics = pics_not_to_be_shown(ismember(pics_not_to_be_shown, fetched_item_blk));
                    all_fetched_pics = [all_fetched_pics;pics];
                    num_pics = numel(pics);
                    if num_pics == 0
                        continue;
                    end
                    npics_scr = npics_scr + num_pics;
                    [extra_time, rep_count] = calculate_extra_time(pics);
                    extra_time_total = extra_time_total + extra_time;
                    if b_last_block
                        categ_info_arr{k}.options{end+1} = sprintf("Keep %d %s cat:%s from block %d", ...
                            num_pics, fetched_pics_cell{blk}{j}.rule_name, categ_info.category, blk);
                    else
                        categ_info_arr{k}.options{end+1} = sprintf("Keep %d %s cat:%s from block %d (will add %2.2f seconds)", ...
                            num_pics, fetched_pics_cell{blk}{j}.rule_name, categ_info.category, blk, extra_time);
                    end
                    categ_info_arr{k}.dict_pics{end+1} = pics;
                end
            end
        end
        for cat_idx = 1:num_cats
            for opt=1:numel(categ_info_arr{cat_idx}.options)
                options{opt_idx} = categ_info_arr{cat_idx}.options{opt};
                dict_pics{opt_idx} = categ_info_arr{cat_idx}.dict_pics{opt};
                opt_idx = opt_idx + 1;
            end
        end
        % Others
        num_pics = numel(others);
        npics_scr = npics_scr + num_pics;
        [extra_time, rep_count] = calculate_extra_time(others);
        extra_time_total = extra_time_total + extra_time;
        options{opt_idx} = sprintf("Keep                   already shown in block %d(best:%d)  (will add %2.2f seconds)", ...
            nxt_scr-1, num_datat_best, extra_time);
        dict_pics{opt_idx} = others;
        opt_idx = opt_idx + 1;
        
        if ~b_last_block
            % Random additions if needed to get to Npics
            if npics_scr >= Npics
                random_pics_count = 0;
            else
                random_pics_count = Npics - npics_scr;
            end
            rep_count = random_pics_count * Nrep;
            extra_time = rep_count * ISI; 
            extra_time_total = extra_time_total + extra_time;
            options{opt_idx} = sprintf("Add %d random pics (will add %2.2f seconds)", ...
                                        random_pics_count, extra_time);
            dict_pics{opt_idx} = unused_available_pics(1:random_pics_count);
        end
        checked = false(numel(options), 1);
    end
    
    function UI = calculate_window_dimensions(UI, options, extra_time_total)
        % Get screen dimensions
        screen_size = get(0, 'ScreenSize');
        
        % Calculate content height
        num_options = numel(options);
        UI.CONTENT_HEIGHT = (num_options * UI.CHECKBOX_ROW_HEIGHT) + UI.MESSAGE_HEIGHT;
        
        % Calculate window height (limited by screen size)
        max_window_height = screen_size(4) * UI.WINDOW_MAX_HEIGHT_RATIO;
        UI.WINDOW_HEIGHT = min(max_window_height, ...
            UI.CONTENT_HEIGHT + UI.HEADER_HEIGHT + UI.FOOTER_HEIGHT);
        
        % Calculate scroll panel height
        UI.SCROLL_PANEL_HEIGHT = UI.WINDOW_HEIGHT - UI.HEADER_HEIGHT - UI.FOOTER_HEIGHT;
        
        % Calculate window position
        UI.WINDOW_LEFT = (screen_size(3) - UI.WINDOW_WIDTH) / 2;
        UI.WINDOW_TOP = (screen_size(4) - UI.WINDOW_HEIGHT) / 2;
    end
    
    function [fig, UI_elements] = create_ui_elements(UI, options, experiment, ISI, nxt_scr, ...
            b_last_block, exp_time_taken)
        % Create main figure
        fig = uifigure('Name', 'Extend Screening', ...
            'NumberTitle', 'off', ...
            'MenuBar', 'none', ...
            'ToolBar', 'none', ...
            'Resize', 'off', ...
            'Position', [UI.WINDOW_LEFT, UI.WINDOW_TOP, UI.WINDOW_WIDTH, UI.WINDOW_HEIGHT]);

        % Create header panel and elements
        UI_elements.header = create_header_panel(fig, UI, experiment, ISI, nxt_scr);

        % Create scroll panel and content
        [UI_elements.scroll_panel, UI_elements.content_panel, UI_elements.scrollbar] = ...
            create_scroll_panel(fig, UI, options);

        % Create footer panel and buttons
        UI_elements.footer = create_footer_panel(fig, UI);

        % Create checkboxes
        [UI_elements.checkboxes, UI_elements.sp_others] = create_checkboxes(UI_elements.content_panel, UI, options);

        % Create message text
        UI_elements.msg = create_message_text(UI_elements.content_panel, UI, options);

        % Store a handle to all checkboxes for the "All" checkbox callback
        UI_elements.header.all_checkbox.UserData = UI_elements.checkboxes;
    end
    
    function header = create_header_panel(fig, UI, experiment, ISI, nxt_scr)
        header.panel = uipanel('Parent', fig, ...
            'Units', 'pixels', ...
            'Position', [0, UI.WINDOW_HEIGHT - UI.HEADER_HEIGHT, UI.WINDOW_WIDTH, UI.HEADER_HEIGHT], ...
            'BorderType', 'none');

        header.text = uicontrol(header.panel, ...
            'Style', 'text', ...
            'Position', [UI.CONTENT_PADDING + 100, UI.CONTENT_PADDING, ...
                        UI.WINDOW_WIDTH - 2*UI.CONTENT_PADDING - 100, UI.HEADER_HEIGHT - 2*UI.CONTENT_PADDING], ...
            'HorizontalAlignment', 'left', ...
            'String', sprintf("Elapsed time: %d mins, %d secs.\n" + ...
                                "SCR %d config:- Npics: %d, Nrep: %d, scr_time: %d mins %d secs", ...
                                time_taken_mins, time_taken_secs, nxt_scr, Npics, Nrep, ...
                                config_scr_time_estimate_mins, config_scr_time_estimate_secs));

        % Add "All" checkbox at the top left
        header.all_checkbox = uicontrol(header.panel, ...
            'Style', 'checkbox', ...
            'String', 'All', ...
            'Position', [UI.CONTENT_PADDING, UI.HEADER_HEIGHT - 20, 60, 20], ...
            'Callback', @allCheckboxCallback);
        
        if contains(experiment.subtask, 'CategLocaliz')
            % Add checkboxes to easily unselect all_but_2
            header.all_but_2_checkbox = uicontrol(header.panel, ...
                'Style', 'checkbox', ...
                'String', 'All_but_2', ...
                'Position', [UI.CONTENT_PADDING, UI.HEADER_HEIGHT - 40, 80, 20], ...
                'Callback', @all_but_2_CheckboxCallback);

            % Add checkboxes to easily unselect all_but_3
            header.all_but_3_checkbox = uicontrol(header.panel, ...
                'Style', 'checkbox', ...
                'String', 'All_but_3', ...
                'Position', [UI.CONTENT_PADDING, UI.HEADER_HEIGHT - 60, 80, 20], ...
                'Callback', @all_but_3_CheckboxCallback);

            % Add checkboxes to easily unselect all all_but_4
            header.all_but_4_checkbox = uicontrol(header.panel, ...
                'Style', 'checkbox', ...
                'String', 'All_but_4', ...
                'Position', [UI.CONTENT_PADDING, UI.HEADER_HEIGHT - 80, 80, 20], ...
                'Callback', @all_but_4_CheckboxCallback);            
        end
        
    end
    
    function [scroll_panel, content_panel, scrollbar] = create_scroll_panel(fig, UI, options)
        % Create main scroll panel
        scroll_panel = uipanel('Parent', fig, ...
            'Units', 'pixels', ...
            'Position', [0, UI.FOOTER_HEIGHT, UI.WINDOW_WIDTH, UI.SCROLL_PANEL_HEIGHT], ...
            'BorderType', 'none');
        
        % Create content panel
        content_panel = uipanel('Parent', scroll_panel, ...
            'Units', 'pixels', ...
            'Position', [0, 0, UI.WINDOW_WIDTH - UI.SCROLLBAR_WIDTH, UI.CONTENT_HEIGHT], ...
            'BorderType', 'none');
        
        % Create scrollbar
        scrollbar = uicontrol('Parent', fig, ...
            'Style', 'slider', ...
            'Units', 'pixels', ...
            'Position', [UI.WINDOW_WIDTH - UI.SCROLLBAR_WIDTH, UI.FOOTER_HEIGHT, ...
                        UI.SCROLLBAR_WIDTH, UI.SCROLL_PANEL_HEIGHT], ...
            'Value', 0, ...
            'Callback', @scrollContent);
        
        % Configure scrollbar
        if UI.CONTENT_HEIGHT > UI.SCROLL_PANEL_HEIGHT
            set(scrollbar, ...
                'Min', 0, ...
                'Max', UI.CONTENT_HEIGHT - UI.SCROLL_PANEL_HEIGHT, ...
                'SliderStep', [0.1, 0.2]);
        else
            set(scrollbar, 'Enable', 'off');
        end
        set(fig, 'WindowScrollWheelFcn', @mouseWheelScroll);
    end
    
    function footer = create_footer_panel(fig, UI)
        footer.panel = uipanel('Parent', fig, ...
            'Units', 'pixels', ...
            'Position', [0, 0, UI.WINDOW_WIDTH, UI.FOOTER_HEIGHT], ...
            'BorderType', 'none');
        
        % Calculate button positions
        ok_left = (UI.WINDOW_WIDTH - 2*UI.BUTTON_WIDTH - UI.BUTTON_SPACING) / 2;
        cancel_left = ok_left + UI.BUTTON_WIDTH + UI.BUTTON_SPACING;
        button_top = (UI.FOOTER_HEIGHT - UI.BUTTON_HEIGHT) / 2;
        
        % Create buttons
        footer.ok_button = uicontrol(footer.panel, ...
            'Style', 'pushbutton', ...
            'String', 'OK', ...
            'Callback', @okButtonCallback, ...
            'Position', [ok_left, button_top, UI.BUTTON_WIDTH, UI.BUTTON_HEIGHT]);
        
        footer.cancel_button = uicontrol(footer.panel, ...
            'Style', 'pushbutton', ...
            'String', 'Cancel', ...
            'Callback', @cancelButtonCallback, ...
            'Position', [cancel_left, button_top, UI.BUTTON_WIDTH, UI.BUTTON_HEIGHT]);
    end
    
    function [checkboxes, sp_others] = create_checkboxes(content_panel, UI, options)
        num_choices = numel(options);
        checkboxes = gobjects(num_choices, 1);
        
        for i = 1:num_choices
            top_position = UI.CONTENT_HEIGHT - (i * UI.CHECKBOX_ROW_HEIGHT);
            checkboxes(i) = uicontrol(content_panel, ...
                'Style', 'checkbox', ...
                'Value', true, ...
                'String', options{i}, ...
                'Position', [UI.CHECKBOX_LEFT_MARGIN, top_position, ...
                            UI.WINDOW_WIDTH - UI.SCROLLBAR_WIDTH - UI.CHECKBOX_LEFT_MARGIN, ...
                            UI.CHECKBOX_HEIGHT], ...
                'Callback', {@individualCheckboxCallback, i});
        end
        if b_last_block
            idx = num_choices;
        else
            idx = num_choices - 1;
        end
        % Create a spinner next to the random pics checkbox
        top_position = UI.CONTENT_HEIGHT - (idx * UI.CHECKBOX_ROW_HEIGHT);
        n_others = max(1, numel(others));
        sp_others = uispinner(content_panel, ...
            'Position', [UI.CHECKBOX_LEFT_MARGIN+55, top_position, ...
                            55, ...
                            UI.CHECKBOX_HEIGHT], ...
            'Limits', [0 n_others], ...
            'Value', numel(others), ...
            'ValueChangedFcn', @(src, event) spinnerCallback(src));
        if numel(others) == 0
            sp_others.Enable = 'off';
        end
    end
    
    function spinnerCallback(src)
        if isempty(others)
            return
        end
        sp_val = src.Value;
        
        [extra_time, rep_count] = calculate_extra_time(others(1:sp_val));
        extra_time_total = extra_time_total + extra_time;
        if ~b_last_block
            opt_count = numel(options)-1;
        else
            opt_count = numel(options);
        end
        
        dict_pics{opt_count} = others(1:sp_val);
        UI_elements.checkboxes(opt_count).String = sprintf("Keep                   others (will add %2.2f seconds)", ...
                                                            extra_time);
        updateScreeningTime();
    end

    function individualCheckboxCallback(hObject, ~, idx)
        % Update screening time as before
        updateScreeningTime();        
    end

    function updateAllCheckbox()
        % Update "All" checkbox state
        all_cb = UI_elements.header.all_checkbox;
        checkboxes = all_cb.UserData;
        all_checked = all(arrayfun(@(cb) get(cb, 'Value'), checkboxes));
        set(all_cb, 'Value', all_checked);
    end

    function update_all_but_2()
        % Update "All but 3" checkbox state
        all_but_2_cb = UI_elements.header.all_but_2_checkbox;
        all_checked = false;
        % Get all checkboxes
        checkboxes = UI_elements.checkboxes;
        
        for i = 1:numel(checkboxes)
            % check if the checkbox contains the text all_but_2
            if contains(get(checkboxes(i), 'String'), 'all_but_2')
                if get(checkboxes(i), 'Value')
                    all_checked = true;
                else
                    all_checked = false;
                    break;
                end
            end
        end
        set(all_but_2_cb, 'Value', all_checked);
    end

    function update_all_but_3()
        % Update "All but 3" checkbox state
        all_but_3_cb = UI_elements.header.all_but_3_checkbox;
        all_checked = false;
        % Get all checkboxes
        checkboxes = UI_elements.checkboxes;
        
        for i = 1:numel(checkboxes)
            % check if the checkbox contains the text all_but_3
            if contains(get(checkboxes(i), 'String'), 'all_but_3')
                if get(checkboxes(i), 'Value')
                    all_checked = true;
                else
                    all_checked = false;
                    break;
                end
            end
        end
        set(all_but_3_cb, 'Value', all_checked);
    end

    function update_all_but_4()
        % Update "All but 4" checkbox state
        all_but_4_cb = UI_elements.header.all_but_4_checkbox;
        all_checked = false;
        % Get all checkboxes
        checkboxes = UI_elements.checkboxes;
        
        for i = 1:numel(checkboxes)
            % check if the checkbox contains the text all_but_4
            if contains(get(checkboxes(i), 'String'), 'all_but_4')
                if get(checkboxes(i), 'Value')
                    all_checked = true;
                else
                    all_checked = false;
                    break;
                end
            end
        end
        set(all_but_4_cb, 'Value', all_checked);
    end

    function msg = create_message_text(content_panel, UI, options)
        msg = uicontrol(content_panel, ...
            'Style', 'text', ...
            'Position', [UI.CONTENT_PADDING, ...
                        UI.CONTENT_HEIGHT - ((numel(options) + 1) * UI.CHECKBOX_ROW_HEIGHT), ...
                        UI.WINDOW_WIDTH - UI.SCROLLBAR_WIDTH - 2*UI.CONTENT_PADDING, ...
                        UI.MESSAGE_HEIGHT]);
    end
    
    function scrollContent(src, ~)
        scroll_pos = get(src, 'Value');
        current_pos = get(UI_elements.content_panel, 'Position');
        set(UI_elements.content_panel, 'Position', [0, -scroll_pos, current_pos(3:4)]);
    end
    
    function mouseWheelScroll(~, eventdata)
        % Get current scroll position
        current_value = get(UI_elements.scrollbar, 'Value');
        
        % Calculate new position (negative scroll moves content up)
        scroll_step = 50; % Pixels to scroll per wheel movement
        new_value = current_value - (eventdata.VerticalScrollCount * scroll_step);
        
        % Ensure new position is within bounds
        min_scroll = get(UI_elements.scrollbar, 'Min');
        max_scroll = get(UI_elements.scrollbar, 'Max');
        new_value = max(min_scroll, min(max_scroll, new_value));
        
        % Update scroll position
        set(UI_elements.scrollbar, 'Value', new_value);
        scrollContent(UI_elements.scrollbar, []);
    end

    function header_text = updateScreeningTime()
        [pics_to_keep, pics_removed, npics_scr, scr_config] = update_selection();
        
        tot_trials = scr_config.Nseq * scr_config.seq_length;
        tot_scr_time = tot_trials * ISI;
        scr_time_estimate_mins = floor(tot_scr_time / 60);
        scr_time_estimate_secs = ceil(mod(tot_scr_time, 60));
        header_text = sprintf("Elapsed time: %d mins, %d secs.\n" + ...
                              "SCR %d config:- Npics: %d, Nrep: %d, scr_time: %d mins %d secs \n" + ...
                              "->SCR %d : Npics %d, seq_count: %d, seq_length: %d, scr_time: %d mins %d secs", ...
                               time_taken_mins, time_taken_secs, nxt_scr, Npics, Nrep, ...
                               config_scr_time_estimate_mins, config_scr_time_estimate_secs, ...
                               nxt_scr, npics_scr, scr_config.Nseq, scr_config.seq_length, ...
                               scr_time_estimate_mins, scr_time_estimate_secs);
        UI_elements.header.text.String = header_text;
        
    end
          
    function [pics_to_keep, pics_removed, npics_scr, scr_config] = update_selection()
        checked = false(numel(options), 1);
        npics_scr = 0;
        pics_removed = [];
        pics_to_keep = [];
        random_pics = [];        
        if ~b_last_block
            opt_count = numel(options) - 1;
        else
            opt_count = numel(options);
        end
        for i = 1:opt_count
            checked(i) = get(UI_elements.checkboxes(i), 'Value');
            if checked(i) == 1
                pics_to_keep = [pics_to_keep; dict_pics{i}(:)];
                num_pics = numel(dict_pics{i});
                npics_scr = npics_scr + num_pics;
                
            else
                pics_removed = [pics_removed; dict_pics{i}(:)];
            end
        end

        if ~b_last_block && npics_scr < Npics
            random_pics_count = Npics - npics_scr;
            
            if contains(experiment.subtask, 'CategLocaliz')
                [random_pics, categ_localiz_history_updated] = get_categ_localiz_random(categ_localiz_history, random_pics_count);
            else
                random_pics = unused_available_pics(1:random_pics_count);
            end
            dict_pics{end} = random_pics;
            chk_rand_pics = get(UI_elements.checkboxes(end), 'Value');
            if chk_rand_pics && numel(random_pics)
                pics_to_keep = [pics_to_keep; random_pics];
                npics_scr = npics_scr + random_pics_count;
            end
        elseif npics_scr >= Npics
            random_pics_count = 0;
            categ_localiz_history_updated = categ_localiz_history;
        end

        if ~b_last_block
            rand_pics_time = random_pics_count * Nrep * ISI;
            UI_elements.checkboxes(end).String = sprintf("Add %d random pics (will add %2.2f seconds)", ...
                                                     random_pics_count, rand_pics_time);
        end
        
        [~, scr_config, extra_rep_pics] = shuffle_rsvp_dynamic_2(experiment, pics_to_keep', nxt_scr);
        updateAllCheckbox();
        update_all_but_2();
        update_all_but_3();
        update_all_but_4();
    end
    
    function [extra_time, rep_count] = calculate_extra_time(pics_list)
        rep_count = numel(pics_list) * Nrep;
        extra_time = rep_count * ISI;
    end

    % Callback function for OK button
    function okButtonCallback(~, ~)
        % Get the state of each checkbox
        
        for i = 1:numel(options)
            checked(i) = get(UI_elements.checkboxes(i), 'Value');
        end

        header_text = updateScreeningTime();
        fprintf("%s \n", header_text);
        if numel(extra_rep_pics)
            fprintf('%d pics will be shown an extra time: %s\n', ...
                     numel(extra_rep_pics), strjoin(experiment.ImageNames.name(extra_rep_pics), ' '));
        end
        % Close the figure
        delete(fig);
    end

    % Callback function for Cancel button
    function cancelButtonCallback(~, ~)
        [pics_to_keep, pics_removed, random_pics, checked, npics_scr, scr_config] = update_selection();

        % Close the figure
        delete(fig);
    end
    % Add this new function after other nested functions:
    function allCheckboxCallback(hObject, ~)
        % Get all checkboxes
        checkboxes = hObject.UserData;
        % Set all checkboxes to the value of the "All" checkbox
        for i = 1:numel(checkboxes)
            set(checkboxes(i), 'Value', get(hObject, 'Value'));
        end
        % update screening time
        updateScreeningTime();
    end

    function all_but_2_CheckboxCallback(hObject, ~)
        % Get all checkboxes
        checkboxes = UI_elements.checkboxes;
        % Set all checkboxes to the value of the "All but 3" checkbox
        for i = 1:numel(checkboxes)
            % check if the checkbox contains the text all_but_3
            if contains(get(checkboxes(i), 'String'), 'all_but_2')
                set(checkboxes(i), 'Value', get(hObject, 'Value'));
            end
        end
        % update screening time
        updateScreeningTime();
    end

    function all_but_3_CheckboxCallback(hObject, ~)
        % Get all checkboxes
        checkboxes = UI_elements.checkboxes;
        % Set all checkboxes to the value of the "All but 3" checkbox
        for i = 1:numel(checkboxes)
            % check if the checkbox contains the text all_but_3
            if contains(get(checkboxes(i), 'String'), 'all_but_3')
                set(checkboxes(i), 'Value', get(hObject, 'Value'));
            end
        end
        % update screening time
        updateScreeningTime();
    end

    function all_but_4_CheckboxCallback(hObject, ~)
        % Get all checkboxes
        checkboxes = UI_elements.checkboxes;
        % Set all checkboxes to the value of the "All but 4" checkbox
        for i = 1:numel(checkboxes)
            % check if the checkbox contains the text all_but_4
            if contains(get(checkboxes(i), 'String'), 'all_but_4')
                set(checkboxes(i), 'Value', get(hObject, 'Value'));
            end
        end
        % update screening time
        updateScreeningTime();
    end
end

function [random_pics, categ_localiz_history] = get_categ_localiz_random(categ_localiz_history, ...
                                                random_pics_count)
    num_cats = numel(categ_localiz_history);
    per_cat_count = floor(random_pics_count / num_cats);
    extras_needed = mod(random_pics_count, num_cats);
    random_pics = [];

    for i_stim_cat = 1:numel(categ_localiz_history)
        cat_history = categ_localiz_history{i_stim_cat}.history;
        cat_pics = [];
        if i_stim_cat <= extras_needed
            req_count = per_cat_count + 1;
        else
            req_count = per_cat_count;
        end
        rand_comb_idxs = randperm(length(cat_history.Unused));
        for k=rand_comb_idxs
            feat_comb = cat_history(k,:);
            unused_pics = feat_comb.Unused{1};
            if numel(cat_pics) < req_count && numel(unused_pics)                
                cat_pics = [cat_pics; unused_pics(1)];
                random_pics = [random_pics; unused_pics(1)];            
                feat_comb.Visited = {true};
                feat_comb.Used = {[feat_comb.Used{1}; unused_pics(1)]};
                feat_comb.Unused = {setdiff(feat_comb.Unused{1}, unused_pics(1))};
            
                % Update categ_localiz_history
                categ_localiz_history{i_stim_cat}.history(k,:) = feat_comb;
            elseif numel(cat_pics) >= req_count
                break;
            end
        end
    end
end