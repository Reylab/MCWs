function selected = select_stims(figs_cells, ss_num, save_figs, ...
                                 ext_lbl, show_sel_count, miniscr_sel_count, b_menu, face_tracking)
    
    %It returns a vector with -1 for stimulus to remove and 1 for stimulus to
    %expand, non changed stimulus will have a value of zero.
    selected = zeros(20*length(figs_cells),1);
    get_next_state = @(x) mod(x-1,3)-1;

    if ~exist('show_sel_count','var')
        show_sel_count = false;
    end
    if ~exist('miniscr_sel_count','var')
        miniscr_sel_count = 0;
    end
    
    % Face/Non-face tracking initialization
    if ~exist('face_tracking', 'var') || isempty(face_tracking)
        face_tracking = struct('enabled', false);
    end
    if face_tracking.enabled
        faces_selected = face_tracking.current_faces;
        nonfaces_selected = face_tracking.current_nonfaces;
    else
        faces_selected = 0;
        nonfaces_selected = 0;
    end

    %pos = {[8.2000   46.6000  753.6000  737.6000],[8.2000+770   46.6000  753.6000  737.6000]};
    xpos = [0.127,0.2917,0.4516,0.614,0.777];
    ypos = [0.91,0.698,0.49,0.28];
    width = 0.02;
    height = 0.0246;
    ncol = length(xpos);
    nrow = length(ypos);
    nstim = 1;
    for fi=1:length(figs_cells)
        for yi=1:nrow
            for xi = 1:ncol
                q=uicontrol('Parent',figs_cells{fi},'Style','pushbutton','String','X','Units','normalized',...
                    'Position',[xpos(xi) ypos(yi) width height],'Visible','on',...
                    'BackgroundColor','yellow','Callback',{@select_button_Callback,nstim},'UserData',0);
    %                 'BackgroundColor',[0.39,0.83,0.07],'Callback',{@select_button_Callback,nstim},'UserData',0);
                nstim = nstim+1;
            end
        end
        if ~b_menu
            set(figs_cells{fi}, 'MenuBar','none');
        end
        set(figs_cells{fi}, 'closerequestfcn', {@save_and_close fi});
        %figs_cells{fi}.Position=pos{mod(fi+1,2)+1};
        figs_cells{fi}.Position(3:4)=[753.6000  737.6000];
    end


    figs_open = true;
    cellfun(@(x) set(x,'Visible',1),figs_cells)
    ofigs = cellfun(@(x)isgraphics(x, 'figure'),figs_cells);


    while figs_open
        uiwait(figs_cells{find(ofigs,1)});
        ofigs = cellfun(@(x)isgraphics(x, 'figure'),figs_cells);
        figs_open = any(ofigs);
    end


    function select_button_Callback(hObject,eventdata,index)
        next_state = get_next_state(selected(index));
        if next_state == 1
    %         set(hObject,'BackgroundColor','yellow');
            set(hObject,'BackgroundColor',[0.39,0.83,0.07]);

        elseif next_state == -1
            set(hObject,'BackgroundColor','red');
        else
    %         set(hObject,'BackgroundColor',[0.39,0.83,0.07]);
            set(hObject,'BackgroundColor','yellow');
        end
        selected(index)= next_state;
        N = sum(selected==1) + miniscr_sel_count;
        
        % Update face/non-face counts in real-time
        if face_tracking.enabled
            prev_faces = faces_selected;
            prev_nonfaces = nonfaces_selected;
            [faces_selected, nonfaces_selected] = count_faces_nonfaces_selected( ...
                selected, face_tracking);
            fprintf('Faces: %d | Non-Faces: %d | Total: %d\n', ...
                faces_selected, nonfaces_selected, faces_selected + nonfaces_selected);
            % Show dialog when reaching 15 Faces
            if faces_selected >= 15 && prev_faces < 15
                msgbox(sprintf('\n 15 Faces Selected'), 'Face Selection');
            end
            % Show dialog when reaching 15 NonFaces
            if nonfaces_selected >= 15 && prev_nonfaces < 15
                msgbox(sprintf('\n 15 NonFaces Selected'), 'NonFace Selection');
            end
        end
        
        if show_sel_count && N > 9
            if N <= 40
                [NREP, NSEQ, seq_length, estimated_duration] = calculate_miniscr_time(N);
                disp(['Miniscr selection count:' num2str(N) ' (' num2str(estimated_duration) ...
                    ' minutes) (NREP:' num2str(NREP) ' NSEQ:' num2str(NSEQ) ' SEQLEN:' num2str(seq_length) ')']);
                if N == 40
                    f = msgbox(sprintf("\n 40 selected for miniscreening"),"Miniscreening selection");
                end
            else
                disp(['Selection count: ' num2str(N) ' (exceeds 40 miniscr limit)']);
            end
        end
    end

    function save_and_close(hObject,eventdata,f_idx)
        % Save the figure
        %     print(figs_cells{f_idx},'-dpng',sprintf('%s_best_resp_subscr %d_fig %d.png', 'select_win', ss_num, f_idx));
        if save_figs
            F = getframe(figs_cells{f_idx});
            imwrite(F.cdata, [sprintf('%s_best_resp_subscr %d_fig %d.png', ext_lbl, ss_num, f_idx)])
        end

        % Close the figure
        closereq;
    end

    function [n_faces, n_nonfaces] = count_faces_nonfaces_selected(selected_vec, ft)
        % Count how many unique faces and non-faces are currently selected
        n_faces = ft.current_faces;
        n_nonfaces = ft.current_nonfaces;
        
        selected_indices = find(selected_vec == 1);
        
        % Get unique stim_numbers to avoid double-counting
        stim_nums_selected = [];
        for ii = 1:numel(selected_indices)
            sel_idx = selected_indices(ii);
            if sel_idx <= numel(ft.stim_numbers)
                stim_nums_selected(end+1) = ft.stim_numbers(sel_idx);
            end
        end
        unique_stim_nums = unique(stim_nums_selected);
        
        % Use pre-computed face_category if available, otherwise fallback to lookup
        use_precomputed = isfield(ft, 'ImageNames') && ...
                          istable(ft.ImageNames) && ...
                          ismember('face_category', ft.ImageNames.Properties.VariableNames);
        
        for ii = 1:numel(unique_stim_nums)
            stim_num = unique_stim_nums(ii);
            
            if use_precomputed
                % Use pre-computed face_category column
                category = ft.ImageNames.face_category{stim_num};
                if strcmpi(category, 'Faces')
                    n_faces = n_faces + 1;
                elseif strcmpi(category, 'Non Faces')
                    n_nonfaces = n_nonfaces + 1;
                end
            elseif isfield(ft, 'lookup_table') && ~isempty(ft.lookup_table)
                % Fallback to lookup table
                img_name = ft.ImageNames.name{stim_num};
                concept_name = regexprep(img_name, '_\d+\.(jpg|jpeg|png)$', '', 'ignorecase');
                concept_name = lower(concept_name);
                match_idx = find(strcmp(ft.lookup_table.Name, concept_name), 1);
                if ~isempty(match_idx)
                    folder_category = ft.lookup_table.Folder{match_idx};
                    if strcmpi(folder_category, 'Faces')
                        n_faces = n_faces + 1;
                    elseif strcmpi(folder_category, 'Non Faces')
                        n_nonfaces = n_nonfaces + 1;
                    end
                end
            end
        end
    end

    % Select everything (just for debugging, otherwise next line should be
    % commented 
    % selected = ones(20*length(figs_cells),1);

end
