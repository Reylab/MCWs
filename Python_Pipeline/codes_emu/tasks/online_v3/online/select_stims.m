function selected = select_stims(figs_cells, ss_num, save_figs, ...
                                 ext_lbl, show_sel_count, miniscr_sel_count, b_menu)
    
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
        if show_sel_count && N > 9
            
            [NREP, NSEQ, seq_length, estimated_duration] = calculate_miniscr_time(N);
            disp(['Miniscr selection count:' num2str(N) ' (' num2str(estimated_duration) ...
                ' minutes) (NREP:' num2str(NREP) ' NSEQ:' num2str(NSEQ) ' SEQLEN:' num2str(seq_length) ')']);
            if N == 40
                f = msgbox(sprintf("\n 40 selected for miniscreening"),"Miniscreening selection");
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


    % Select everything (just for debugging, otherwise next line should be
    % commented 
    % selected = ones(20*length(figs_cells),1);

end
