function choice = keep_same_cat_dlg(Npics, n_scr, num_selected2explore, num_same_units, num_same_categories, num_to_add)
    answer = questdlg(sprintf("(SCR: %d Npics:%d) %d selected2explore, %d same_units, %d same_categories, %d new to be added. Do you wish to keep same category stimuli?", ...
                      Npics, n_scr, num_selected2explore, num_same_units, num_same_categories, num_to_add), ...
	'Same category stimuli', ...
	'Yes','No', 'No');
    choice = 0;
    % Handle response
    switch answer
        case 'Yes'
            fprintf("Keeping %d same category stimuli \n", num_same_categories)
            choice = 1;
        case 'No'
            choice = 0;
    end
end

