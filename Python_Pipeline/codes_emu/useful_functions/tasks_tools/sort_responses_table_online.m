function sorted_datat=sort_responses_table_online(data,priority_chnums)
%     d_sps0 = sortrows(data(data.min_spk_test==0,:),{'ntrials','zscore'},{'ascend','descend'});
%     d_sps1_st0 = sortrows(data(data.min_spk_test==1 & data.p_test==0,:),{'zscore','good_lat','ntrials',},{'descend','descend','descend'});
%     d_sps1_st0 = [d_sps1_st0(d_sps1_st0.zscore>=3.5 | isnan(d_sps1_st0.zscore),:);...
%                   d_sps1_st0(d_sps1_st0.zscore<3.5,:)];%part in 3.5
%     d_sps1_st1 = sortrows(data(data.min_spk_test==1 & data.p_test==1,:),{'good_lat','zscore','ntrials',},{'descend','descend','descend'});
%     sorted_datat=[d_sps1_st1;d_sps1_st0;d_sps0];
    

%     data_ms1 = sortrows(data(data.min_spk_test==1,:),{'p_test','zscore','ntrials',},{'descend','descend','ascend'});
%     data_ms0 = sortrows(data(data.min_spk_test==0,:),{'p_test','zscore','ntrials',},{'descend','descend','ascend'});
% 
%     nanlat = isnan(data_ms1.onset);
%     sorted_datat=[data_ms1(data_ms1.good_lat==1,:);...
%         data_ms1(data_ms1.good_lat==0 & ~nanlat,:);...
%         data_ms1(data_ms1.good_lat==0 & nanlat,:);...
%         data_ms0];

    ch_nums = cellfun(@str2double,regexp(data.channel,'\d+(\.\d+)?','match'));

    if exist('priority_chnums','var')
        data_ms1_st1_gl1 = sortrows(data(data.min_spk_test==1 & data.p_test==1 & data.good_lat==1 & ismember(ch_nums,priority_chnums),:),{'zscore','ntrials',},{'descend','descend'});
        data_ms1_st0_hzc_gl1_ht = sortrows(data(data.min_spk_test==1 & data.p_test==0 & ~(data.zscore<4.5) & data.good_lat==1  & data.ntrials>4 & ismember(ch_nums,priority_chnums),:),{'zscore','ntrials',},{'descend','descend'});
        data_ms1_st0_hzc_gl1_lt = sortrows(data(data.min_spk_test==1 & data.p_test==0 & ~(data.zscore<4.5) & data.good_lat==1 & data.ntrials<=4 & ismember(ch_nums,priority_chnums),:),{'zscore','ntrials',},{'descend','descend'});
        top_part = [data_ms1_st1_gl1; data_ms1_st0_hzc_gl1_ht; data_ms1_st0_hzc_gl1_lt];
        
        top_part = [top_part;sortrows(data(data.min_spk_test==1 & data.p_test==1 & data.good_lat==1 & ~ismember(ch_nums,priority_chnums),:),{'zscore','ntrials',},{'descend','descend'})];
        top_part = [top_part;sortrows(data(data.min_spk_test==1 & data.p_test==0 & ~(data.zscore<4.5) & data.good_lat==1  & data.ntrials>4 & ~ismember(ch_nums,priority_chnums),:),{'zscore','ntrials',},{'descend','descend'});];
        top_part = [top_part;sortrows(data(data.min_spk_test==1 & data.p_test==0 & ~(data.zscore<4.5) & data.good_lat==1 & data.ntrials<=4 & ~ismember(ch_nums,priority_chnums),:),{'zscore','ntrials',},{'descend','descend'});];
    else
        data_ms1_st1_gl1 = sortrows(data(data.min_spk_test==1 & data.p_test==1 & data.good_lat==1,:),{'zscore','ntrials',},{'descend','descend'});
        data_ms1_st0_hzc_gl1_ht = sortrows(data(data.min_spk_test==1 & data.p_test==0 & ~(data.zscore<4.5) & data.good_lat==1  & data.ntrials>4,:),{'zscore','ntrials',},{'descend','descend'});
        top_part = [data_ms1_st1_gl1; data_ms1_st0_hzc_gl1_ht];
    end
        
    data_ms1_st1_gl0 = sortrows(data(data.min_spk_test==1 & data.p_test==1 & data.good_lat==0 & ~isnan(data.onset),:),{'zscore','ntrials','dura',},{'descend','descend','descend'});

    data_ms1_st1_gl0_nanlat = sortrows(data(data.min_spk_test==1 & data.p_test==1 & data.good_lat==0 & isnan(data.onset),:),{'zscore','ntrials',},{'descend','descend'});
    

    data_ms1_st0_hzc_gl0 = sortrows(data(data.min_spk_test==1 & data.p_test==0 & ~(data.zscore<4.5) & data.good_lat==0 & ~isnan(data.onset),:),{'zscore','ntrials',},{'descend','descend'});
    data_ms1_st0_hzc_gl0_nanlat = sortrows(data(data.min_spk_test==1 & data.p_test==0 & ~(data.zscore<4.5) & data.good_lat==0 & isnan(data.onset),:),{'zscore','ntrials',},{'descend','descend'});

    data_ms1_st0_lzc_gl1 = sortrows(data(data.min_spk_test==1 & data.p_test==0 & (data.zscore<4.5) & data.good_lat==1,:),{'zscore','ntrials',},{'descend','descend'});
    data_ms1_st0_lzc_gl0 = sortrows(data(data.min_spk_test==1 & data.p_test==0 & (data.zscore<4.5) & data.good_lat==0 & ~isnan(data.onset),:),{'zscore','ntrials',},{'descend','descend'});
    data_ms1_st0_lzc_gl0_nanlat = sortrows(data(data.min_spk_test==1 & data.p_test==0 & (data.zscore<4.5) & data.good_lat==0 & isnan(data.onset),:),{'zscore','ntrials',},{'descend','descend'});

    data_ms0 = sortrows(data(data.min_spk_test==0,:),{'ntrials','zscore'},{'descend','descend'});
    
    sorted_datat=[top_part;...
%         data_ms1_st1_gl1;...
%         data_ms1_st0_hzc_gl1_ht;...
        data_ms1_st1_gl0;...
        data_ms1_st0_hzc_gl1_lt; ...
        data_ms1_st0_hzc_gl0;...
        data_ms1_st1_gl0_nanlat;...
        data_ms1_st0_hzc_gl0_nanlat;...
        
        data_ms1_st0_lzc_gl1 ;...
        data_ms1_st0_lzc_gl0;...
        data_ms1_st0_lzc_gl0_nanlat;...
        data_ms0];

    % Remove duplicate rows, keep first occurrence
    cols = {'stim_number', 'class', 'channel'};  % columns for unique
    [~, ia, ~] = unique(sorted_datat(:, cols), 'rows', 'stable');
    % counts = accumarray(ic, 1);
    % dup_groups = find(counts > 1);
    % dupes = find(ismember(ic, dup_groups));
    % % Show duplicates
    % for g = dup_groups'
    %     disp("Duplicate group: " + string(g));
    %     disp(sorted_datat(ic == g, :));   % all rows belonging to this group
    % end
    sorted_datat = sorted_datat(ia, :);

%      sorted_datat = sorted_datat(sorted_datat.ntrials>3 ,:);

end