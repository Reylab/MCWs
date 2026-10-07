function omap = build_all_stim_onset_map(stimulus_struct, finalevents_struct, Nscr, stim_list)
% BUILD_ALL_STIM_ONSET_MAP  Same onset-map-building loop already used at the
% top of test_rescue_grape_metrics.m, generalized to take an explicit
% stim_list so it can build either the stim_list-scoped map (existing
% behavior) or the session-wide "every scored stimulus" map (new, for the
% overall summary) without duplicating a second copy of this loop by hand.
%
% VARIABLES
%   stimulus_struct/finalevents_struct - loaded stimulus.mat/finalevents.mat structs
%   Nscr      - number of completed sub-screenings to loop over
%   stim_list - which stim IDs to include (pass unique(per_stim_all.stim_number)
%               for the session-wide map, or your usual stim_list for the
%               original scoped map)
%   omap      - output: containers.Map, stim ID -> real onset times (ms)
%   want      - Map used as an O(1) membership test for stim_list
%   i/pic_on/stim - sub-screening loop variables (index, onset grid, stimulus list)
%   j/stim_num/terna - per-stimulus loop variables (index, ID, onset-grid coordinates)
%   ts        - this stimulus's onset times extracted from pic_on this sub-screening
    omap = containers.Map('KeyType','double','ValueType','any');
    want = containers.Map(num2cell(stim_list), num2cell(true(size(stim_list))));

    for i = 1:Nscr
        pic_on = finalevents_struct.pics_onset{i};
        stim = stimulus_struct.stimulus{i};
        for j = 1:numel(stim)
            stim_num = stim(j).ID;
            if ~isKey(want, stim_num), continue; end
            terna = stim(j).terna_onset;
            if ndims(pic_on) == 3
                ts = pic_on(sub2ind(size(pic_on), terna(:,1), terna(:,2), terna(:,3)));
            else
                ts = pic_on(terna(:,1), terna(:,3));
                ts = ts(1:length(ts));
            end
            ts = ts(:).';
            if ~isKey(omap, stim_num), omap(stim_num) = []; end
            omap(stim_num) = [omap(stim_num), ts];
        end
    end
end
