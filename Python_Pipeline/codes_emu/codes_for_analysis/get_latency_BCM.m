function [good_lat,onset,dura]= get_latency_BCM(IFR,ejex,sample_period,IFR_thr,t_down,over_threshold_time,below_threshold_time)

t_min_lat = 80;
t_max_lat = 1000;
onset = NaN; dura = NaN; good_lat = false;

times_ind_post = find(ejex>t_min_lat,1)-1:1:find(ejex>t_max_lat,1);
timeline = IFR(times_ind_post);
over_threshold = timeline > IFR_thr;
over_th_shift = circshift(over_threshold,[0 1]); % the first component should be forced to 0 to avoid problems if the last one is 1
over_th_shift(1)=0;
over_threshold(1)=0;
crossing_points = over_threshold - over_th_shift;
up_crossing_index = find(crossing_points == 1);
down_crossing_index = find(crossing_points == -1);
overthreshold_interval_index = [];

if ~isempty(up_crossing_index)
    if isempty(down_crossing_index)
        down_crossing_index = length(crossing_points);
    end
    if down_crossing_index(1)<up_crossing_index(1)
        down_crossing_index = down_crossing_index(2:end);
    end
    if length(up_crossing_index) == length(down_crossing_index) + 1    % if length(up_crossing_index) = length(down_crossing_index) + 1, it means that the last interval is length(crossing_points) + 1 - up_crossing_index(end)
        if length(up_crossing_index) > 1
            overthreshold_interval_index = down_crossing_index - up_crossing_index(1:end-1);
        end
        overthreshold_interval_index = [overthreshold_interval_index,length(crossing_points)-up_crossing_index(end)];
        down_crossing_index = [down_crossing_index length(crossing_points)];
        if length(up_crossing_index) > 1
            saca = (up_crossing_index(2:end)-down_crossing_index(1:end-1))*sample_period<t_down;
            down_crossing_index(saca)=[];
            up_crossing_index(find(saca)+1)=[];
            overthreshold_interval_index = down_crossing_index - up_crossing_index;
        end
        overthreshold_interval_first = find (overthreshold_interval_index >= (over_threshold_time/sample_period),1);
    else
        overthreshold_interval_index = down_crossing_index - up_crossing_index;
        if length(up_crossing_index) > 1
            saca = (up_crossing_index(2:end)-down_crossing_index(1:end-1))*sample_period<t_down;
            down_crossing_index(saca)=[];
            up_crossing_index(find(saca)+1)=[];
            overthreshold_interval_index = down_crossing_index - up_crossing_index;
        end
        overthreshold_interval_first = find (overthreshold_interval_index >= (over_threshold_time/sample_period),1);
    end
else
    overthreshold_interval_first = [];
end

if ~isempty(overthreshold_interval_index)
    ups = ejex(times_ind_post(up_crossing_index));
    durations = overthreshold_interval_index*sample_period;
else
    ups = -1;
    durations = -1;
end

if isempty(overthreshold_interval_first)
    [~,ss]=max(durations);
    if durations~=-1
        onset = ups(ss);
        dura = durations(ss);
    end
else
    onset= ups(overthreshold_interval_first);
    off_ind = overthreshold_interval_first;
    cc=overthreshold_interval_first+1;
    while cc<=length(ups)
        if ups(cc) - (durations(cc-1)+ups(cc-1)) < below_threshold_time
            off_ind = off_ind + 1;
            cc = cc+1;
        else
            cc=length(ups)+1;
        end
    end
    dura = durations(off_ind) + ups(off_ind) - onset;
    good_lat = true;
end
