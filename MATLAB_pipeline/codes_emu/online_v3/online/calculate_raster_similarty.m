function rasters_similarty =  calculate_raster_similarty(rastera,rasterb, delta_ms)
if ~exist('delta_ms','var') || isempty(delta_ms)
    delta_ms = 1;
end
total_events = 0;
matches = 0;

for i=1:numel(rastera)
    if (numel(rastera{i})==1 && rastera{i}==9999) && (numel(rasterb{i})==1 && rasterb{i}==9999)
        continue
    elseif (numel(rastera{i})==1 && rastera{i}==9999)
        total_events = total_events + numel(rasterb{i});
        continue
    elseif (numel(rasterb{i})==1 && rasterb{i}==9999)
        total_events = total_events + numel(rastera{i});
        continue
    end
    total_events = total_events + numel(rastera{i}) + numel(rasterb{i});
    
    times_concat = [rastera{i},rasterb{i}];
	membership = [zeros(1,numel(rastera{i})), ones(1,numel(rasterb{i}))];
    [times_concat_sorted , indices] = sort(times_concat);
    membership_sorted = membership(indices);
    diffs = diff(times_concat_sorted);
    inds = find(diffs <= delta_ms  & membership_sorted(1:end-1) ~= membership_sorted(2:end));

    if isempty(inds)
        continue
    end
    matches = matches + sum(inds(1:end-1) + 1 ~= inds(2:end)) + 1;
end
if total_events==0
    rasters_similarty = 1;
else
    rasters_similarty = matches/(total_events-matches);
end
