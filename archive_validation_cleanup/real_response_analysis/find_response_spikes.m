function [inside,outside] = classify_class_spikes(class_num,grapes,times_cc,omap,stimlist,ch,pre,post)
    % class_num  - which class to score (e.g. 1)
    %   grapes     - loaded grapes_blanks.mat struct (ground truth raster)
    %   times_cc   - cluster_class loaded from the times_*.mat file, [class time_ms]
    %   omap       - containers.Map from build_onset_map, scoped to stimlist
    %   stimlist  - the stim IDs you're restricting to (must match what built omap)
    %   ch         - channel number (for the field name lookup)

    cf = ['chan' num2str(ch)];
    cls = ['class' num2str(class_num)];

    %%  verify times/grapes agreement
    recon_times = reconstruct_class_times(grapes.rasters.(cf).(cls).stim,omap);

    tt = sort(times_cc(times_cc(:,1) == class_num,2));
    hit = 0;
    j = 1; 
    nt = numel(tt);
    tol  = 0.5;   % ms -- 1:1 spike-time match tolerance (same convention used elsewhere in the repo)
    min_hit_frac = 0.70;
    for i = 1:numel(recon_times)
        rt = recon_times(i);
        while j<= nt && tt(j)< (rt-tol)
            j = j+1; 
        end
        if j <= nt && abs(tt(j) - rt) <= tol
            hit = hit +1;
        end
        hit_frac = hit / max(numel(recon_times), 1);
        times_matches_raster = hit_frac >= min_hit_frac;
    
        if times_matches_raster
            class_abs_times = times_cc(times_cc(:,1) == class_num, 2);
            fprintf('[classify] class%d: 1:1 time match confirmed (%.0f%% of %d recon spikes matched within %.1fms) -- using cluster_class directly.\n', ...
                    class_num, 100*hit_frac, numel(recon_times), tol);
        else
            class_abs_times = recon_times;
            fprintf('[classify] class%d: time match FAILED (%.0f%%, need %.0f%%) -- falling back to raster reconstruction (post-onset only, no outside-window coverage).\n', ...
                    class_num, 100*hit_frac, 100*min_hit_frac);
        end
    end
    %% onset times real
    all_onsets = [];
    for s = stimlist(:)'
        if isKey(omap,s)
            all_onsets = [all_onsets,omap(s)];
        end
    end

    all_onsets = sort(all_onsets);

    %% spike distance to onset
    n = numel(class_abs_times);
    is_inside = false(n,1);
    for i = 1:n
        t = class_abs_times(i);
        j = find(all_onsets <=t,1, 'last');
        cands = all_onsets(max(j-1,1):min(j+1,numel(all_onsets)));
        if any(t >=cands-pre & t<= cands+post)
            is_inside(i) = true;
        end
    end
    inside  = class_abs_times(is_inside);
    outside = class_abs_times(~is_inside);
    fprintf('[classify] class%d: %d inside window, %d outside (of %d total)\n', ...
            class_num, numel(inside), numel(outside), n);

    function n = raster_spike_count(stimc, stimlist)
        n = 0;
        for s = stimlist(:)'
            
            if s >(stimc) || isempty(stimc{s})
                continue
            end
            
            tr = stimc{s};
            
            if ~iscell(tr)
                tr = num2cell(tr,2)
            end

            for k = 1:numel(tr)
                v = tr{k}(:).';
                n = n + nnz(v~=9999);
            end
        end
    end

    function t = reconstruct_class_times(stimc, omap)
        t = [];
        for s = 1:numel(stimc)
            if isempty(stimc{s}) || ~isKey(omap, s), continue; end
            tr = stimc{s};
            if ~iscell(tr), tr = num2cell(tr,2); end
            O = omap(s);
            if numel(O) ~= numel(tr), continue; end  
            for k = 1:numel(tr)
                rel = tr{k}(:).';
                rel = rel(rel >= 0 & rel < 9000);      % post-onset only // blanks caveat
                if ~isempty(rel), t = [t, rel + O(k)]; end 
            end
        end
        t = sort(t(:));
    end
end

