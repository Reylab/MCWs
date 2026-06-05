function done = is_trial_done_pd(times,experiment)
    if isempty(times)
       done = false;
       return 
    end    
    ISI_ms = experiment.ISI*1000;
    bursts = [1;find(diff(times)>(ISI_ms*0.5))+1];

    if length(bursts)==experiment.seq_length 
       done = true;
       return 
    end
    
    last_time = times(1);
    for j=2:experiment.seq_length
        aprox_time = last_time + ISI_ms;% aprox time
        det = find(abs(times - aprox_time) < ISI_ms *0.3);
        if ~isempty(det)
            if length(det)>1
                det = det(1); %maybe use the one closest to aprox_time?
            end
            last_time = times(det);
            continue;
        end
        prev = find(((aprox_time -times)  < ISI_ms *1.2)& ((aprox_time -times)  > ISI_ms *0.7));
        next  = find(((times - aprox_time) < ISI_ms *1.2)& ((times - aprox_time) > ISI_ms *0.7));
        
        if isempty(prev) && isempty(next)
            done = false;
            return
        elseif ~isempty(prev) && isempty(next)
            last_time = times(prev(1))+ ISI_ms;
        elseif isempty(prev) && ~isempty(next)
            last_time = times(next(1))- ISI_ms;
        else %best option both are there
            last_time = (times(prev(1))+times(next(1)))/2;
        end

    end
    done = true;

end