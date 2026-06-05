function [complete_times, text_out] = fix_photodiode_times(times,ntrial,experiment)
    complete_times = [];    
    text_out = '';
    if isempty(times)
       return 
    end
    ISI_ms = experiment.ISI*1000;
    
    [seq_length,NISI,~] = size(experiment.order_pic);
    %first try the easy way
    aux = [1;find(diff(times)>(ISI_ms*0.9))+1];
    if length(aux)==seq_length
        complete_times = times(aux);
        text_out = 'no changes needed';
        return 
    end
    APROX_SEARCH = 50; %in ms
    %these repairs can be done better (FC)
    new_times = NaN(seq_length,1);
    new_times(1) = times(1); %The first deteccion is always fine
    for j=2:seq_length
        aprox_time = new_times(j-1) + ISI_ms;% aprox time
        det = find(abs(times - aprox_time) < APROX_SEARCH);
        if ~isempty(det)
            if length(det)>1
                det = det(1); %maybe use the one closest to aprox_time?
            end
            new_times(j) = times(det);
            continue;
        end
        prev = find(((aprox_time -times)  < ISI_ms *1.2)& ((aprox_time -times)  > ISI_ms *0.7));
        next  = find(((times - aprox_time) < ISI_ms *1.2)& ((times - aprox_time) > ISI_ms *0.7));
        
        if isempty(prev) && isempty(next)
%             text_out{end+1} = sprintf('Multiple and consecutive events lost on trial: %d',ntrial);
            text_out = sprintf('Multiple and consecutive events lost on trial: %d',ntrial);
            return
        elseif ~isempty(prev) && isempty(next)
            new_times(j) = times(prev(1))+ ISI_ms;
        elseif isempty(prev) && ~isempty(next)
            new_times(j) = times(next(1))- ISI_ms;
        else %best option both are there
            new_times(j) = (times(prev(1))+times(next(1)))/2;
        end
        text_out = sprintf('Times repaired on trial: %d',ntrial);

    end
    complete_times = new_times;

end

