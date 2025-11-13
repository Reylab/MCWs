function [complete_times, text_out] = fix_onset_times(times,values,ntrial,experiment,scr_config)
	if ~exist('scr_config','var') || isempty(scr_config) scr_config=experiment;
	end
    complete_times = [];    
    text_out = {};
    ISI_ms = scr_config.ISI*1000;
    get_pic_event = @(ipic,ntrial) experiment.pic(mod(ipic,2)+1,ceil(3*ntrial/scr_config.Nseq));
    [seq_length,NISI,~] = size(scr_config.order_pic);

    inds_pics_onset = find(ismember(values,experiment.pic(:)));

    pics_times = times(inds_pics_onset); %I need the sign to make subtractions
    pics_values = values(inds_pics_onset);
    if isempty(pics_values)
        return
    end
    %<-- add remove spurious pics values 
    %I should check that the time differences between the pulses are the ones expected.
    if size(inds_pics_onset,1)== (seq_length *NISI)
        complete_times = pics_times;
        return
    end
    %% ...fixing lossed pics_onset...:
    
    %creates the first event 
    % (it requires that the rest are there, otherwise its not possible to be sure about the next)
    % a best alternative is using the expected time from other type  of event
    if pics_values(1) ~= get_pic_event(1,ntrial)
        if (pics_values(1) == get_pic_event(2,ntrial)) && (size(pics_values,1)+2) >= (seq_length *NISI)
            text_out{end+1} = sprintf('First pic_on event added on trial %d',ntrial);
            pics_values = [get_pic_event(1,ntrial); pics_values];
            pics_times = [pics_times(1)-ISI_ms;  pics_times];
        else
            text_out{end+1} = sprintf('Initial and more pic_on events lost on trial %d',ntrial);
            return
        end
    end
    
    %these repairs can be done better (FC)
    new_times = NaN(seq_length,1);
    new_times(1) = pics_times(1);
    ev_counter = 1; %used cbmex pics_on
    for j=2:seq_length
        aprox_time = new_times(j-1) + ISI_ms;% aprox time
        det = find(abs(pics_times - aprox_time) < ISI_ms *0.3);
        if ~isempty(det)
			if numel(det)==1 
				new_times(j) = pics_times(det);
				continue;
			else
				text_out{end+1} =  sprintf('Spurious event with pic value close to pic %d in %d', j,ntrial);
				[~, det]=min(abs(pics_times - aprox_time));
				new_times(j) = pics_times(det);
			end
        end
        prev = find(((aprox_time -pics_times)  < ISI_ms *1.3)& ((aprox_time -pics_times)  > ISI_ms *0.7));
        next  = find(((pics_times - aprox_time) < ISI_ms *1.3)& ((pics_times - aprox_time) > ISI_ms *0.7));
        
        if isempty(prev) && isempty(next)
            text_out{end+1} = sprintf('Multiple and consecutive events lost on trial: %d',ntrial);
            return
        elseif ~isempty(prev) && isempty(next)
            new_times(j) = pics_times(prev)+ ISI_ms;
        elseif isempty(prev) && ~isempty(next)
            new_times(j) = pics_times(next)- ISI_ms;
        else %best option both are there
            new_times(j) = (pics_times(prev)+pics_times(next))/2;
        end

    end
    complete_times = new_times;

end

