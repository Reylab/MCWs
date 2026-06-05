function [spikes, detecctions] = remove_collisions(data, detecctions, chan_label, spikes, b_make_plots)                
    n_disp_secs = 10;
    n_disp_samples = n_disp_secs * 30000;
    y_uV_lim = 2000;
    if b_make_plots
        % plot detections before removing remove_collisions
        f1 = figure(1);
        %f1.Position = [100 100 540 400];
        clf
        for i = 1:numel(data)
            subplot(length(channels),1,i)
            x = 1:datacounter(i);
            y = data{i}(1:datacounter(i));
            % plot first n_disp samples
            plot(x(1:n_disp_samples), y(1:n_disp_samples))
            hold on
            y = zeros(size(y));
            y(detecctions(i)) = 1;
            x = x(1:n_disp_samples);
            y = y(1:n_disp_samples);
            isNZ = (~y==0);
            scatter(x(isNZ), y(isNZ), 'r*')
            title(chan_label{i})
            ylim([-y_uV_lim y_uV_lim])
            xticks(1:1:n_disp_secs)
        end

    end
    [detecctions,is_art] = remove_collisions_bundle(chan_label,detecctions);
    if b_make_plots
        % plot detections after removing remove_collisions
        f2 = figure(2);
        f2.Position = [100 100 540 400];
        clf
        for i = 1:numel(data)
            subplot(length(channels),1,i)
            x = 1:datacounter(i);
            y = data{i}(1:datacounter(i));
            % plot first n_disp samples
            plot(x(1:n_disp_samples), y(1:n_disp_samples))
            hold on
            y = zeros(size(y));
            y(detecctions(i)) = 1;
            x = x(1:n_disp_samples);
            y = y(1:n_disp_samples);
            isNZ = (~y==0);
            scatter(x(isNZ), y(isNZ), 'r*')
            title(chan_label{i})
            ylim([-y_uV_lim y_uV_lim])
            xticks(1:1:n_disp_secs)
        end
    end
    parfor i = 1:numel(data)
        if ~isempty(is_art{i})
            spikes{i} = spikes{i}(~is_art{i},:);
        end
    end
end