function grapes = update_grapes_with_blank_on_spikes(blank_on_onset, grapes, spikes_all_channels, ...
                                                     cluster_labels_all_channels, channels)
    for chi=1:length(channels)
        chanN = ['chan' num2str(channels(chi))];
        classes = fieldnames(grapes.rasters.(chanN));
        for cls = 1:numel(classes)
            class = classes{cls};
            if strcmp(class, 'details')
                continue
            end
            
            grapes.rasters.(chanN).(class).blank_on_spikes = cell(size(blank_on_onset));
            grapes.rasters.(chanN).(class).blank_on_onset = blank_on_onset;
            ch_spikes = spikes_all_channels{chi};
            for subscr_idx =1:length(blank_on_onset)
                subscr_blanks = blank_on_onset{subscr_idx};
                subscr_spikes = cell(size(subscr_blanks));
                for seq = 1:length(subscr_blanks)
                    time_blank_on = subscr_blanks{seq};
                    ind_spikes = (ch_spikes>=time_blank_on(1)) & (ch_spikes<=time_blank_on(2));
                    if strcmp(class, 'mu')                        
                        spikes = ch_spikes(ind_spikes);
                    else
                        spikes = ch_spikes(cluster_labels_all_channels{chi}==str2num(class(end)) & ind_spikes);
                    end
                    if isempty(spikes)
                        spikes = [9999];
                    end
                    subscr_spikes{seq} = spikes;
                end
                grapes.rasters.(chanN).(class).blank_on_spikes{subscr_idx} = subscr_spikes;
            end
        end
    end
end

