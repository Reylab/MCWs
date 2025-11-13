function [] = Compare_pipelines(after_collision,compare_minus_1,pipeline1,pipeline2,pipeline_type)
%Compare_pipelines(0,1,"/mnt/data0/Python_pipeline_results/python_ouput_results/-1 of patient7 EMU002/mRPRC07 raw_336_spikes.mat","/mnt/data0/Python_pipeline_results/python_ouput_results/patient7 EMU002/mRPRC07 raw_336_spikes.mat","python")
cd /home/hauderc/Documents/GitHub/full_processing_pipeline/output
py_file_list = dir('**/*spikes.mat');
cd /mnt/data0/sEEG_DATA/MCW-FH_007/EMU/EMU-002_subj-MCW-FH_007_task-RSVPdynamic_scr_run-01
mat_file_list = dir('**/*spikes.mat');
for i=1:length(mat_file_list)
    %mat_filepath = strcat(mat_file_list(i).folder,"/",mat_file_list(i).name);
    %py_filepath = strcat(py_file_list(i).folder,"/",py_file_list(i).name);
    mat_filepath = pipeline1;
    py_filepath = pipeline2;
    spikes_mat = load(mat_filepath);
    spikes_py = load(py_filepath);

    if ~after_collision
        if isfield(spikes_mat,'spikes_all')
            mat_spikes = spikes_mat.spikes_all;
            mat_index_array = spikes_mat.index_all;

        else
            mat_spikes = spikes_mat.spikes;
            mat_index_array = spikes_mat.index;

        end
        if isfield(spikes_py,'spikes_all')
            py_spikes = spikes_py.spikes_all;
            py_index_array = spikes_py.index_all;
        else
            py_spikes = spikes_py.spikes;
            py_index_array = spikes_py.index;
        end
    else
        mat_spikes = spikes_mat.spikes;
        mat_index_array = spikes_mat.index;
        py_spikes = spikes_py.spikes;
        py_index_array = spikes_py.index;
    end
    [mat_dir,mat_name,~] = fileparts(mat_filepath);
    [py_dir,py_name,~] = fileparts(py_filepath);
    mat_dir = char(mat_dir);
    py_dir = char(py_dir);
    A = regexp(mat_name,'\d*','Match');
    B = regexp(mat_name,'\S*','Match');
    label = cell2mat(B(1));
    channel = str2num(cell2mat(A(2)));
    %find associated spikes and not associated spikes
    associated_spikes = [];
    mat_not_associated = [];
    py_not_associated =[];
    ref_ms = 1.5;
    j=1;
    [C,ipy] = setdiff(py_index_array,mat_index_array);
    t_test = [];

    mat_nsx_filepath_ = "/mnt/data0/sEEG_DATA/MCW-FH_007/EMU/EMU-002_subj-MCW-FH_007_task-RSVPdynamic_scr_run-01/NSx.mat";%append(mat_dir,'/NSx.mat');
    metadata = load(mat_nsx_filepath_);
    NSx = metadata.NSx;
    chan_ID = [NSx.chan_ID];
    chanidx = find(chan_ID==channel);
    raw_data_filepath = append(NSx(chanidx).label,'_',num2str(channel),NSx(chanidx).ext);
    conversion = NSx(chanidx).conversion;
    dc = double(NSx(chanidx).dc);
    sr = double(NSx(chanidx).sr);

    associated_spikes = [];
    not_associated = [];
    %convert ms to samples
    mat_indexs = floor(mat_index_array * sr/1000);
    count = 0;
    for k=1:numel(py_index_array)
        idx = k;
        py_index = floor(py_index_array(idx) * sr/1000);
        ref_sample = floor(ref_ms * sr/1000);
        j = find(mat_indexs>py_index-ref_sample & mat_indexs<py_index+ref_sample);
        if numel(j)==1
            associated_spikes = [associated_spikes;[j idx]];
        elseif numel(j)==0
            py_not_associated = [py_not_associated; idx];
        else
            count = count+1;
            difference = zeros(numel(j),2);
            for w=1:numel(j)
                difference(w,:) = [j(w),abs(py_index-mat_indexs(j(w)))];
            end
            [~,k]=min(difference(:,2));
            imat = difference(k,1);
            associated_spikes = [associated_spikes;[imat,idx]];
        end
    end
    py_indexs = floor(py_index_array * sr/1000);
    for k=1:numel(mat_index_array)
        mat_index = floor(mat_index_array(k) * sr/1000);
        ref_sample = floor(ref_ms * sr/1000);
        j = find(py_indexs>mat_index-ref_sample+1 & py_indexs<mat_index+ref_sample);
        if numel(j) ==0
            mat_not_associated = [mat_not_associated; k];
        end
    end
    %root mean square claculation
    associated_rms_threshold = [];
    rms_wave_error = [];
    for k=1:numel(associated_spikes(:,1))
        mat_wave = mat_spikes(associated_spikes(k,1),:);
        py_wave = py_spikes(associated_spikes(k,2),:);
        wave_difference = mat_wave-py_wave;
        rms_wave_error = [rms_wave_error;rms(wave_difference)];
        %py_idx = floor(spkes_py.index(associated_spikes(k,2))/1000*30000);
        %mat_idx = floor(spkes_mat.index(associated_spikes(k,1))/1000*30000);
        %py_threshold_idx = floor(py_index_array(associated_spikes(k,2))/1000*30000/9000000)+1;
        %mat_threshold_idx = floor(spikes_mat.index(associated_spikes(k,1))/1000*30000/9000000)+1;
        %associated_rms_threshold = [associated_rms_threshold;[rms(spikes_mat.spikes(associated_spikes(k,1),:))/rms(spikes_mat.threshold(mat_threshold_idx)),rms(spikes_py.spikes(associated_spikes(k,2),:))/rms(spikes_py.threshold(py_threshold_idx))]];
    end
    f=figure
    vs = violinplot(rms_wave_error,cellstr("Spikes"),'ShowMean',true);
    ylabel('Root mean square error')
    title(sprintf("RMSE VS Associated Spikes (%s %d)",label,channel))
    exportgraphics(f,sprintf('/home/hauderc/Documents/GitHub/full_processing_pipeline/output/%s %d RMSE.png',label,channel))
    %interquartile range zoom
    f_zoom = figure
    vs_zoom = violinplot(rms_wave_error,cellstr("Spikes"),'ShowMean',true);
    range=quantile(rms_wave_error,[0,.975]);
    ylim(range)
    ylabel('Root mean square error')
    title(sprintf("RMSE VS Associated Spikes (%s %d)",label,channel))
    exportgraphics(f_zoom,sprintf('/home/hauderc/Documents/GitHub/full_processing_pipeline/output/%s %d RMSE_zoom.png',label,channel))
    if compare_minus_1
        f1 = fopen(raw_data_filepath,'r','l');
        initial_index = 1;
        fseek(f1,initial_index*2,'bof');
        x_raw = fread(f1,'int16=>double')*conversion+dc;
        fclose(f1);
        if pipeline_type == "python"
            py_par = spikes_py.par;
            s = py_par.process_info.sos;

            py_x = pyrunfile('/home/hauderc/Documents/GitHub/full_processing_pipeline/python_spikes.py','x', x_raw=x_raw,detect_order=py_par.detect_order,detect_fmin=py_par.detect_fmin,detect_fmax=py_par.detect_fmax,sr=sr,param=py_par,s=s);
            py_x = double(py_x);

            py_x_2 = pyrunfile('/home/hauderc/Documents/GitHub/full_processing_pipeline/python_spikes.py','x',x_raw=x_raw,detect_order=py_par.sort_order,detect_fmin=py_par.sort_fmin,detect_fmax=py_par.sort_fmax,sr=sr,param=py_par,s=s);
            py_x_2 = double(py_x_2);

            x = py_x';
            x_2 = py_x_2';
        elseif pipeline_type == "MATLAB"
            x = filt_signal(x_raw,mat_par.detect_order,mat_par.detect_fmin,mat_par.detect_fmax,mat_par.sr,mat_par);
            x_2 = filt_signal(x_raw,mat_par.sort_order,mat_par.sort_fmin,mat_par.sort_fmax,mat_par.sr,mat_par);

            py_x = x';
            py_x_2 = x_2';
        end
        quick_comparison_plot(rms_wave_error,associated_spikes,spikes_mat,spikes_py,x,py_x,x_2,py_x_2,1)
    end
    if ~isempty(mat_not_associated) || ~isempty(py_not_associated)
        f1 = fopen(raw_data_filepath,'r','l');
        initial_index = 1;
        fseek(f1,initial_index*2,'bof');

        x_raw = fread(f1,'int16=>double')*conversion+dc;
        fclose(f1);

        mat_par = spikes_mat.par;

        x = filt_signal(x_raw,mat_par.detect_order,mat_par.detect_fmin,mat_par.detect_fmax,mat_par.sr,mat_par);

        x_2 = filt_signal(x_raw,mat_par.sort_order,mat_par.sort_fmin,mat_par.sort_fmax,mat_par.sr,mat_par);

        py_par = spikes_py.par;
        s = py_par.process_info.sos;

        py_x = pyrunfile('/home/hauderc/Documents/GitHub/full_processing_pipeline/python_spikes.py','x', x_raw=x_raw,detect_order=py_par.detect_order,detect_fmin=py_par.detect_fmin,detect_fmax=py_par.detect_fmax,sr=sr,param=py_par,s=s);
        py_x = double(py_x);

        py_x_2 = pyrunfile('/home/hauderc/Documents/GitHub/full_processing_pipeline/python_spikes.py','x',x_raw=x_raw,detect_order=py_par.sort_order,detect_fmin=py_par.sort_fmin,detect_fmax=py_par.sort_fmax,sr=sr,param=py_par,s=s);
        py_x_2 = double(py_x_2);
    end

    mat_not_associated_rmse = [];
    mat_not_associated_rmse_threshold = [];
    ratios = [];
    for k=1:numel(mat_not_associated)
        ms_index = mat_index_array(mat_not_associated(k));
        indexes = [floor(ms_index * sr/1000)-21,floor(ms_index * sr/1000)+42];
        mat_not_associated_wave = [x(indexes(1):indexes(2))];
        threshold_idx = floor(indexes(1)/9000000)+1;
        mat_threshold = spikes_mat.threshold(threshold_idx);
        mat_not_associated_wave_threshold = mat_not_associated_wave/mat_threshold;
        mat_peak = min(mat_not_associated_wave);
        mat_ratio = mat_peak/mat_threshold;

        py_not_associated_wave = [py_x(indexes(1):indexes(2))]';
        threshold_idx = floor(indexes(1)/9000000)+1;
        py_threshold = spikes_py.threshold(threshold_idx);
        py_not_associated_wave_threshold = py_not_associated_wave/py_threshold;
        py_peak = min(py_not_associated_wave);
        py_ratio = py_peak/py_threshold;

        wave_difference = mat_not_associated_wave - py_not_associated_wave;
        wave_difference_threshold = mat_not_associated_wave_threshold - py_not_associated_wave_threshold;

        mat_not_associated_rmse = [mat_not_associated_rmse;rms(wave_difference)];
        mat_not_associated_rmse_threshold = [mat_not_associated_rmse_threshold;rms(wave_difference_threshold)];
        ratios = [ratios;mat_ratio/py_ratio];
    end
    %wo normalization to threshold
    if ~isempty(mat_not_associated_rmse)
        f_mat_detected = figure;
        vs_mat_detected = violinplot(mat_not_associated_rmse,cellstr('Spikes'));
        title(sprintf('RMSE Vs MATLAB Detected (%s %d)',label, channel))
        ylabel('Roor Mean Square Error')
        exportgraphics(f_mat_detected,sprintf('/home/hauderc/Documents/GitHub/full_processing_pipeline/output/%s %d mat_detected.png',label,channel))
        %w normalization to threshold
        f_mat_detected_threshold = figure;
        vs_mat_detected_threshold = violinplot(mat_not_associated_rmse_threshold,cellstr('Spikes'));
        title(sprintf('RMSE Vs MATLAB Detected w/Threshold Normilation (%s %d)',label,channel))
        ylabel('Root Mean Square Error')
        exportgraphics(f_mat_detected_threshold,sprintf('/home/hauderc/Documents/GitHub/full_processing_pipeline/output/%s %d mat_detected_threshold.png',label,channel))
        %supplemental violin plot
        f_mat_detected_threshold_ratio = figure;
        vs_mat_detected_threshold_ratio = violinplot(ratios,cellstr('Spikes'));
        title(sprintf('Peak-threshold Ratio Vs MATLAB Detected (%s %d)',label,channel))
        ylabel('Peak-threshold Ratio')
        exportgraphics(f_mat_detected_threshold_ratio,sprintf('/home/hauderc/Documents/GitHub/full_processing_pipeline/output/%s %d mat_detected_threshold_ratio.png',label,channel))
    end

    %py_not_assocaited
    py_not_associated_rmse = [];
    py_not_associated_rmse_threshold = [];
    ratios = [];
    for k=1:numel(py_not_associated)
        ms_index = py_index_array(py_not_associated(k));
        indexes = [floor(ms_index * sr/1000)-21,floor(ms_index * sr/1000)+42];
        py_not_associated_wave = [py_x(indexes(1):indexes(2))]';
        threshold_idx = floor(indexes(1)/9000000)+1;
        py_threshold = spikes_py.threshold(threshold_idx);
        py_not_associated_wave_threshold = py_not_associated_wave/py_threshold;
        py_peak = max(py_not_associated_wave);
        py_ratio = py_peak/py_threshold;

        mat_not_associated_wave = [x(indexes(1):indexes(2))];
        threshold_idx = floor(indexes(1)/9000000)+1;
        mat_threshold = spikes_mat.threshold(threshold_idx);
        mat_not_associated_wave_threshold = mat_not_associated_wave/mat_threshold;
        mat_peak = max(mat_not_associated_wave);
        mat_ratio = mat_peak/mat_threshold;

        wave_difference = py_not_associated_wave - mat_not_associated_wave;
        wave_difference_threshold  = py_not_associated_wave_threshold - mat_not_associated_wave_threshold;
        py_not_associated_rmse = [py_not_associated_rmse;rms(wave_difference)];
        py_not_associated_rmse_threshold = [py_not_associated_rmse_threshold;rms(wave_difference_threshold)];
        ratios = [ratios;py_ratio/mat_ratio];
    end
    if ~isempty(py_not_associated_rmse)
        %wo normalization to threshold
        f_py_detected = figure;
        violinplot(py_not_associated_rmse,cellstr('Spikes'));
        title(sprintf('RMSE Vs python Detected (%s %d)',label,channel))
        ylabel('Root Mean Square Error')
        exportgraphics(f_py_detected,sprintf('/home/hauderc/Documents/GitHub/full_processing_pipeline/output/%s %d py_detected.png',label,channel))
        %w normalization to threshold
        f_py_detected_threshold = figure;
        violinplot(py_not_associated_rmse_threshold,cellstr('Spikes'));
        title(sprintf('RMSE Vs python Detected w/Threshold Normilation (%s %d)',label,channel))
        ylabel('Root Mean Square Error')
        exportgraphics(f_py_detected_threshold,sprintf('/home/hauderc/Documents/GitHub/full_processing_pipeline/output/%s %d py_detedted_threshold.png',label,channel))
        %supplemental violin plot
        f_py_detected_threshold_ratio = figure;
        violinplot(ratios,cellstr('Spikes'));
        title(sprintf('Peak-threshold Ratio Vs python Detected (%s %d)',label,channel))
        ylabel('Peak-threshold Ratio')
        exportgraphics(f_py_detected_threshold_ratio,sprintf('/home/hauderc/Documents/GitHub/full_processing_pipeline/output/%s %d py_detected_threshold_ratio.png',label,channel))
    end
end

% for k = 1:numel(py_not_associated)
%     ms_index = spikes_mat.index(py_not_associated(k));
%     py_indexes = [py_indexes;floor(ms_index*sr/1000)-21,floor(ms_index*sr/1000)+42];
%     py_not_associated_waves = [py_not_associated_waves;py_x(py_indexes(k,1):py_indexes(k,2))];
%     py_not_associated_mins = [py_not_associated_mins;min(py_not_associated_waves(i,:))];
%     py_not_associated_maxes = [py_not_associated_maxes;max(py_not_associated_waves(i,:))];
% end
%
% mat_not_associated_rms_threshold = [];
% for i=1:numel(mat_not_associated)
%     mat_threshold_idx = floor(spikes_mat.index(associated_spikes(i,1))/1000*30000/9000000)+1;
%     mat_not_associated_rms_threshold = [mat_not_associated_rms_threshold;rms(spikes_mat.spikes(mat_not_associated(i),:))/rms(spikes_mat.threshold(mat_threshold_idx))];
% end
% py_not_associated_rms_threshold = [];
% for i=1:numel(py_not_associated)
%     py_threshold_idx = floor(py_index_array(associated_spikes(i,2))/1000*30000/9000000)+1;
%     py_not_associated_rms_threshold = [py_not_associated_rms_threshold;rms(spikes_py.spikes(py_not_associated(i),:))/rms(spikes_py.threshold(py_threshold_idx))];
% end
% %Differences between associated spikes
% associated_spikes_difference = [];
% for i=1:numel(associated_rms_threshold(:,1))
%     associated_spikes_difference = [associated_spikes_difference; abs(associated_rms_threshold(i,1)-associated_rms_threshold(i,2))];
% end
% associated_spikes_difference = sort(associated_spikes_difference);
% figure
% subplot(5,2,9:10)
% plot(1:numel(associated_spikes_difference),associated_spikes_difference)
% title("Difference between samples")
% %violin plot of associated spikes
% subplot(5,2,1:4)
% title('Associated Spikes')
% vs = violinplot(associated_rms_threshold,cellstr(["matlab","python"]),'ShowMean',true,'BoxColor',[0,0,0])
% %rank sum test between the two spike sets
% [p,h] = ranksum(associated_rms_threshold(1),associated_rms_threshold(2))
% %violin plot of matlab not associated spikes
% subplot(5,2,[5,7])
% title('Not Associated Spikes')
% vs1 = violinplot(mat_not_associated_rms_threshold,cellstr("matlab"),'ShowMean',true,'BoxColor',[0,0,0])
% subplot(5,2,[6,8])
% vs2 = violinplot(py_not_associated_rms_threshold,cellstr("python"),'ShowMean',true,'BoxColor',[0,0,0])
% %find index in raw and send raw data through a 4th order get 19 index
% %before and 43 after
% %matlab  code for mat_not_associated
% mat_nsx_filepath_ = append(mat_dir,'/NSx.mat');
% metadata = load(mat_nsx_filepath_);
% NSx = metadata.NSx;
% chan_ID = [NSx.chan_ID];
% chanidx = find(chan_ID==channel);
% raw_data_filepath = append(NSx(chanidx).label,'_',num2str(channel),NSx(chanidx).ext);
% conversion = NSx(chanidx).conversion;
% dc = NSx(chanidx).dc;
% sr = NSx(chanidx).sr;
% f1 = fopen(raw_data_filepath,'r','l');
% initial_index = 1;
% fseek(f1,initial_index*2,'bof'); %this moves to the end-lts of the file
%
% x_raw = fread(f1,'int16=>double')*conversion+dc;
% fclose(f1);
% [b,a] = ellip(4,0.1,40,[300,3000]*2/sr);
% x = filtfilt(b,a,x_raw);
%
% %convert ms to sample for index
% mat_not_associated_waves = [];
% mat_not_associated_mins = [];
% mat_not_associated_maxes = [];
% mat_indexes = [];
% for i=1:numel(mat_not_associated)
%     ms_index = spikes_mat.index(mat_not_associated(i));
%     mat_indexes = [mat_indexes;floor(ms_index * sr/1000)-21,floor(ms_index * sr/1000)+42];
%     mat_not_associated_waves = [mat_not_associated_waves;x(mat_indexes(i,1):mat_indexes(i,2))'];
%     mat_not_associated_mins = [mat_not_associated_mins;min(mat_not_associated_waves(i,:))];
%     mat_not_associated_maxes = [mat_not_associated_maxes;max(mat_not_associated_waves(i,:))];
% end
% %wrapper for python code
% py_x = pyrunfile('/home/hauderc/Documents/GitHub/full_processing_pipeline/python_spikes.py','x', x_raw=x_raw,sr=sr);
% py_x = double(py_x);
% py_not_associated_waves = [];
% py_not_associated_mins = [];
% py_indexes = [];
% py_not_associated_maxes = [];
% for i = 1:numel(py_not_associated)
%     ms_index = spikes_mat.index(py_not_associated(i));
%     py_indexes = [py_indexes;floor(ms_index*sr/1000)-21,floor(ms_index*sr/1000)+42];
%     py_not_associated_waves = [py_not_associated_waves;py_x(py_indexes(i,1):py_indexes(i,2))];
%     py_not_associated_mins = [py_not_associated_mins;min(py_not_associated_waves(i,:))];
%     py_not_associated_maxes = [py_not_associated_maxes;max(py_not_associated_waves(i,:))];
% end
% %compare min and maxes of not_associated_spikes
% known_pairs = intersect(mat_indexes,py_indexes,'rows');
% possible_pairs = [];
% std_py =  std(associated_rms_threshold);
% std_py = std_py(2);
% for i=1:numel(py_indexes(:,1))
%     mat_min_idx = find(mat_not_associated_mins<py_not_associated_mins(i)+5 & mat_not_associated_mins > py_not_associated_mins(i)-5);
%     mat_max_idx = find(mat_not_associated_maxes < py_not_associated_maxes(i)+5 & mat_not_associated_maxes > py_not_associated_maxes(i)-5);
%     for j = 1:numel(mat_min_idx)
%         if ismember(mat_min_idx(j),mat_max_idx)
%             %w = find(mat_max_idx == mat_min_idx(j));
%             %
%             if abs(mat_indexes(mat_min_idx(j))-py_indexes(i,1))<=64
%                 possible_pairs = [possible_pairs;mat_not_associated_waves(mat_min_idx(j),:),py_not_associated_waves(i,:)...
%                     ,mat_indexes(mat_min_idx(j),1),py_indexes(i,1)];
%             end
%         end
%     end
% end
% %wave plots
% avg_coherence = [];
% for i=1:numel(possible_pairs(:,1))
%     coherence = mscohere(possible_pairs(i,1:64),possible_pairs(i,65:128));
%     avg_coherence = [avg_coherence;mean(coherence)];
% end
% matched_waves = find(avg_coherence>.80);
% % for i=1:20
% %     figure
% %     plot(1:64,possible_pairs(matched_waves(i),1:64),1:64,possible_pairs(matched_waves(i),65:128))
% % end
% matched_waves = possible_pairs(matched_waves,:);
% rms_threshold_matched_waves = [];
% for i=1:numel(matched_waves(:,1))
%     mat_idx = matched_waves(i,129);
%     py_idx = matched_waves(i,130);
%     mat_threshold_idx = floor(mat_idx/9000000)+1;
%     py_threshold_idx = floor(py_idx/9000000)+1;
%     rms_threshold_matched_waves = [rms_threshold_matched_waves;rms(matched_waves(i,1:64))/rms(spikes_mat.threshold(mat_threshold_idx)),rms(matched_waves(i,65:128))/rms(spikes_py.threshold(py_threshold_idx))];
% end
% figure
% vs3 = violinplot(rms_threshold_matched_waves,cellstr(["matlab","python"]),'ShowMean',true,'BoxColor',[0,0,0])
% %matlab ranksum
% [mat_p,mat_h] = ranksum(rms_threshold_matched_waves(:,1),associated_rms_threshold(:,1))
% %python ranksum
% [py_p,py_h] = ranksum(rms_threshold_matched_waves(:,2),associated_rms_threshold(:,2))
    function filtered = filt_signal(x,order,fmin,fmax,sr,par)
        %HIGH-PASS FILTER OF THE DATA
        [b,a] = ellip(order,0.1,40,[fmin fmax]*2/sr);

        if par.preprocessing && ~isempty(par.process_info)
            [sos,g] = tf2sos(b,a);
            g = g * par.process_info.G;
            sos = [par.process_info.SOS; sos];
            filtered = fast_filtfilt(sos, g, x);
        else
            filtered = fast_filtfilt(b, a, x);
        end
    end
end