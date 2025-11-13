function quick_comparison_plot(rms_wave_error,associated_spikes,spikes_mat,spikes_py,x,py_x,x_2,py_x_2,same_pipeline)
%quick_comparison_plot(rms_wave_error,associated_spikes,spikes_mat,spikes_py,x,py_x,x_2,py_x_2,0)
%quick_comparison_plot(rms_wave_error,associated_spikes,spikes_mat,spikes_py,x,py_x,x_2,py_x_2,1)
I = find(18<rms_wave_error); 
idxs = associated_spikes(I,:);

mat_wave = [];
py_wave =[];
py_sample_index = [];
mat_sample_index = [];
filtered_idxs = [];
filtered_data = [];
py_previous_sample_index = [];


if isfield(spikes_mat,'spikes_all')
    mat_spikes = spikes_mat.spikes_all;
    mat_index_array = spikes_mat.index_all;
else
    mat_spikes = spikes_mat.spikes;
    mat_index_array = spikes_mat.index;
end

for i=1:numel(idxs(:,1))
    mat_wave = [mat_wave; mat_spikes(idxs(i,1),:)];
    py_wave = [py_wave;spikes_py.spikes(idxs(i,2),:)];
    py_time_index = spikes_py.index(idxs(i,2));
    mat_time_index = mat_index_array(idxs(i,1));
    py_sample_index = [py_sample_index;floor(py_time_index/1000*30000)];
    mat_sample_index = [mat_sample_index;floor(mat_time_index/1000*30000)];
    filtered_idxs = [filtered_idxs;mat_sample_index(i)-300,mat_sample_index(i)+300];
    mat_data = x(filtered_idxs(i,1):filtered_idxs(i,2))';
    py_data = py_x(filtered_idxs(i,1):filtered_idxs(i,2));
    mat_data_2 = x_2(filtered_idxs(i,1):filtered_idxs(i,2))';
    py_data_2 = py_x_2(filtered_idxs(i,1):filtered_idxs(i,2));
    filtered_data = [filtered_data;mat_data,py_data,mat_data_2,py_data_2];

%     py_previous_wave = [py_wave;spikes_py.spikes(idxs(i,2)-1,:)];
%     py_previous_time_index = spikes_py.index(idxs(i,2)-1);
%     py_previous_sample_index = [py_previous_sample_index;floor(py_previous_time_index/1000*30000)];
%     py_previous_data = py_x(filtered_idxs(i,1):filtered_idxs(i,2));

    mat_threshold_idx = floor(mat_sample_index(i)/9000000)+1;
    py_threshold_idx = floor(py_sample_index(i)/9000000)+1;
    mat_threshold = spikes_mat.threshold(mat_threshold_idx);
    py_threshold = spikes_py.threshold(py_threshold_idx);

    
    figure
    plot(1:numel(filtered_data(i,1:601)),filtered_data(i,1:601),1:numel(filtered_data(i,602:1202)),filtered_data(i,602:1202)...
        ,1:numel(filtered_data(i,1203:1803)),filtered_data(i,1203:1803),1:numel(filtered_data(i,1804:2404)),filtered_data(i,1804:2404))
    hold on
%     mat_line = ones(1,601).*-mat_threshold;
%     py_line =ones(1,601).*-py_threshold;
%     plot(1:601,mat_line,'-b')
%     plot(1:601,py_line,':')
   yline(-mat_threshold,'-b');
   yline(-py_threshold,':');
    xline(300,'-b')
    xlim([280 350])
    
    if py_sample_index(i)>mat_sample_index(i)
        xline(300+py_sample_index(i)-mat_sample_index(i),':')
        %xline(301+py_previous_sample_index(i)-mat_sample_index(i),'-')

    elseif py_sample_index(i)<mat_sample_index(i)
        xline(300-mat_sample_index(i)+py_sample_index(i),':')
        %xline(301+mat_sample_index(i)-py_previous_sample_index(i),'-')
    else
        xline(300,':')
        %xline(301,'-')
    end
    title("filter plots")
    if ~same_pipeline
        legend('matlab 4th order','python 4th order','matlab 2nd order','python 2nd order')
    else
        legend('pipeline1 4th order','pipeline2 4th order','pipeline1 2nd order','pipeline2 2nd order')
    end
    hold off
    figure
    hold on
    plot(1:numel(mat_wave(i,:)),mat_wave(i,:),1:numel(py_wave(i,:)),py_wave(i,:))
    title('plot of interpolation')
    hold off
    disp(mat_sample_index(i)-(mat_threshold_idx-1)*9000000)
    disp(py_sample_index(i)- (py_threshold_idx-1)*9000000-1)
end

