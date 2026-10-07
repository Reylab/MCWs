function tpl = load_stored_template(template_file, ch, class_num)
% LOAD_STORED_TEMPLATE  Loads one class's wave_clus template waveform from a
% templates_wc_offline.mat file (as saved by Do_clustering.m /
% build_templates.m): cls_centers{idx} is a (max_class x nsamples) matrix
% for the channel at channels(idx), row class_num = that class's template.
%
% VARIABLES
%   template_file - input: path to a templates_wc_offline.mat
%   ch/class_num  - input: which channel/class to pull
%   tpl           - output: 1 x nsamples template waveform, or [] if the
%                   file/channel/class isn't available (with a warning)
    tpl = [];
    if isempty(template_file) || ~exist(template_file, 'file')
        return
    end
    S = load(template_file, 'cls_centers', 'channels');
    idx = find(S.channels == ch, 1);
    if isempty(idx)
        warning('load_stored_template: channel %d not found in %s', ch, template_file);
        return
    end
    centers = S.cls_centers{idx};
    if ~isnumeric(centers) || size(centers,1) < class_num
        warning('load_stored_template: class%d template not available for ch%d in %s', class_num, ch, template_file);
        return
    end
    tpl = centers(class_num, :);
end
