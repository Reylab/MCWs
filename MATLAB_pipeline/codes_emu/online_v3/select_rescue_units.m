function grapes = select_rescue_units(grapes, mode)
% SELECT_RESCUE_UNITS  Reshape a grapes*_rescue.mat structure so downstream
% code that keys on 'mu' / 'classN' sees only ONE variant, under the
% canonical field names.
%
% do_structure_rescue_BCM_online3 writes, per channel:
%   mu, classN                          (post-rescue population)
%   mu_prerescue, classN_prerescue       (pre-rescue baseline)
%   classN_rescue                        (re-admitted spikes only)
%
% create_responses_data_parallel / plot_channel_grapes enumerate units with
% startsWith(x,{'mu','class'}), so all of the above would be plotted as
% separate units. This trims/renames them:
%
%   mode = 'all'        leave every field untouched
%          'post'       keep mu, classN                    (drop *_prerescue, *_rescue)
%          'prerescue'  mu_prerescue->mu, classN_prerescue->classN
%          'rescued'    classN_rescue->classN              (drop mu*, classN, classN_prerescue)
%
% Details entries are moved to match. Unknown fields pass through untouched.

    if nargin < 2 || isempty(mode), mode = 'post'; end
    mode = lower(char(mode));
    if strcmp(mode, 'all')
        return;
    end
    if ~ismember(mode, {'post','prerescue','rescued'})
        error('select_rescue_units: unknown mode "%s".', mode);
    end
    if ~isfield(grapes, 'rasters')
        return;
    end

    keep_details = {'sr', 'output_name', 'ch_label'};
    chs = fieldnames(grapes.rasters);
    n_empty = 0;

    for ci = 1:numel(chs)
        ch = chs{ci};
        R  = grapes.rasters.(ch);
        D  = struct();
        if isfield(R, 'details'), D = R.details; end

        newR = struct();
        newD = struct();
        for k = 1:numel(keep_details)
            if isfield(D, keep_details{k})
                newD.(keep_details{k}) = D.(keep_details{k});
            end
        end

        fn = fieldnames(R);
        for fi = 1:numel(fn)
            f = fn{fi};
            if strcmp(f, 'details'), continue; end
            tok = regexp(f, '^(mu|class\d+)(_prerescue|_rescue)?$', 'tokens', 'once');
            if isempty(tok)
                newR.(f) = R.(f);                 % passthrough unrecognised field
                continue;
            end
            unit = tok{1};
            suf  = tok{2};
            switch mode
                case 'post',      want = isempty(suf);
                case 'prerescue', want = strcmp(suf, '_prerescue');
                case 'rescued',   want = strcmp(suf, '_rescue');
            end
            if want
                newR.(unit) = R.(f);
                if isfield(D, f), newD.(unit) = D.(f); end
            end
        end

        newR.details = newD;
        grapes.rasters.(ch) = newR;

        has_unit = any(~cellfun('isempty', ...
            regexp(fieldnames(newR), '^(mu|class\d+)$', 'once')));
        if ~has_unit, n_empty = n_empty + 1; end
    end

    if n_empty > 0
        warning('select_rescue_units:noUnits', ...
                ['mode "%s": %d/%d channel(s) have no matching unit fields. ' ...
                 'Was grapes_rescue.mat built by the current ' ...
                 'do_structure_rescue_BCM_online3?'], mode, n_empty, numel(chs));
    end
end
