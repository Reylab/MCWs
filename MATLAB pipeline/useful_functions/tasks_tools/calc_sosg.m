function [sos,g] = calc_sosg(notches, z_det,p_det,K,MAX_NOTCHES)
        P = p_det;
        Z = z_det;   
        if ~isempty(notches)
            [~, ix_notch] = sort(notches.abs_db,'descend');
            n_notches = min(MAX_NOTCHES, length(notches.abs_db));

            for ni = 1:n_notches
                zpix = ix_notch(ni)*2+(-1:0);
                Z(end+1:end+2) = notches.Z(zpix);
                P(end+1:end+2) = notches.P(zpix);
                K = K *notches.K(ix_notch(ni));
            end
        end
        [sos,g] = zp2sos(Z,P,K);
end