function [resp, reac_time] = response_check(f_key, h_key, resp_start, kb_id)

[pressed, first_press, ~, ~, ~] = KbQueueCheck(kb_id);
keys_hit = KbName(first_press);

if pressed
    %if multiple responses were recorded, we'll take the first one
    f_resp = first_press(f_key);
    h_resp = first_press(h_key);
    nc_shift_key = KbName('9(');
    shift_resp = first_press(nc_shift_key);
    
    if f_resp && h_resp
        if f_resp < h_resp
            resp = 1;
            reac_time = f_resp-resp_start;
        else
            resp = 2;
            reac_time = h_resp-resp_start;
        end
    elseif f_resp
        resp = 1;
        reac_time = f_resp - resp_start;
    elseif h_resp
        resp = 2;
        reac_time = h_resp-resp_start;
    elseif strcmpi(keys_hit, 'ESCAPE')
        ShowCursor;
        Priority(0);
        ListenChar(1);
        sca;
        return
    elseif shift_resp
        resp = 3;
        reac_time = shift_resp - resp_start;
    else
        resp = 0;
        reac_time = 0;
    end
    KbQueueFlush(kb_id);
else
    resp = 0;
    reac_time = 0;
end

end

