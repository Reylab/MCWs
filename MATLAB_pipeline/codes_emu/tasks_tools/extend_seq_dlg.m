function [b_extend_seq] = extend_seq_dlg(extra_pics, extra_time)
b_extend_seq = false;
quest = sprintf(['Additional pics: %d, this can add %.2f seconds to the next subscreening. ' ...
                 'Keep additional pics?'],extra_pics, extra_time);
dlgtitle = 'Extend sequence?';
btn1 = 'Yes';
btn2 = 'No';
defbtn = 'No';
answer = questdlg(quest,dlgtitle,btn1,btn2,defbtn);
switch answer
    case 'Yes'
        disp('Extending sequence')
        b_extend_seq = true;
    case 'No'
        disp('Removing pics to keep the same sequence length')
        b_extend_seq = false;
end

end

