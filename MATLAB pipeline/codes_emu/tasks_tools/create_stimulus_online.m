function stimulus = create_stimulus_online(order_pic, NISI, pics2use, names, ISI, order_ISI)
%creates stimulus struct using names and order pics
Npic=length(names);
stimulus = [];
for ipic=1:Npic
    % i is the place in the sequence, j is the ISI index, and k is seq number
    [i, j, k]=ind2sub(size(order_pic),find(order_pic==ipic));  
    terna = [i j k];
    for iISI=1:NISI
        stimulus((ipic-1)*NISI+iISI).ID = pics2use(ipic);   
        stimulus((ipic-1)*NISI+iISI).ISI = ISI(iISI);   
        stimulus((ipic-1)*NISI+iISI).name = names{ipic};
        [indISI, ~] = find(order_ISI==iISI);
        whichISItrials = terna(:,2)==iISI;
        stimulus((ipic-1)*NISI+iISI).terna_onset(:,1) = terna(whichISItrials,1);
        stimulus((ipic-1)*NISI+iISI).terna_onset(:,3) = terna(whichISItrials,3); %sequence number
        stimulus((ipic-1)*NISI+iISI).terna_onset(:,2) = indISI(terna(whichISItrials,3));
    end
end

end

