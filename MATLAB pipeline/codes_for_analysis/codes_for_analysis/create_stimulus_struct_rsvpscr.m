function create_stimulus_struct_rsvpscr(experiment,Nseq_final)

% Nrep_final = experiment.Nrep;
Npic=length(experiment.ImageNames);
ISI = experiment.ISI;
% Nseq_final = experiment.Nseq;
order_ISI = experiment.order_ISI(:,1:Nseq_final);
NISI = numel(ISI);

for ipic=1:Npic
    [i j k]=ind2sub(size(experiment.order_pic(:,:,1:Nseq_final)),find(experiment.order_pic(:,:,1:Nseq_final)==ipic)); % i is the place in the sequence, j is the ISI index, and k is seq number 
    terna = [i j k];
    for iISI=1:NISI
        stimulus((ipic-1)*NISI+iISI).ID = ipic+iISI/10;   
        stimulus((ipic-1)*NISI+iISI).ISI = ISI(iISI);   
        stimulus((ipic-1)*NISI+iISI).name = experiment.ImageNames{ipic};
        [indISI indrep] = find(order_ISI==iISI);
%         for irep=1:numel(indrep)/2
            whichISItrials = terna(:,2)==iISI;
            stimulus((ipic-1)*NISI+iISI).terna_onset(:,1) = terna(whichISItrials,1);
            stimulus((ipic-1)*NISI+iISI).terna_onset(:,3) = terna(whichISItrials,3); %sequence number
            stimulus((ipic-1)*NISI+iISI).terna_onset(:,2) = indISI(terna(whichISItrials,3));
    end
end
save stimulus stimulus ISI order_ISI;
