function [NREP, NSEQ, seq_length, estimated_duration] = calculate_miniscr_time(N)
    min_seq_length = 60;
    if N < 10
        error('Less than 10 images inside miniscr folder.')
    elseif N >= 10 && N < 12
        NREP = 18;
    elseif N >= 12 && N < 15
        NREP = 15;
    elseif N >= 15 && N <= 40 % 40 -> 5 mins
        NREP = 12;
    else 
        error('More than 40 images inside miniscr folder.')
    end

    Nrepxseq = ceil(min_seq_length/N);
    seq_length = N*Nrepxseq;

    NSEQ = NREP/Nrepxseq;

    estimated_duration = ((0.5*seq_length + 3 + 5)* NSEQ)/60; %3 sec for beginning and end blanks, 5 sec for inter sequence time       
end