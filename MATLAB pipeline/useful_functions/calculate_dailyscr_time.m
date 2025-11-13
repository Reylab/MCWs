function [NSEQ, seq_length, estimated_duration] = calculate_dailyscr_time(N, NREP)
    min_seq_length = 60;

    Nrepxseq = ceil(min_seq_length/N);
    seq_length = N*Nrepxseq;

    NSEQ = NREP/Nrepxseq;

    estimated_duration = ((0.5*seq_length + 3 + 5)* NSEQ)/60; %3 sec for beginning and end blanks, 5 sec for inter sequence time       
end