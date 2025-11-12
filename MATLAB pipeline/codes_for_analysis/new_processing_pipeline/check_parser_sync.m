function status = check_parser_sync(sync_file_1,sync_file_2, thr)
    %function to check if the parser could keep the synchronization between nsps
    % true is the 
    if ~exist('sync_file_1','var') || isempty(sync_file_1)
        sync_file_1 = 'RecordingSync_1272.NC5';
    end
    if ~exist('sync_file_2','var') || isempty(sync_file_2)
        sync_file_2 = 'RecordingSync_2272.NC5';
    end
    if ~exist('thr','var') || isempty(thr)
        thr = 1600;
    end
    nsp1=read_NCx(sync_file_1);
    nsp1 = nsp1>thr;

    nsp2=read_NCx(sync_file_2);
    nsp2 = nsp2>thr;
    if any(nsp2)== 0 && any(nsp1)== 0
        warning('Without synchronization pulses to compare')
    end
    status= sum(nsp1 ~= nsp2)==0;
    fprintf('Check sync with %s and %s: ',sync_file_1,sync_file_2)
    if status
         fprintf('SUCCESS.\n')
    else
        fprintf(2,'FAIL.\n')
    end
end
