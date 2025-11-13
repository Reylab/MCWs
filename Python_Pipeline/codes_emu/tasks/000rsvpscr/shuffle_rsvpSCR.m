function shuffle_rsvpSCR(Nrep,Path_pics)
clc

% Path_pics=[];

% Nrep = 15;
% Nrep = 2;
ISI = 0.5;

if ~exist('Path_pics','var') || isempty(Path_pics),    Path_pics=[pwd '_pic']; end

if numel(ISI)~=1
    error('This code is meant to be used with a single ISI value\n')
end    

min_seq_length = ceil(30/ISI); % 30 secs is the minimum duration per trial (the actual length will be between min_seq_length and 2*min_seq_length)

a=dir(sprintf('%s%c*',Path_pics,filesep));
b = zeros(length(a),1);
b = b>0;
names_to_fix = false;
for i = 1:length(a)
%     b(i) = ~isempty(strfind (lower(a(i).name), '.jp'));
    b(i) = contains (lower(a(i).name), '.jp');
    if b(i) && a(i).name(1)==' '
        names_to_fix = true;
        warning('Leading spaces inf file: %s%c%s',Path_pics,filesep,a(i).name)
    end
end
if names_to_fix
    error('fix image naming');
end
a = a(b);
if isempty(a)
    error(['No pictures for this session in ' Path_pics]);
end
Npics=length(a);
    
% Npics=15;
% if Npics<=20
% %     Nrep = 24; % for miniRSVP with Npics=20?
%     Nrep = 15;
% else
%     Nrep = 15;
% end

if Npics<min_seq_length
    Nrepxseq = ceil(min_seq_length/Npics);
    seq_length = Npics*Nrepxseq;
    Nseq = Nrep/Nrepxseq;
    rep_step = Nrepxseq;
    if mod(Nseq,1)~=0
        error('Cannot use Nrep=%d. Select Nrep=%d or Nrep=%d\n',Nrep,floor(Nseq)*Nrepxseq,ceil(Nseq)*Nrepxseq)
    end
else
    Nseqxrep = floor(Npics/min_seq_length);
    seq_length = floor(Npics/Nseqxrep);
    Nseq = Nseqxrep * Nrep;
    rep_step = 1;
    if seq_length*Nseqxrep<Npics
        error('Cannot use %d pics. Please delete some and take it down do %d pics\n',Npics,seq_length*Nseqxrep)
    end
end

estimated_duration = ((ISI*seq_length + 3 + 5)* Nseq)/60; %3 sec for beginning and end blanks, 5 sec for inter sequence time

fprintf('Estimated duration of the experiment with %d pics (%d trials each) in %d sequences of %2.1f secs: %1.1f min\n',Npics,Nrep,Nseq,seq_length*ISI,estimated_duration)

keyval = input('If this is fine by you, press y to continue; any other key will terminate this program  ','s');
% keyval='y';

if strcmpi(keyval,'y')
    order_pic = NaN(seq_length,1,Nseq);
    valid=1;
    while valid==1
        iseq=0;
        for i=1:rep_step:Nrep
            while valid==1
                stims_seq = [];
                if exist('Nrepxseq','var')
                    for j=1:Nrepxseq
                        stims_seq = [stims_seq ; randperm(Npics)'];
                    end
                    iseq=iseq+1;
                    order_pic(:,1,iseq) = stims_seq;
                else
                    all_stims = randperm(Npics)';
                    for j=1:Nseqxrep
                        order_pic(:,1,(i-1)*Nseqxrep+j) = all_stims(1+seq_length*(j-1):seq_length*j);
                    end
                end
                T=order_pic(:);
                if ~any(diff(T(~isnan(T)))==0)
                    break;
                else
                    iseq=iseq-1;
                end
            end
        end
        T=order_pic(:);
        if ~any(diff(T(~isnan(T)))==0), break; end
    end     
else
%     return
    error('Shuffle aborted. Revise and do it again')
end

save('order_pics_RSVP_SCR','order_pic','ISI','Nrep','estimated_duration','seq_length') 

fprintf('\n\n order_pics_RSVP_SCR has been created \n\n')

create_lines_change_RSVP_SCR    
