% clear all
% clc
load order_pics_RSVP_SCR

% ISI = [0.5];
ISI = sort(ISI);
NISI=numel(ISI);
seq_length = size(order_pic,1);
% Nrep=20;
Nseq = size(order_pic,3);

tmean_ch = 10;
nmean_ch = round(seq_length*ISI/tmean_ch);

% load(fullfile(pwd,'order_pics.mat'))  % use shuffle_rsvp to create a new one (current seq=10,nISI=3,nrep=40)
% if NISI>1
%     [~,order_ISI]=sort(rand(NISI,Nrep));
% else
%     order_ISI = ones(1,Nrep);
    order_ISI = ones(1,Nseq);
% end
cumISI = cumsum(ISI);

%BLANK > LINEON + LINE CHANGE
min_timech_blank = 0.3;
max_rand_timech_blank = 0.2;

colors = [[255 0 0];[0 255 0]];
col_vals = [[1 1];[1 2];[2 1];[2 2]];

%pics change between ISI/4 and 3*ISI/4

% nchanges_blank=NaN*ones(1,Nseq);
nchanges_blank=zeros(1,Nseq); % no changes in the blank
nchanges_pic=NaN*ones(1,Nseq);
lines_change = cell(1,Nseq);
% subcell 1 has another subcell where each row is a change. 
% col 1 is the blank number where it changes
% col 2 the time when it changes (in sec from the onset (previous times))
% col 3 is the color of top line
% col 4 is the color of bottom line

% subcell 2 has another subcell where each row is a change. 
% col 1 is the ISI (1 being the shortest)
% col 2 the time when it changes (in sec from the onset (previous times))
% col 3 is the color of top line
% col 4 is the color of bottom line
% col 5 is the pic in the sequence where the change takes place

for irep=1:Nseq
% %     nchanges_blank(irep) = randi(2);
%     if irep==1
%         nchanges_blank(irep)=1;
%         blank_changes=2;
%         lines_change{irep}{1}{1,1} = 2;
%         lines_change{irep}{1}{1,2} = min_timech_blank+max_rand_timech_blank*rand(1);
%     else
% %         nchanges_blank(irep) = randi(2)-1;
%         nchanges_blank(irep) = double(rand(1)>0.75);
% %         t2=randperm(NISI+1);
% %         blank_changes = sort(t2(1:nchanges_blank(irep)));
%         for j=1:nchanges_blank(irep)
% %             lines_change{irep}{1}{j,1} = blank_changes(j);
%             lines_change{irep}{1}{j,1} = 2;
%             lines_change{irep}{1}{j,2} = min_timech_blank+max_rand_timech_blank*rand(1);
%         end
%     end
    for k=1:4
        lines_change{irep}{1}{nchanges_blank(irep)+1,k}=NaN;
    end

%     nchanges_pic(irep) = randi(3)+1;
    nchanges_pic(irep) = nmean_ch+randi(3)-2;
    
%     ichange=1;
    t2=randperm(seq_length-2);
    t2=sort(t2(1:nchanges_pic(irep)))+1;
    for j=1:nchanges_pic(irep)
        lines_change{irep}{2}{j,1} = 1;
        lines_change{irep}{2}{j,2} = rand(1)*ISI(1)/2 + ISI(1)/4;
%         lines_change{irep}{2}{ichange,5} = t2(ichange)+1;
        lines_change{irep}{2}{j,5} = t2(j);
        lines_change{irep}{2}{j,6} = 1;
%         ichange=ichange+1;
    end    
    for k=1:6
%         lines_change{irep}{2}{ichange,k}=NaN;
        lines_change{irep}{2}{nchanges_pic(irep)+1,k}=NaN;
    end
end

blank_ord=1:2:NISI*2+1;
ISI_ord=2:2:NISI*2;


for irep=1:Nseq
%     ich=1;
%     ch_blank=[]; ch_pic=[];
%     for j=1:size(lines_change{irep}{1},1)-1
%         ch_blank = [ch_blank lines_change{irep}{1}{j,1}];
%         real_time(ich,1)=1; %blank change
%         real_time(ich,2)=j; %line change row (num change)
%         real_time(ich,3)=blank_ord(lines_change{irep}{1}{j,1}); %real time order
%         ich=ich+1;
%     end
%     for j=1:size(lines_change{irep}{2},1)-1
%         ch_pic = [ch_pic lines_change{irep}{2}{j,6}];
%         real_time(ich,1)=2; %pic change
%         real_time(ich,2)=j; %line change row (num change)
%         real_time(ich,3)=ISI_ord(lines_change{irep}{2}{j,6}); %real time order
%         ich=ich+1;
%     end
%     
%     [RT,indRT] = sort(real_time(:,3));
    
    ok=0;
    while ok==0
%         cols_seq = [randperm(4) randperm(4)];
        cols_seq=[];
        for gg=1:ceil(max(nchanges_pic)/4)+1
            cols_seq = [cols_seq randperm(4)];
        end
        if ~any(diff(cols_seq)==0)
            ok=1;
        end
    end
    color_start.up{irep} = colors(col_vals(cols_seq(1),1),:);
    color_start.down{irep} = colors(col_vals(cols_seq(1),2),:);
%     for kk=1:numel(indRT)
%         lines_change{irep}{real_time(indRT(kk),1)}{real_time(indRT(kk),2),3}=colors(col_vals(cols_seq(kk+1),1),:);
%         lines_change{irep}{real_time(indRT(kk),1)}{real_time(indRT(kk),2),4}=colors(col_vals(cols_seq(kk+1),2),:);
%     end
    for kk=1:size(lines_change{irep}{2},1)-1
        lines_change{irep}{2}{kk,3}=colors(col_vals(cols_seq(kk+1),1),:);
        lines_change{irep}{2}{kk,4}=colors(col_vals(cols_seq(kk+1),2),:);
    end
end
     
save 'variables_RSVP_SCR' 'ISI' 'seq_length' 'Nseq' 'Nrep' 'order_ISI' 'color_start' 'nchanges_blank' 'nchanges_pic' 'lines_change' 
   
fprintf('\n\n variables_RSVP_SCR has been created \n\n')
   