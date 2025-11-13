function scr_config = shuffle_rsvpSCR_online3(Nrep,Npics,subtask,ImageNames,sorted_lists,n_scr,ini,fin)

ISI = 0.5;
if contains(subtask,'DynamicSeman') || contains(subtask,'CategLocaliz')
% if contains(subtask,'DynamicSeman')
    colors = [[0 0 0];[0 0 0]];
else 
    colors = [[255 0 0];[255 255 0]];
end

if numel(ISI)~=1
    error('This code is meant to be used with a single ISI value\n')
end    

min_seq_length = ceil(30/ISI); % 30 secs is the minimum duration per trial (the actual length will be between min_seq_length and 2*min_seq_length)

% if contains(subtask,'DynamicSeman')
%     Npics = 300;
% end

if Npics<min_seq_length
    Nrepxseq = ceil(min_seq_length/Npics);
    seq_length = Npics*Nrepxseq;
    Nseq = Nrep/Nrepxseq;
    rep_step = Nrepxseq;
    if mod(Nseq,1)~=0
        warning('Cannot use Nrep=%d. Using',Nrep,ceil(Nseq)*Nrepxseq)
        Nrep = ceil(Nseq)*Nrepxseq;
        Nseq = Nrep/Nrepxseq;
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


if contains(subtask,'DynamicSeman') 
        if n_scr == 1
            for mn = 1:Nseq
                selLists{mn} = sorted_lists(mn);
            end 
        elseif n_scr == 2
            for mn = 21:21+Nseq
                selLists{mn} = sorted_lists(mn);
            end 
        else
            for mn = 36:50
                selLists{mn} = sorted_lists(mn);
            end 
        end



    order_pic = NaN(seq_length,1,Nseq);
%     for i = 1:Nrep
%         fileName = selectedLists(i).name;
%         folderPath ='/home/user/share/experimental_files/pics/pics_space/seman_lists';
%         fullFilePath = fullfile(folderPath, fileName);
% 
%         tableData = readcell(fullFilePath);
%         if length(tableData) > 300
%             tableData(1) = [];
%         end
%AQUI TENGO QUE HACER UNA DIVISION ENTRE LAS LISTAS PARA CADA BLOQUE 
%ahora solo esta haciendo las primeras listas
if n_scr == 1
    f_list = 1;
    l_list = 20;
elseif n_scr == 2 
    f_list = 21;
    l_list = 35;
else 
    f_list = 36;
    l_list = 50;
end 

for i = f_list:l_list
    
    fileName = char(sorted_lists(i));
    folderPath ='/home/user/share/experimental_files/pics/pics_space/seman_lists';
    fullFilePath = fullfile(folderPath, fileName);

    tableData = readcell(fullFilePath);
    if length(tableData) > 60
             tableData(1) = [];
        end
        uniqueList=unique(tableData);
        sList = sort(uniqueList); 
        numItems = numel(tableData);
        indexedList = zeros(numItems, 1);

        for m= 1:length(tableData) 
%             [~, idx] = ismember(ImageNames.name, tableData{m});
            indexedList(m) = find(contains(ImageNames.name, tableData{m}));
        end

        iseq=0;
            stims_seq = [];

                all_stims = indexedList;

                for j=1:60
                    if n_scr == 1
                        order_pic(j,1,i) = all_stims(j);
                    elseif n_scr == 2
                        order_pic(j,1,i-20) = all_stims(j);
                    else 
                        order_pic(j,1,i-35) = all_stims(j);
                    end


                end

    end
else
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
end


ISI = sort(ISI);
seq_length = size(order_pic,1);
Nseq = size(order_pic,3);

tmean_ch = 10;
nmean_ch = round(seq_length*ISI/tmean_ch);

order_ISI = ones(1,Nseq);

col_vals = [[1 1];[1 2];[2 1];[2 2]];

%pics change between ISI/4 and 3*ISI/4

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
    for k=1:4
        lines_change{irep}{1}{nchanges_blank(irep)+1,k}=NaN;
    end

    nchanges_pic(irep) = nmean_ch+randi(3)-2;
    t2=randperm(seq_length-2);
    t2=sort(t2(1:nchanges_pic(irep)))+1;
    for j=1:nchanges_pic(irep)
        lines_change{irep}{2}{j,1} = 1;
        lines_change{irep}{2}{j,2} = rand(1)*ISI(1)/2 + ISI(1)/4;
        lines_change{irep}{2}{j,5} = t2(j);
        lines_change{irep}{2}{j,6} = 1;
    end    
    for k=1:6
        lines_change{irep}{2}{nchanges_pic(irep)+1,k}=NaN;
    end
end

for irep=1:Nseq
    ok=0;
    while ok==0
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
    for kk=1:size(lines_change{irep}{2},1)-1
        lines_change{irep}{2}{kk,3}=colors(col_vals(cols_seq(kk+1),1),:);
        lines_change{irep}{2}{kk,4}=colors(col_vals(cols_seq(kk+1),2),:);
    end
end

scr_config = struct;
scr_config.order_pic = order_pic;
scr_config.ISI=ISI;
scr_config.seq_length=seq_length;
scr_config.Nseq=Nseq;
scr_config.Nrep=Nrep;
scr_config.order_ISI=order_ISI;
scr_config.color_start=color_start;
scr_config.nchanges_blank=nchanges_blank;
scr_config.nchanges_pic=nchanges_pic;
scr_config.lines_change=lines_change;
