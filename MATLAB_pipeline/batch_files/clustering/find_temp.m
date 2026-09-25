function [clust_num temp auto_sort] = find_temp(tree,clu,par) %,spikes,ipermut)


min_clus = par.min_clus;

c_ov = par.c_ov;
elbow_min = par.elbow_min;
clu = clu(1:end-1,3:end)+1; %first dim temp

y = tree(1:end,5);
maxdiff = max(diff(tree(1:end,6:end)),[],2);
maxdiff(maxdiff<0)=0;
prop = (y(2:end)+maxdiff.*(maxdiff>0))./y(1:end-1);
aux = find(prop<elbow_min,1,'first')+1; %percentaje of the rest

% The next if removes the particular case where just a class is found at the 
% lowest temperature and with just a small change the rest appears
% all together at the next temperature
if ~isempty(aux) && par.mintemp==0 && aux==2
    aux = find(prop(2:end)<elbow_min,1,'first')+2; %percentaje of the rest
end

tree = tree(1:end-1,5:end);
dt = diff(tree);

% if sum(dt(1:end,:) > min_clus,'all') == 0 || sum(tree(:,2:end) >= min_clus,'all') == 0)
%     min_clus = min(max(dt,[],'all'),max(tree(:,2:end),[],'all') - 1;
% end

% only the reliable (pre-elbow) temperatures should be allowed to set
% min_clus -- rows at/after aux are already discarded as unreliable, so
% a spike there (e.g. the main cluster shattering) shouldn't set the bar
if ~isempty(aux) && aux > 2
    dt_reliable = dt(1:aux-2,:);
else
    dt_reliable = dt;
end

if sum(dt_reliable(:,2:end) > min_clus,'all') == 0
    min_clus = max(dt_reliable(:,2:end),[],'all') - 1;
end
clus = zeros(size(tree));
clus(tree(:,:) >= min_clus)=1; %only check the ones that cross the thr


clus = clus & [ones(size(clus(1,:)));dt(1:end,:)>min_clus];

for ii = 1:size(clus,1)
    detect = find(clus(ii,:),1,'last');
    if ~isempty(detect)
        clus(ii,1:detect)=1;
    end
end

auto_sort.elbow = size(tree,1);
if ~isempty(aux)
    clus(aux:end,1:end)=0;
    auto_sort.elbow = aux;
end
auto_sort.peaks = clus;

for ti = size(clus,1):-1:1
    detect = find(clus(ti,:));
    for ci = 1:length(detect) %the clusters removed aren't detected
        cl = (clu(ti,:) == detect(ci));
        for tj = ti-1:-1:1
            toremove = find(clus(tj,:));
            for j = 1:length(toremove)
                totest = (clu(tj,:) == toremove(j));
                if nnz(cl & totest)/min(nnz(totest),nnz(cl)) >= c_ov
                    clus(tj,toremove(j))=0;
                end
            end
        end
    end
end

[temp clust_num]=find(clus);
