D = '/mnt/data0/python_p10EMU07CH368/times_2026-10-06_13-46-07';
tree = load(fullfile(D, 'data_mRFUS07 raw_368.dg_01'));
clu  = load(fullfile(D, 'data_mRFUS07 raw_368.dg_01.lab'));
T = load(fullfile(D, 'times_mRFUS07 raw_368.mat'), 'par');
par = T.par;
fprintf('par: min_clus=%g c_ov=%g elbow_min=%g mintemp=%g maxtemp=%g tempstep=%g\n', par.min_clus, par.c_ov, par.elbow_min, par.mintemp, par.maxtemp, par.tempstep);
fprintf('tree %s, clu %s\n', mat2str(size(tree)), mat2str(size(clu)));
% --- replay find_temp step by step ---
y = tree(:,5);
maxdiff = max(diff(tree(:,6:end)),[],2); maxdiff(maxdiff<0)=0;
prop = (y(2:end)+maxdiff.*(maxdiff>0))./y(1:end-1);
aux = find(prop<par.elbow_min,1,'first')+1;
if ~isempty(aux) && par.mintemp==0 && aux==2, aux = find(prop(2:end)<par.elbow_min,1,'first')+2; end
fprintf('elbow (aux) = row %d  -> temperature %.3f\n', aux, par.mintemp + aux*par.tempstep);
fprintf('\nrow  temp   largest  prop(next/this)  next clusters (cols 6..12 of tree)\n');
for r = 1:min(size(tree,1),26)
    p = NaN; if r <= numel(prop), p = prop(r); end
    fprintf('%3d  %.3f  %6d   %6.3f      %s\n', r, tree(r,2), tree(r,5), p, mat2str(tree(r,6:min(12,end))));
end
tr = tree(1:end-1,5:end); dt = diff(tr);
min_clus = par.min_clus;
if ~isempty(aux) && aux > 2, dt_rel = dt(1:aux-2,:); else, dt_rel = dt; end
if sum(dt_rel(:,2:end) > min_clus,'all') == 0, min_clus = max(dt_rel(:,2:end),[],'all') - 1; end
fprintf('\nmin_clus used = %g\n', min_clus);
clus = zeros(size(tr)); clus(tr >= min_clus) = 1;
clus = clus & [ones(size(clus(1,:))); dt > min_clus];
cand = clus;
fprintf('\ncandidate peaks before the elbow cut (row = tree row, col = cluster # in tree cols 5..):\n');
[r, c] = find(cand);
for i = 1:numel(r)
    tag = ''; if ~isempty(aux) && r(i) >= aux, tag = '  <-- at/after elbow: REMOVED by clus(aux:end,:)=0'; end
    fprintf('  row %2d temp %.3f  cluster %d  size %d  grew by %d%s\n', r(i), tr(r(i),1)*0 + tree(r(i),2), c(i), tr(r(i),c(i)), dt(max(r(i)-1,1),c(i)) * (r(i)>1), tag);
end
[clust_num, temp, auto_sort] = find_temp(tree, clu, par);
fprintf('\nfind_temp result: temps %s, clusters %s, elbow %d\n', mat2str(temp'), mat2str(clust_num'), auto_sort.elbow);
