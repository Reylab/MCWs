function labels = msort(spikes, branch_depth,npca,min2split)
    if ~exist('branch_depth','var'), branch_depth=2; end
    if ~exist('npca','var'), npca=10; end
    if ~exist('min2split','var'), min2split = 20; end
    full_npca = min(npca*2, size(spikes,2) );
    spikes = spikes';
    n_spk2pca = min(size(spikes,2),2000);
    [U,~,~] = svd(spikes(:,randperm(size(spikes,2),n_spk2pca)));
    coef = U(:,1:full_npca)'*(spikes);
    if isa(spikes, 'single')
        coef = double(coef);
    end
    labels = branch_cluster(coef,branch_depth,npca,min2split);
        

end

function labels_new = branch_cluster(features, branch_depth,npca,min2split)
    %feaures #features  x #samples
    if numel(features) == 0
        labels_new = [];
        return
    end
    n_spk2pca = min(size(features,2),1000);
    npca = min(npca,size(features,1));
    %
    [U,~,~] = svd(features(:,randperm(size(features,2),n_spk2pca)));
    coef = U(:,1:npca)'*(features);
    
    labels1=isosplit5_mex(coef);
    if min(labels1)<0
        error('Unexpected error in isosplit5.')
    end
    K= max(labels1);
    if K<=1 || branch_depth<=1
        labels_new = labels1;
        return 
    end
    label_offset = 0;
    labels_new = zeros(size(labels1));
    for k = 1:K
        inds_k = find(labels1==k);
        if size(inds_k,2) > min2split
            labels_k = branch_cluster(features(:,inds_k), branch_depth-1,npca,min2split);
            K_k = max(labels_k);
            labels_new(inds_k)=label_offset+labels_k;
            label_offset = label_offset + K_k;
        else
            labels_new(inds_k) = label_offset+1;
            label_offset = label_offset + 1;
        end
    end
end