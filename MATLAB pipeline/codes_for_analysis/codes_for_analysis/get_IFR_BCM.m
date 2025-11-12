function [ejex,IFR]= get_IFR_BCM(spikes1,alpha_gauss,sigma_gauss,sample_period,time_pre_ms,time_pos_ms)


half_ancho_gauss = alpha_gauss *sigma_gauss;
N_gauss = 2*round(half_ancho_gauss/sample_period)+1; % Number of points of the gaussian window
ejex = -time_pre_ms-half_ancho_gauss:sample_period:time_pos_ms+half_ancho_gauss;  %should be the same length as spike_timeline
int_window = gausswin(N_gauss, alpha_gauss);
int_window = 1000*int_window/sum(int_window)/sample_period;

if iscell(spikes1)
    lst = numel(spikes1);
    spikes1 = sort(horzcat(spikes1{:}));
else
    lst = size(spikes1,1);
    spikes1=sort(spikes1(:));
    spikes1=spikes1(spikes1~=10000);
end

spikes_tot=spikes1(spikes1 < time_pos_ms+half_ancho_gauss & spikes1 > -time_pre_ms-half_ancho_gauss & spikes1~=0);


spike_timeline = histcounts(spikes_tot,(-time_pre_ms-half_ancho_gauss:sample_period:time_pos_ms+half_ancho_gauss))/lst;
integ_timeline_stim_cut = conv(spike_timeline, int_window, 'same');
IFR =  integ_timeline_stim_cut(find(ejex>-time_pre_ms,1)-1:1:find(ejex>time_pos_ms,1));
ejex=ejex(find(ejex>-time_pre_ms,1)-1:1:find(ejex>time_pos_ms,1));