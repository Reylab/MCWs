function par = par_cb_lfp() 
%just for one nsp
par.device_ix = 1; %defaul devices index 
par.devices = [...
    struct('name','NSP1 (192.168.137.3)','ip','192.168.137.3',...
        'type','BRK','parallel',false,'instance',0,'auto_sel','\w0[12]-\w'),...
    struct('name','NSP2 (192.168.137.178)','ip','192.168.137.178',...
        'type','BRK','parallel',false,'instance',1,'auto_sel','^m'),...
    struct('name','RIPPLE','ip','-',...
        'type','RIP','parallel',false,'instance',0,'auto_sel','^(m[RL].*)(?<!ref-\d*)$'),...
    struct('name','NCx','ip', 'selected folder',...
        'type','NCx','parallel',false,'instance',0,'auto_sel','^m')];

par.use_parallel = false;
par.cbmex_close_ok = true;
par.instance = 0; 
% par.mapfile = '/home/user/share/experimental_files/rec_metadata/test.map';
par.mapfile = '/home/user/share/experimental_files/rec_metadata/MCW-FH_012.map';
par.channels = 'all'; %vector of channels ID or all to use all channels
par.ffftlength = 2^ceil(log2(30000*2))/30000; %seconds it will be increased to use a power of 2 
%^this has to be the largest one in the used freqs
par.freqs_notch = [60 120]; 
par.notch_width = 1;
%plotting
par.n_blocks  = 5;

%custom notchs
par.k_bartlett = 50;
par.db_offset4thr = 5;

%plot continuos
par.plot_cont_sec = 90;
par.detect_sign = 'neg'; % 'neg', 'pos' or 'abs'
par.thr = 5;

%gui plots and basic filters
par.x_power_manual = struct;
par.x_power_manual.('f500').min = 0;
par.x_power_manual.('f500').max =  250;
par.x_power_manual.('f500').min_zoom = 0;
par.x_power_manual.('f500').max_zoom = 200;

par.x_power_manual.('f1000').min = 0;
par.x_power_manual.('f1000').max =  500;
par.x_power_manual.('f1000').min_zoom = 0;
par.x_power_manual.('f1000').max_zoom = 200;

par.x_power_manual.('f2000').min = 0;
par.x_power_manual.('f2000').max =  1000;
% par.x_power_manual.('f2000').max =  10000;
par.x_power_manual.('f2000').min_zoom = 0;
par.x_power_manual.('f2000').max_zoom = 200;

par.x_power_manual.('f10000').min = 0;
par.x_power_manual.('f10000').max =  3000;
par.x_power_manual.('f10000').min_zoom = 0;
par.x_power_manual.('f10000').max_zoom = 300;

par.x_power_manual.('f7500').min = 0;
par.x_power_manual.('f7500').max =  3000;
par.x_power_manual.('f7500').min_zoom = 0;
par.x_power_manual.('f7500').max_zoom = 300;

par.x_power_manual.('f30000').min = 0;
par.x_power_manual.('f30000').max =  10000;
par.x_power_manual.('f30000').min_zoom = 0;
par.x_power_manual.('f30000').max_zoom = 300;

%filters
par.custom_filter = struct;
par.custom_filter.('f500').enable = false;
par.custom_filter.('f500').bp1 = 2;
par.custom_filter.('f500').bp2 =  100;
par.custom_filter.('f500').order =  2;
par.custom_filter.('f500').filter_type =  'ellip_order'; %'ellip_order' or 'ellip_stop_band'
par.custom_filter.('f500').fstop1 = 0.5;
par.custom_filter.('f500').fstop2 = 130;
par.custom_filter.('f500').Rp = 0.07;
par.custom_filter.('f500').Rs = 20;

par.custom_filter.('f1000').enable = false;
par.custom_filter.('f1000').bp1 = 2;
par.custom_filter.('f1000').bp2 =  200;
par.custom_filter.('f1000').enable = false;
par.custom_filter.('f1000').filter_type =  'ellip_order'; %'ellip_order' or 'ellip_stop_band'
par.custom_filter.('f1000').fstop1 = 0.5;
par.custom_filter.('f1000').fstop2 = 260;
par.custom_filter.('f1000').Rp = 0.07;
par.custom_filter.('f1000').Rs = 20;

par.custom_filter.('f2000').enable = true;
par.custom_filter.('f2000').bp1 = 1;
par.custom_filter.('f2000').bp2 =  120;
par.custom_filter.('f2000').order =  2;
par.custom_filter.('f2000').filter_type =  'ellip_order'; %'ellip_order' or 'ellip_stop_band'
par.custom_filter.('f2000').fstop1 = 0.5;
par.custom_filter.('f2000').fstop2 = 800;
par.custom_filter.('f2000').Rp = 0.07;
par.custom_filter.('f2000').Rs = 20;

par.custom_filter.('f7500').enable = 1;
par.custom_filter.('f7500').bp1 = 2;
par.custom_filter.('f7500').bp2 =  3000;
par.custom_filter.('f7500').order =  2;
par.custom_filter.('f7500').filter_type =  'ellip_order'; %'ellip_order' or 'ellip_stop_band'
par.custom_filter.('f7500').fstop1 = 0.5;
par.custom_filter.('f7500').fstop2 = 4000;
par.custom_filter.('f7500').Rp = 0.07;
par.custom_filter.('f7500').Rs = 20;

par.custom_filter.('f10000').enable = false;
par.custom_filter.('f10000').bp1 = 2;
par.custom_filter.('f10000').bp2 =  3000;
par.custom_filter.('f10000').order =  2;
par.custom_filter.('f10000').filter_type =  'ellip_order'; %'ellip_order' or 'ellip_stop_band'
par.custom_filter.('f10000').fstop1 = 0.5;
par.custom_filter.('f10000').fstop2 = 4000;
par.custom_filter.('f10000').Rp = 0.07;
par.custom_filter.('f10000').Rs = 20;

par.custom_filter.('f30000').enable = 1;
par.custom_filter.('f30000').bp1 = 300;
par.custom_filter.('f30000').bp2 =  3000;
par.custom_filter.('f30000').order =  2;
par.custom_filter.('f30000').filter_type =  'ellip_order'; %'ellip_order' or 'ellip_stop_band'
par.custom_filter.('f30000').fstop1 = 0.5;
par.custom_filter.('f30000').fstop2 = 4000;
par.custom_filter.('f30000').Rp = 0.07;
par.custom_filter.('f30000').Rs = 20;

end
 
