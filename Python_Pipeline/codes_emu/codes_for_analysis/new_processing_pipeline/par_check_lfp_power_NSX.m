function par=par_check_lfp_power_NSX()
    % call update_lfp_par for each freq (30000,5000,etc) and
    % channel type ('analog_input','front_end')
    % if the fitting, custom filter and/or notch is not configured will be disable
    par = [
%     lfp_par(30000,'analog_input','notch_q',60,'freqs_lim_zoom',[0 300]);    
%     lfp_par(30000,'front_end','notch_width',1,'freqs_lim_zoom',[0 300],...
%         'filter_pass',[300 3000],'filter_order',4,'freqs_fit',[1 300 3000],...
%         'threshold',4.5,'freqs_lim',[0 7500]);        
%     lfp_par(2000,'analog_input','freqs_lim_zoom',[0 300],'notch_q',60);
%     lfp_par(2000,'front_end','notch_q',60,'filter_pass',[3 8],...
%         'freqs_lim_zoom',[0 300],'filter_stop',[0.5 20],'freqs_lim',[0 1000],...
%         'freqs_fit',[1 300 1000]);  ]

   lfp_par(30000,'analog_input','freqs_lim',[0 4000],'freqs_lim_zoom',[0 300]);
    
    lfp_par(30000,'front_end','notch_width',1,'freqs_lim_zoom',[0 300],...
        'filter_pass',[300 3000],'filter_order',2,'freqs_fit',[1 300 1000 3000],... %'freqs_fit',[1 300 3000],...
        'threshold',5,'freqs_lim',[0 10000]);
%     lfp_par(30000,'front_end','notch_width',1,'freqs_lim_zoom',[0 300],...
%         'filter_pass',[1 600],'filter_order',2,'freqs_fit',[1 300 1000],... %'freqs_fit',[1 300 3000],...
%         'threshold',5,'freqs_lim',[0 3000]);
    lfp_par(2000,'analog_input','freqs_lim_zoom',[0 300],'notch_q',60);
        
%     lfp_par(2000,'front_end','notch_width',3,'filter_pass',[3 8],...
%         'freqs_lim_zoom',[0 300],'filter_stop',[0.5 20],'freqs_lim',[0 1000],...
%         'freqs_fit',[1 300 1000]); 

        lfp_par(2000,'front_end','notch_width',1,...
        'filter_pass',[1 600],'freqs_lim_zoom',[0 300],'filter_order',2,'freqs_lim',[0 1000],...
        'freqs_fit',[1 300 1000]);  
        lfp_par(7500,'front_end','notch_width',1,...
        'filter_pass',[1 120],'freqs_lim_zoom',[0 300],'filter_order',2,'freqs_lim',[0 1000],...
        'freqs_fit',[1 300 1000 3000]); 
%         lfp_par(2000,'front_end','notch_width',1,...
%         'filter_pass',[100 500],'freqs_lim_zoom',[0 300],'filter_order',2,'freqs_lim',[0 1000],...
%         'freqs_fit',[1 300 1000]);  
    lfp_par(10000,'front_end','notch_width',1,...
        'filter_pass',[1 600],'freqs_lim_zoom',[0 300],'filter_order',2,'freqs_lim',[0 3000],...
        'freqs_fit',[1 300 1000],'threshold',5);  
    ];
end

function new_table = lfp_par(freq,chtype,varargin)

    p = inputParser;
    p.addParameter('freqs_lim_zoom',[0 300]);
    p.addParameter('freqs_lim',[0 1000]);
    p.addParameter('freqs_fit',[]); %[2 300 3000]
    p.addParameter('filter_pass',[]); %[2 3000]
    p.addParameter('filter_order',[]); %4
    p.addParameter('filter_stop',[]); %[0.5 4000]
    p.addParameter('filter_Rs',40);
    p.addParameter('filter_Rp',0.1);
    p.addParameter('notch_q',[]) % 30
    p.addParameter('notch_width',[]); %1 Hz
    p.addParameter('threshold',[]); %4.5 Hz
    p.parse(varargin{:});
    par = p.Results;
    
    if strcmp(chtype,'analog_input')
        unit = 'mV';
    elseif strcmp(chtype,'front_end')
        unit = 'uV';
    else
        error('unrecognized chtype')
    end
    if ~isempty(par.filter_order) && ~isempty(par.filter_stop)
        error('multiple filter types configured')
    end
    if ~isempty(par.filter_pass) && isempty(par.filter_order) && isempty(par.filter_stop)
        error('filter requires more parameters')
    end
    
    par.unit = unit;
    par.freq = freq;
    new_table = struct2table(par,'AsArray',true);
    new_table.Properties.RowNames = {[num2str(freq) '_' unit]};
end