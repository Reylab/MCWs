import numpy as np
class par():
    def __init__(self, fast_analysis=False, micros = True, which_system_micro = 'RIP',
                 nowait = False, remove_ref_chs = [],do_power_plot = True,notchfilter = True, 
                 do_sorting = True, do_loop_plot = True, extract_events = True, extra_stims_win = False,
                 is_online = False, plot_best_stims_wins = False, copy2miniscrfolder = False,
                 show_sel_count = False,show_best_stims_wins = False, best_stims_nwins = 8,ch_grapes_nwins = 2,
                 max_spikes_plot = 5000,**kwargs):
        self.fast_analysis = fast_analysis 
        self.micros = micros
        self.which_system_micro = which_system_micro
        self.nowait= nowait
        self.remove_ref_chs = remove_ref_chs
        self.do_power_plot = do_power_plot
        self.notchfilter = notchfilter
        self.do_sorting = do_sorting
        self.do_loop_plot = do_loop_plot
        self.extract_events = extract_events
        self.extra_stims_win = extra_stims_win
        self.is_online = is_online
        self.plot_best_stims_wins = plot_best_stims_wins
        self.copy2miniscrfolder =copy2miniscrfolder
        self.show_sel_count = show_sel_count
        self.show_best_stims_wins = show_best_stims_wins
        self.best_stims_nwins =best_stims_nwins
        self.ch_grapes_nwins = ch_grapes_nwins
        self.max_spikes_plot = max_spikes_plot
        if self.fast_analysis == True:
            self.fast_analysis = fast_analysis 
            self.micros = False
            self.which_system_micro = which_system_micro
            self.nowait= nowait
            self.remove_ref_chs = remove_ref_chs
            self.do_power_plot = do_power_plot
            self.notchfilter = notchfilter
            self.do_sorting = do_sorting
            self.do_loop_plot = False
            self.extract_events = False
            self.extra_stims_win = extra_stims_win
            self.is_online = is_online
            self.plot_best_stims_wins = plot_best_stims_wins
            self.copy2miniscrfolder =copy2miniscrfolder
            self.show_sel_count = show_sel_count
            self.show_best_stims_wins = show_best_stims_wins
            self.best_stims_nwins =best_stims_nwins
            self.ch_grapes_nwins = ch_grapes_nwins
            self.max_spikes_plot = max_spikes_plot
        
        self.segments_length = 5
        self.sr = 30000
        self.preprocessing = False
        self.cont_segment = True
        self.max_spikes_pot = 5000
        self.print2file = True
        self.cont_plot_samples = 100000
        self.to_plot_std = 1
        self.all_classes_ax = 'mean'
        self.plot_feature_states = False
        self.line_freq=60
        self.mintemp = 0.00
        self.maxtemp = .251
        self.tempstep = 0.01
        self.SWCycles = 100
        self.KNearNeighb = 11
        self.min_clus = 20
        self.randomseed = 0
        self.temp_plot = 'log'
        self.c_ov = 0.7
        self.elbow_min = 0.4
        self.tmax = 'all'
        self.tmin = 0
        self.w_pre = 20
        self.w_post = 44
        self.alignment_window = 10
        self.stdmin = 5
        self.stdmax = 50
        self.detect_fmin = 300
        self.detect_fmax = 3000
        self.detect_order =4
        self.sort_fmin = 300
        self.sort_fmax = 3000
        self.sort_order = 2
        self.ref_ms = 1.5
        self.detection = 'neg'
        self.int_factor = 5
        self.interpolation = 'y'
        self.parallel = True
        self.process_info = []

    def update_parameters(self,new_par,par,type,NaN_fill=False):
        detection_params = ['channels','preprocessing','segments_length','sr','tmax'
                            ,'tmin','w_pre','w_post','alignment_window','stdmin','stdmax','detect_fmin'
                            ,'detect_fmax','sort_fmin','sort_fmax','ref_ms','detection'
                            ,'int_factor','interpolation','alignmment_window','sort_order','detect_order','detection_date'
                            ]
        
        clus_params = ['min_inputs','max_inputs','scales','features','template_sdnum',
                       'template_k','template_k_min','template_type','force_feature','match',
                       'max_spk','permut','mintemp','maxtemp','tempstep','SWCycles',
                       'KNearNeighb','min_clus','randomseed','force_auto','sorting_date','elbow_min','c_ov']
        
        batch_ploting_params = ['temp_plot','max_spikes_plot','print2file','plot_feature_stats','line_freq']

        new_par_names = list(vars(new_par).keys())
        load_par_names = list(vars(par).keys())
        if (type== 'detect') or (type == 'relevant'):
            for i, detection_param in enumerate(detection_params):
                if detection_param in load_par_names:
                    field = str(detection_param)
                    exec("new_par.%s = par.%s" %(field,field))
                elif NaN_fill:
                    field = str(detection_param)
                    exec("new_par.%s = None" %field)

        if type == 'batch_plot':
            for i, batch_ploting_param in enumerate(batch_ploting_params):
                if load_par_names in batch_ploting_params:
                    field = str(detection_param)
                    exec("new_par.%s = par.%s" %(field,field))

        if (type == 'clus') or (type == 'relevant'):
            for i, clus_param in enumerate(clus_params):
                field = str(clus_param)
                exec("new_par.%s = par.%s" %(field,field))

        if type == 'none':
            for i, new_par_name in enumerate(new_par_names):
                if not (np.any((new_par_name in clus_params) or np.any(new_par_name in detection_param))):
                    field = str(new_par_name)
                    if new_par_name in load_par_names:
                        exec("new_par.%s = par.%s" % (field,field))
                    else:
                        exec("new_par.%s = []"%field)
        
        if 'ref' in list(vars(par).keys()):
            new_par.ref_ms = float(par.ref) / par.sr * 1000
        
        return new_par