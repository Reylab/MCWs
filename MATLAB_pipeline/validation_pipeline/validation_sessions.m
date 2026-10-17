function cfg = validation_sessions(id)
% VALIDATION_SESSIONS  Every validation run as one row; returns the cfg for
% run_validation.
%
%   run_validation('pt11_ch328_second_pass')      % run_validation looks the id up
%   cfg = validation_sessions('pt11_ch328_second_pass');
%   validation_sessions()                         % list every id
%
% Row: id | channel folder | ch | stim list | reference | arm(s) | output folder | flags
%   reference / arm: {name, subfolder, times file, grapes file, class}
%                    (arm class [] = matched to the reference class by spike time)
%   flags: 'detections' -> cfg.venn_detections (all-detections venn)
%          'channel'    -> cfg.channel_subsets (det_* subsets)
% Shared settings for every run: tol 0.5 ms, min_recall 0.5, dyn_buffer 30 ms,
% x_jitter 0.12.

SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel/';
t368 = 'times_mRFUS07 raw_368.mat'; t328 = 'times_mLFUS08 raw_328.mat';
gb = 'grapes_blanks.mat'; gr = 'grapes_blanks_rescue.mat';
s368 = [410 330 532 776 492 496 72 124 951 130 166 273 585 35 533 16 542 3 490 180];
s328 = [76 386 858 69 391 875 139 1015 513];
s12  = [215 198 209 211 210];
gt368 = {'gt', 'hand selected', t368, gb, 3};
gt328 = {'gt', 'hand selected', t328, gb, 1};
wc368 = {'wave_clus', 'wave_clus_20261008', t368, gb, 1};
wc328 = {'wave_clus', 'wave_clus_20261008', t328, gb, 1};
rec = @(t) {'recluster_rescued', 'python_recluster_20261007_rescued', t, gr, []};
gt  = @(t, c) {'gt', 'hand selected', t, gb, c};
rec13 = @(t) {'recluster_rescued', 'python_recluster_20261013_rescued', t, gr, []};
rec14 = @(t) {'recluster_rescued', 'python_recluster_20261014_rescued', t, gr, []};   % 2026-10-14 python runs   % 2026-10-13 python runs (pt6 emu7, pt9 emu3)
t334 = 'times_mLFUS05 raw_334.mat'; t337 = 'times_mLFUS08 raw_337.mat';

R = {
% id                                folder                        ch   stims  reference  arms                                                                     output folder                                    flags
'pt10_ch368_sd1_sd2',               'pt 10 emu 7 ch 368',         368, s368, gt368, {{'python_sd1', 'python_rescue_test_sd1', t368, gr, []}, {'python_sd2', 'python_rescue_test_sd2', t368, gr, []}}, 'validation_gt_vs_python_sd1_sd2', {}
'pt10_ch368_norefract',             'pt 10 emu 7 ch 368',         368, s368, gt368, {{'python_norefract_sd1', 'python_rescue_no_refract_sd1', t368, gr, []}, {'python_norefract_sd2', 'python_rescue_no_refract_sd2', t368, gr, []}}, 'test_resp_gt_vs_python_sd1_sd2', {}
'pt10_ch368_norefract_val',         'pt 10 emu 7 ch 368',         368, s368, gt368, {{'python_norefract_sd1', 'python_rescue_no_refract_sd1', t368, gr, []}, {'python_norefract_sd2', 'python_rescue_no_refract_sd2', t368, gr, []}}, 'validation_gt_vs_python_norefract_sd1_sd2', {}
'pt10_ch368_recluster_sd1',         'pt 10 emu 7 ch 368',         368, s368, gt368, {{'python_recluster_sd1', 'python_recluster_20261006_sd1', t368, gr, []}}, 'validation_gt_vs_python_recluster_20261006_sd1', {'detections', 'channel'}
'pt10_ch368_gt_vs_wave_clus',       'pt 10 emu 7 ch 368',         368, s368, gt368, {{'wave_clus', 'wave_clus_20261008', t368, gb, []}},                       'validation_gt_vs_wave_clus_20261008', {'detections'}
'pt10_ch368_gt_vs_wave_clus_forced', 'pt 10 emu 7 ch 368',         368, s368, gt368, {{'wave_clus_forced', 'wave_clus_forced_20261009', t368, gb, []}},         'validation_gt_vs_wave_clus_forced_20261009', {'detections', 'channel'}
'pt10_ch368_wave_clus_forced_vs_python', 'pt 10 emu 7 ch 368',    368, s368, {'wave_clus_forced', 'wave_clus_forced_20261009', t368, gb, 3}, {{'python_recluster_sd1', 'python_recluster_20261006_sd1', t368, gr, []}}, 'validation_wave_clus_forced_vs_python_sd1_20261009', {'detections', 'channel'}
'pt10_ch368_wave_clus_vs_python', 'pt 10 emu 7 ch 368',         368, s368, wc368, {{'python_recluster_sd1', 'python_recluster_20261006_sd1', t368, gr, []}}, 'validation_wave_clus_vs_python_sd1_20261008', {'detections', 'channel'}
'pt11_ch328_second_pass',           'pt 11 emu 17 ch 328',        328, s328, gt328, {{'recluster_second_pass', 'python_recluster_20261008_second_pass', t328, gr, []}}, 'validation_gt_vs_python_recluster_20261008_second_pass', {'detections', 'channel'}
'pt11_ch328_norefract',             'pt 11 emu 17 ch 328',        328, s328, gt328, {{'python_norefract_sd1', 'python_rescue_no_refract_sd1', t328, gr, []}, {'python_norefract_sd2', 'python_rescue_no_refract_sd2', t328, gr, []}}, 'validation_gt_vs_python_norefract_sd1_sd2', {}
'pt11_ch328_gt_vs_wave_clus',       'pt 11 emu 17 ch 328',        328, s328, gt328, {{'wave_clus', 'wave_clus_20261008', t328, gb, []}},                       'validation_gt_vs_wave_clus_20261008', {'detections', 'channel'}
'pt11_ch328_wave_clus_vs_python',   'pt 11 emu 17 ch 328',        328, s328, wc328, {{'recluster_second_pass', 'python_recluster_20261008_second_pass', t328, gr, []}}, 'validation_wave_clus_vs_python_sd1_20261008', {'detections', 'channel'}
'pt12_ch299',                       'pt 12 emu 18 ch 299',        299, s12,  gt('times_mRFUS02 raw_299.mat', 1), {rec('times_mRFUS02 raw_299.mat')}, 'validation_gt_vs_python_recluster_20261007', {}
'pt12_ch300_class1',                'pt 12 emu 18 ch 300',        300, [94 24 12 22 16 1 8 85 13 80], gt('times_mRFUS03 raw_300.mat', 1), {rec('times_mRFUS03 raw_300.mat')}, 'validation_gt_vs_python_recluster_20261007_class1', {}
'pt12_ch300_class2',                'pt 12 emu 18 ch 300',        300, [94 24 12 22 16 1 8 85 13 80], gt('times_mRFUS03 raw_300.mat', 2), {rec('times_mRFUS03 raw_300.mat')}, 'validation_gt_vs_python_recluster_20261007_class2', {}
'pt12_ch305',                       'pt 12 emu 18 ch 305',        305, s12,  gt('times_mRFUS08 raw_305.mat', 1), {rec('times_mRFUS08 raw_305.mat')}, 'validation_gt_vs_python_recluster_20261007', {}
'pt4_ch259',                        'pt 4 emu 7 ch 259',          259, [228 294 163 45 155 60 84 122 131 79 106 267 318 178 107 72 283 196 43], gt('times_mLPBTG03 raw_259.mat', 1), {rec('times_mLPBTG03 raw_259.mat')}, 'validation_gt_vs_python_recluster_20261007', {}
'pt6_ch333',                        'pt 6 emu 13 ch 333',         333, [642 372 589 25 82 96 179 293 436 177 98 647 40 285 613 394 582 626 632 113], gt('times_mLFUS04 raw_333.mat', 1), {rec('times_mLFUS04 raw_333.mat')}, 'validation_gt_vs_python_recluster_20261007', {}
'pt9_ch331',                        'pt 9 emu 3 ch 331 check',    331, [96 176 43 324 414 744 482 204 785 250 236 12 78 95 503 177 125 196 208 230], {'gt', 'hand selected rebuilt 4class', 'times_mLFUS02 raw_331.mat', gb, 4}, {rec('times_mLFUS02 raw_331.mat')}, 'validation_gt_vs_python_recluster_20261007', {}
'pt9_ch335',                        'zneed rerun grapes blanks changednew trial pt 9 emu 3 ch 335', 335, [288 156 272 278 55 147 71 12 64 282 146 25 7 228 83 615 275 482 63 46], gt('times_mLFUS06 raw_335.mat', 2), {rec('times_mLFUS06 raw_335.mat')}, 'validation_gt_vs_python_recluster_20261007', {}
'pt6emu7_ch334_class1',             'pt 6 emu 7 ch 334',          334, [242 266 184 191 203 201 7 258 8 9 255 10 241 408 246 6 227 182 192 197], gt(t334, 1), {rec13(t334)}, 'validation_gt_vs_python_recluster_20261013_class1', {}
'pt6emu7_ch334_class2',             'pt 6 emu 7 ch 334',          334, [255 66 408 258 64 34 148 162 164 172 143 145 146 147 149 158 159 160 161 163], gt(t334, 2), {rec13(t334)}, 'validation_gt_vs_python_recluster_20261013_class2', {}
'pt6emu7_ch334_class3',             'pt 6 emu 7 ch 334',          334, [2 408 143 145 146 147 148 149 158 159 160 161 162 163 164 167 168 169 170], gt(t334, 3), {rec13(t334)}, 'validation_gt_vs_python_recluster_20261013_class3', {}
'pt9emu3_ch337_class3',             'pt 9 emu 3 ch 337',          337, [531 175 436 696 16 25 43 130 744 613 228 324 565 84 356 98 71 110 727 134], gt(t337, 3), {rec13(t337)}, 'validation_gt_vs_python_recluster_20261013_class3', {}
'pt9emu3_ch337_class4',             'pt 9 emu 3 ch 337',          337, [418 730 18 157 450 131 132 204 472 724 436 684 611 468 39 406 96 162 696 98], gt(t337, 4), {rec13(t337)}, 'validation_gt_vs_python_recluster_20261013_class4', {}
% sessions from the 2026-10-13 list (GT class + stimuli per unit); arms {} until the python sort is converted
'pt10emu29_ch291_class2',            'pt 10 emu 29 ch 291',   291, [69 64 52 59 61 71 60 50 54 68 151 53 58 55 62 165 168 66 146 65], gt('times_mLFUS03 raw_291.mat', 2), {rec13('times_mLFUS03 raw_291.mat')}, 'validation_gt_vs_python_recluster_20261013_class2', {}
'pt10emu29_ch296_class2',            'pt 10 emu 29 ch 296',   296, [107 101 102 103 118 106 108 111 98 110 112 114 131 109 155 104 97 117 167 56], gt('times_mLFUS08 raw_296.mat', 2), {{'recluster_rescued', 'python_recluster_20261014_rescued', 'times_mLFUS08 raw_296.mat', gr, []}}, 'validation_gt_vs_python_recluster_20261014_class2', {}
'pt10emu29_ch368_class1',            'pt 10 emu 29 ch 368',   368, [187 173 179 152 206 54 62 155 144 193 165 163 175 178 160 72 210 136 55 143], gt('times_mRFUS07 raw_368.mat', 1), {{'recluster_rescued', 'python_recluster_20261014_rescued', 'times_mRFUS07 raw_368.mat', gr, []}}, 'validation_gt_vs_python_recluster_20261014_class1', {}
'pt11emu17_ch321_class2',            'pt 11 emu 17 ch 321',   321, [157 810 149 79 150 577 256 519 545 988 943 218 104 147 631 140 390 937 650 913], gt('times_mLFUS01 raw_321.mat', 2), {rec13('times_mLFUS01 raw_321.mat')}, 'validation_gt_vs_python_recluster_20261013_class2', {}
'pt11emu17_ch321_class5',            'pt 11 emu 17 ch 321',   321, [42 358 761 131 946 395 782 34 55 380 6 41 106 149 142 157 224 519 476 12], gt('times_mLFUS01 raw_321.mat', 5), {rec13('times_mLFUS01 raw_321.mat')}, 'validation_gt_vs_python_recluster_20261013_class5', {}
'pt11emu17_ch323_class3',            'pt 11 emu 17 ch 323',   323, [18 989 628 629 20 24 175 182 135 4 873 215 423 7 22 749 643 875 1002 123], gt('times_mLFUS03 raw_323.mat', 3), {rec14('times_mLFUS03 raw_323.mat')}, 'validation_gt_vs_python_recluster_20261014_class3', {}
'pt11emu17_ch324_class4',            'pt 11 emu 17 ch 324',   324, [150 157 79 104 149 256 147 545 631 943 988 899 6 82 116 387 483 519 577 755], gt('times_mLFUS04 raw_324.mat', 4), {rec14('times_mLFUS04 raw_324.mat')}, 'validation_gt_vs_python_recluster_20261014_class4', {}
'pt11emu17_ch326_class1',            'pt 11 emu 17 ch 326',   326, [905 85 95 73 104 645 163 38 784 172 926 604 60 94 81 461 97 164 111 98], gt('times_mLFUS06 raw_326.mat', 1), {rec14('times_mLFUS06 raw_326.mat')}, 'validation_gt_vs_python_recluster_20261014_class1', {}
'pt11emu17_ch327_class1',            'pt 11 emu 17 ch 327',   327, [1000 119 123 702 122 81 63 111 37 108 97 74 130 386 86 98 749 513 604 865], gt('times_mLFUS07 raw_327.mat', 1), {rec14('times_mLFUS07 raw_327.mat')}, 'validation_gt_vs_python_recluster_20261014_class1', {}
'pt12emu14_ch300_class1',            'pt 12 emu 14 ch 300',   300, [64 168 124 85 144 77 176 193 71 248 604 167 218 167 76 255 74 246 8 185], gt('times_mRFUS03 raw_300.mat', 1), {rec14('times_mRFUS03 raw_300.mat')}, 'validation_gt_vs_python_recluster_20261014_class1', {}
'pt12emu14_ch300_class2',            'pt 12 emu 14 ch 300',   300, [123 526 35 254 202 291 278 158 1090 63 87 128 162 95 40 64 58 52 55 81], gt('times_mRFUS03 raw_300.mat', 2), {rec14('times_mRFUS03 raw_300.mat')}, 'validation_gt_vs_python_recluster_20261014_class2', {}
'pt12emu18_ch290_class5',            'pt 12 emu 18 ch 290',   290, [236 226 227 233 237 217 219 221 222 232], gt('times_mLFUS02 raw_290.mat', 5), {{'recluster_rescued', 'python_recluster_20261014_rescued', 'times_mLFUS02 raw_290.mat', gr, []}}, 'validation_gt_vs_python_recluster_20261014_class5', {}
'pt12emu18_ch294_class3',            'pt 12 emu 18 ch 294',   294, [187 180 169 177 181 175 191 185 172 190], gt('times_mLFUS06 raw_294.mat', 3), {{'recluster_rescued', 'python_recluster_20261014_rescued', 'times_mLFUS06 raw_294.mat', gr, []}}, 'validation_gt_vs_python_recluster_20261014_class3', {}
'pt13emu11_ch290_class3',            'pt 13 emu 11 ch 290',   290, [194 32 208 165 48 207 182 34 146 216 193 236 5 7 26 27 29 35 41 45], gt('times_mRFUS02 raw_290.mat', 3), {rec14('times_mRFUS02 raw_290.mat')}, 'validation_gt_vs_python_recluster_20261014_class3', {}
'pt14emu7_ch291_class1',             'pt 14 emu 7 ch 291',    291, [217 157 177 67 37 52 144 159 240 189 237 60 87 117 147 197 199 213 35 54], gt('times_mLFUS03 raw_291.mat', 1), {{'recluster_rescued', 'python_recluster_20261014_rescued', 'times_mLFUS03 raw_291.mat', gr, []}}, 'validation_gt_vs_python_recluster_20261014_class1', {}
'pt14emu7_ch363_class3',             'pt 14 emu 7 ch 363',    363, [49 72 180 179 80 149 153 199 170 139 63 70 210 220 132 26 33 109 163 3], gt('times_mRFUS02 raw_363.mat', 3), {rec13('times_mRFUS02 raw_363.mat')}, 'validation_gt_vs_python_recluster_20261013_class3', {}
'pt14emu7_ch364_class5',             'pt 14 emu 7 ch 364',    364, [229 19 69 159 39 49 89 119 9 29 109 149 239 59 139 169 219 99 199 129], gt('times_mRFUS03 raw_364.mat', 5), {{'recluster_rescued', 'python_recluster_20261014_rescued', 'times_mRFUS03 raw_364.mat', gr, []}}, 'validation_gt_vs_python_recluster_20261014_class5', {}
'pt14emu7_ch369_class3',             'pt 14 emu 7 ch 369',    369, [72 180 163 40 69 21 43 80 121 204 236 1 2 3 4 5 6 7 8 9], gt('times_mRFUS08 raw_369.mat', 3), {{'recluster_rescued', 'python_recluster_20261014_rescued', 'times_mRFUS08 raw_369.mat', gr, []}}, 'validation_gt_vs_python_recluster_20261014_class3', {}
'pt4emu7_ch322_class3',              'pt 4 emu 7 ch 322',     322, [539 281 158], gt('times_mLHIPH02 raw_322.mat', 3), {rec14('times_mLHIPH02 raw_322.mat')}, 'validation_gt_vs_python_recluster_20261014_class3', {}
'pt5emu4_ch271_class2',              'pt 5 emu 4 ch 271',     271, [40 28 35 197 230 25 120 209 523 513 114 119 622 279 239 806 294 234 123 878], gt('times_mLPBTG06 raw_271.mat', 2), {rec14('times_mLPBTG06 raw_271.mat')}, 'validation_gt_vs_python_recluster_20261014_class2', {}
'pt6emu13_ch262_class3',             'pt 6 emu 13 ch 262',    262, [25 632 395], gt('times_mRAMY06 raw_262.mat', 3), {rec14('times_mRAMY06 raw_262.mat')}, 'validation_gt_vs_python_recluster_20261014_class3', {}
'pt6emu13_ch264_class3',             'pt 6 emu 13 ch 264',    264, [615 44 842], gt('times_mRAMY08 raw_264.mat', 3), {rec14('times_mRAMY08 raw_264.mat')}, 'validation_gt_vs_python_recluster_20261014_class3', {}
'pt6emu13_ch337_class2',             'pt 6 emu 13 ch 337',    337, [615 66 823 516 239 265 408 6 44 524 643 842 87 195 745 37 368 384 581 270], gt('times_mLFUS08 raw_337.mat', 2), {rec14('times_mLFUS08 raw_337.mat')}, 'validation_gt_vs_python_recluster_20261014_class2', {}
'pt6emu13_ch334_class3',             'pt 6 emu 13 ch 334',    334, [133 418 404 717 39 180 3 107 53 265 60 175 134 132 7], gt('times_mLFUS05 raw_334.mat', 3), {rec14('times_mLFUS05 raw_334.mat')}, 'validation_gt_vs_python_recluster_20261014_class3', {}
'pt6emu13_ch334_class4',             'pt 6 emu 13 ch 334',    334, [736 839 744 159 239 366 9 8 214 48 7 11 10 350 524 29 642 270 331 486], gt('times_mLFUS05 raw_334.mat', 4), {rec14('times_mLFUS05 raw_334.mat')}, 'validation_gt_vs_python_recluster_20261014_class4', {}
};

if nargin < 1
    fprintf('%s\n', R{:, 1});
    cfg = R(:, 1);
    return
end
r = R(strcmp(R(:, 1), id), :);
assert(~isempty(r), 'validation_sessions: unknown id ''%s'' -- validation_sessions() lists them', id);
cfg.root = [SC r{2}];
cfg.ch = r{3};
cfg.stim_list = r{4};
cfg.gt = unit(cfg.root, r{5});
a = cellfun(@(x) unit(cfg.root, x), r{6});
cfg.arms = a;
cfg.out_dir = fullfile(cfg.root, r{7});
cfg.tol        = 0.5;   % ms, spike-time match tolerance
cfg.min_recall = 0.5;   % match_cc threshold for calling an arm class "matched"
cfg.dyn_buffer = 30;    % ms on each side of onset..onset+dura (green window lines)
cfg.x_jitter   = 0.12;  % half-width of x jitter on the GT-vs-ARM paired-line panels
if any(strcmp(r{8}, 'detections')), cfg.venn_detections = true; end
if any(strcmp(r{8}, 'channel')),    cfg.channel_subsets = true; end
end

function u = unit(root, c)
    u = struct('name', c{1}, 'dir', fullfile(root, c{2}), 'times_file', c{3}, 'grapes_file', c{4}, 'class', c{5});
end
